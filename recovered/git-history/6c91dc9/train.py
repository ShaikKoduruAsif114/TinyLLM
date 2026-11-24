import inspect
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

# Prefer torch.amp.autocast for newer PyTorch, but remain compatible.
# This allows both the newer `torch.amp` and older `torch.cuda.amp` APIs.
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# `torch.amp.autocast` requires a `device_type` argument even when disabled,
# while `torch.cuda.amp.autocast` only accepts `enabled`.
_AUTOMATIC_DEVICE_TYPE = "device_type" in inspect.signature(autocast).parameters

def autocast_context(enabled: bool, device_type: str = "cuda"):
    if _AUTOMATIC_DEVICE_TYPE:
        return autocast(device_type=device_type, enabled=enabled)
    return autocast(enabled=enabled)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from dataloader import create_dataloaders
from model.gpt import GPT
from tokeniser.bpe_tokeniser import BPETokenizer


# ============================================================
# Model configuration
# ============================================================
config: Dict[str, object] = {
    "vocab_size": 4096,
    "block_size": 128,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    "dropout": 0.1,
    "pos_encoding": "rope",
}

# ============================================================
# Training configuration
# ============================================================
batch_size = 32
epochs = 10
learning_rate = 3e-4
weight_decay = 0.01
seed = 42
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
latest_checkpoint = checkpoint_dir / "latest.pt"
best_checkpoint = checkpoint_dir / "best.pt"


def set_seed(seed_value: int) -> None:
    """Set random seeds for reproducible training."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_loss: float,
) -> None:
    """Save training state to a checkpoint file."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> Tuple[int, float]:
    """Load checkpoint state if it exists and return resume state."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    return epoch, best_val_loss


def train_one_epoch(
    model: GPT,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    use_amp: bool,
) -> float:
    """Train the model for one epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    step_count = 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for input_ids, targets in progress:
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(use_amp, device.type):
            logits, loss, _ = model(input_ids, targets)

        if loss is None:
            raise RuntimeError("Model did not return a loss for training batch.")

        if use_amp and scaler is not None:
            # Scale the loss, backpropagate, then unscale before gradient clipping.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        step_count += 1
        progress.set_postfix({"loss": total_loss / step_count})

    return total_loss / max(step_count, 1)


def validate(
    model: GPT,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Evaluate the model on the validation set and return average loss."""
    model.eval()
    total_loss = 0.0
    step_count = 0

    with torch.no_grad():
        for input_ids, targets in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast_context(use_amp, device.type):
                _, loss, _ = model(input_ids, targets)

            if loss is None:
                raise RuntimeError("Model did not return a loss for validation batch.")

            total_loss += loss.item()
            step_count += 1

    return total_loss / max(step_count, 1)


def generate_sample(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
) -> str:
    """Generate a sample text sequence from the model."""
    model.eval()

    token_ids = tokenizer.encode(prompt)
    context = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=50,
    )

    generated_ids = generated_ids[0].cpu().tolist()
    return tokenizer.decode(generated_ids)


def main() -> None:
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    set_seed(seed)

    print("Device:", device)
    print("Using AMP:", use_amp)

    model = GPT(config).to(device)
    print(f"Total model parameters: {model.count_parameters():,}")

    train_loader, val_loader = create_dataloaders()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # GradScaler signature changed in recent PyTorch versions.
    # Use the new device-aware constructor if available.
    scaler_args = {}
    if inspect.signature(GradScaler).parameters.get("device_type") is not None:
        scaler_args["device_type"] = "cuda" if use_amp else "cpu"
    scaler = GradScaler(enabled=use_amp, **scaler_args)

    start_epoch = 1
    best_val_loss = float("inf")

    if latest_checkpoint.exists():
        print(f"Loading checkpoint: {latest_checkpoint}")
        start_epoch, best_val_loss = load_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler,
        )
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}")

    tokenizer = BPETokenizer()
    tokenizer.load("vocab.json", "merges.json")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
        )

        val_loss = validate(
            model,
            val_loader,
            device,
            use_amp,
        )

        perplexity = math.exp(val_loss)
        current_lr = scheduler.get_last_lr()[0]

        tqdm.write("=" * 80)
        tqdm.write(f"Epoch {epoch}/{epochs}")
        tqdm.write(f"Train Loss      : {train_loss:.4f}")
        tqdm.write(f"Validation Loss : {val_loss:.4f}")
        tqdm.write(f"Perplexity      : {perplexity:.4f}")
        tqdm.write(f"Learning Rate   : {current_lr:.6g}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_checkpoint, model, optimizer, scheduler, epoch, best_val_loss)
            tqdm.write(f"Saved new best checkpoint: {best_checkpoint}")

        # Save the latest checkpoint after validation and after best checkpoint updates.
        save_checkpoint(latest_checkpoint, model, optimizer, scheduler, epoch, best_val_loss)

        # Scheduler.step() is called after validation and checkpointing so the current epoch's
        # learning rate is used for both training and validation.
        scheduler.step()

        sample = generate_sample(model, tokenizer, prompt="Once upon a time", device=device)
        tqdm.write("--------------------------------------------------")
        tqdm.write(f"Epoch {epoch}")
        tqdm.write("Prompt:")
        tqdm.write("Once upon a time")
        tqdm.write("")
        tqdm.write("Generated:")
        tqdm.write(sample)
        tqdm.write("--------------------------------------------------")

    tqdm.write("Training complete.")


if __name__ == "__main__":
    main()

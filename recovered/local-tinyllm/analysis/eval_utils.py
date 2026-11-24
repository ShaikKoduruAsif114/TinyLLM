"""Shared utilities for checkpoint evaluation and analysis plots."""

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model.gpt import GPT
from model.positional_encoding import LearnedPositionalEncoding


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = ("rope", "learned", "sinusoidal")


class DeterministicValDataset(Dataset):
    """Non-overlapping, fixed-count validation windows matching the notebook."""

    def __init__(self, data_path, block_size, max_samples=784):
        self.data = np.load(data_path)
        self.block_size = block_size
        self.starts = list(
            range(0, len(self.data) - block_size - 1, block_size)
        )[:max_samples]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        x = self.data[start : start + self.block_size]
        y = self.data[start + 1 : start + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_loader(block_size, batch_size=32):
    dataset = DeterministicValDataset(ROOT / "data" / "val.npy", block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


def load_eval_model(model_name, context_length):
    """Load a trained model with an evaluation-only context length."""
    checkpoint_path = ROOT / "checkpoints" / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train or provide the three *_best.pt files."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = copy.deepcopy(checkpoint["config"])
    config["block_size"] = context_length
    model = GPT(config)

    state = dict(checkpoint["model_state_dict"])
    for key in list(state):
        if key.endswith("attn.bias"):
            del state[key]

    # Keep the trained learned-position table at its original size; the
    # forward pass clamps any evaluation positions beyond that table.
    position_key = "pos_encoding.pos_embed.weight"
    if model_name == "learned" and position_key in state:
        trained_length = state[position_key].shape[0]
        if model.pos_encoding.pos_embed.num_embeddings != trained_length:
            model.pos_encoding = LearnedPositionalEncoding(
                max_seq_len=trained_length,
                n_embd=config["n_embd"],
            )

    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


@torch.no_grad()
def evaluate(model, loader):
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        _, loss, _ = model(inputs, targets)
        tokens = targets.numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens

    if total_tokens == 0:
        raise ValueError("Validation set contains no complete windows.")
    average_loss = total_loss / total_tokens
    return average_loss, math.exp(average_loss)
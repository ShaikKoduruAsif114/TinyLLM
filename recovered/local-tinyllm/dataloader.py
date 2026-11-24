# ============================================================
# dataloader.py
# DataLoader for TinyStories GPT Training
# ============================================================

import torch
from torch.utils.data import DataLoader

from dataset import TinyStoriesDataset


# ============================================================
# Configuration
# ============================================================

BLOCK_SIZE = 128
BATCH_SIZE = 32

TRAIN_PATH = "data/train.npy"
VAL_PATH = "data/val.npy"


# ============================================================
# Build DataLoaders
# ============================================================

def create_dataloaders():

    train_dataset = TinyStoriesDataset(
        TRAIN_PATH,
        BLOCK_SIZE
    )

    val_dataset = TinyStoriesDataset(
        VAL_PATH,
        BLOCK_SIZE
    )

    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

    val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

    return train_loader, val_loader


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader = create_dataloaders()

    print("=" * 60)
    print("Train Loader")
    print("=" * 60)

    x, y = next(iter(train_loader))

    print("Input Shape :", x.shape)
    print("Target Shape:", y.shape)

    print()

    print("First Sample")
    print(x[0][:20])

    print()

    print("First Target")
    print(y[0][:20])

    print()

    print("Next Token Verification")

    print("Input : ", x[0][:10].tolist())
    print("Target:", y[0][:10].tolist())
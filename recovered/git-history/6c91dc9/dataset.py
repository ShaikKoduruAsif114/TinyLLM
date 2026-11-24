# ============================================================
# dataset.py
# TinyStories Dataset for GPT Training
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset


class TinyStoriesDataset(Dataset):
    """
    Dataset for GPT Next-Token Prediction

    Input:
        [t1, t2, t3, t4]

    Target:
        [t2, t3, t4, t5]
    """

    def __init__(self, data_path, block_size):

        self.block_size = block_size

        print("=" * 60)
        print(f"Loading dataset : {data_path}")
        print("=" * 60)

        self.data = np.load(data_path)

        print(f"Loaded {len(self.data):,} tokens\n")

    ############################################################
    # Number of training samples
    ############################################################

    def __len__(self):

        return len(self.data) - self.block_size

    ############################################################
    # Return one training example
    ############################################################

    def __getitem__(self, idx):

        x = self.data[idx : idx + self.block_size]

        y = self.data[idx + 1 : idx + self.block_size + 1]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


# ============================================================
# Test Dataset
# ============================================================

if __name__ == "__main__":

    BLOCK_SIZE = 128

    dataset = TinyStoriesDataset(
        data_path="data/train.npy",
        block_size=BLOCK_SIZE
    )

    print("Dataset Length :", len(dataset))

    x, y = dataset[0]

    print("\nInput Shape :", x.shape)
    print("Target Shape:", y.shape)

    print("\nFirst 20 Input Tokens:")
    print(x[:20])

    print("\nFirst 20 Target Tokens:")
    print(y[:20])

    print("\nVerify Next Token Prediction")

    print("Input : ", x[:10].tolist())
    print("Target:", y[:10].tolist())
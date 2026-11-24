# ============================================================
# dataset.py
# TinyStories Dataset for GPT Training
# ============================================================

import numpy as np
import random
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

    def __init__(self, data_path, block_size, samples_per_epoch=None):

        self.block_size = block_size

        # We load the full token stream once here. Random sampling is used
        # in __getitem__ so that each epoch draws non-overlapping random
        # chunks instead of millions of heavily overlapping stride-1 samples.
        print("=" * 60)
        print(f"Loading dataset : {data_path}")
        print("=" * 60)

        self.data = np.load(data_path)

        print(f"Loaded {len(self.data):,} tokens\n")

        self.samples_per_epoch = (
            len(self.data) // self.block_size
            if samples_per_epoch is None
            else samples_per_epoch
        )

        # Precompute the maximum valid starting index for a block.
        self.max_start_index = len(self.data) - self.block_size - 1

    ############################################################
    # Number of training samples
    ############################################################

    def __len__(self):

        # The length of the dataset now controls how many random samples
        # are drawn per epoch, instead of using a stride-1 window over tokens.
        return self.samples_per_epoch

    ############################################################
    # Return one training example
    ############################################################

    def __getitem__(self, idx):

        # Ignore the incoming idx because sampling is random.
        # This still keeps the Dataset API compatible with DataLoader.
        start = random.randint(0, self.max_start_index)

        x = self.data[start : start + self.block_size]
        y = self.data[start + 1 : start + self.block_size + 1]

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
        block_size=BLOCK_SIZE,
        samples_per_epoch=1000
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
# ============================================================
# prepare_data.py
# Prepare TinyStories Dataset for GPT Training
# ============================================================

from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from tokeniser.bpe_tokeniser import BPETokenizer


# ============================================================
# Configuration
# ============================================================

VOCAB_PATH = "vocab.json"
MERGES_PATH = "merges.json"

TRAIN_OUTPUT = "data/train.npy"
VAL_OUTPUT = "data/val.npy"

VAL_SPLIT = 0.01          # 1% validation

# Number of stories to encode
# None = Use Full TinyStories Dataset
NUM_STORIES = 100000


# ============================================================
# Load Tokenizer
# ============================================================

print("=" * 60)
print("Loading Tokenizer")
print("=" * 60)

tokenizer = BPETokenizer()
tokenizer.load(VOCAB_PATH, MERGES_PATH)

print("Tokenizer Loaded Successfully!\n")


# ============================================================
# Load TinyStories Dataset
# ============================================================

print("=" * 60)
print("Loading TinyStories Dataset")
print("=" * 60)

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train"
)

# Use subset for local development
if NUM_STORIES is not None:
    dataset = dataset.select(range(NUM_STORIES))

print(f"Number of Stories : {len(dataset):,}")


# ============================================================
# Encode Entire Dataset
# ============================================================

print("\n" + "=" * 60)
print("Encoding Stories")
print("=" * 60)

all_tokens = []

EOS_ID = tokenizer.token_to_id["<EOS>"]

for story in tqdm(dataset):

    text = story["text"]

    token_ids = tokenizer.encode(text)

    all_tokens.extend(token_ids)

    # Add EOS token after every story
    all_tokens.append(EOS_ID)

print("\nEncoding Complete!")

all_tokens = np.array(all_tokens, dtype=np.uint16)

print(f"Total Tokens : {len(all_tokens):,}")


# ============================================================
# Train / Validation Split
# ============================================================

print("\n" + "=" * 60)
print("Creating Train / Validation Split")
print("=" * 60)

split_index = int(len(all_tokens) * (1 - VAL_SPLIT))

train_tokens = all_tokens[:split_index]
val_tokens = all_tokens[split_index:]

print(f"Training Tokens   : {len(train_tokens):,}")
print(f"Validation Tokens : {len(val_tokens):,}")


# ============================================================
# Save Encoded Dataset
# ============================================================

print("\n" + "=" * 60)
print("Saving Encoded Dataset")
print("=" * 60)

np.save(TRAIN_OUTPUT, train_tokens)
np.save(VAL_OUTPUT, val_tokens)

print("Saved Successfully!")

print(f"\nTrain File : {TRAIN_OUTPUT}")
print(f"Validation File : {VAL_OUTPUT}")


# ============================================================
# Dataset Statistics
# ============================================================

print("\n" + "=" * 60)
print("Dataset Statistics")
print("=" * 60)

print(f"Vocabulary Size : {len(tokenizer.token_to_id):,}")
print(f"Total Tokens    : {len(all_tokens):,}")
print(f"Train Tokens    : {len(train_tokens):,}")
print(f"Val Tokens      : {len(val_tokens):,}")

print("\nDataset Preparation Complete!")
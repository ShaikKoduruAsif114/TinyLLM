# CELL 0
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# CELL 1
# ============================================================
# STEP 1 : Clone TinyLLM Repository
# ============================================================

!git clone https://github.com/shaikkoduruasif114/TinyLLM.git

%cd TinyLLM

print("Current Directory:")
!pwd

print("\nProject Structure:")
!find . -maxdepth 2 -type f | sort

# CELL 2
# ============================================================
# STEP 2 : Restore Checkpoints + Encoded Dataset
# ============================================================

import shutil
import os

BACKUP = "/kaggle/input/datasets/shaikkoduruasif114/tiny-backup"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Copy checkpoints
for f in os.listdir(f"{BACKUP}/checkpoints"):
    shutil.copy(
        f"{BACKUP}/checkpoints/{f}",
        f"checkpoints/{f}"
    )

# Copy encoded dataset
shutil.copy(f"{BACKUP}/encoded_data/train.npy", "data/train.npy")
shutil.copy(f"{BACKUP}/encoded_data/val.npy", "data/val.npy")

# Copy tokenizer
shutil.copy(f"{BACKUP}/encoded_data/vocab.json", "vocab.json")
shutil.copy(f"{BACKUP}/encoded_data/merges.json", "merges.json")

print("="*60)
print("Project Restored Successfully")
print("="*60)

!ls -lh checkpoints
print()
!ls -lh data

# CELL 3
# ============================================================
# STEP 3 : Verify Checkpoint Loading (CPU First)
# ============================================================

import torch
import sys

sys.path.append(".")

from model.gpt import GPT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

ckpt = torch.load(
    "checkpoints/rope_best.pt",
    map_location="cpu"      # <-- IMPORTANT
)

print("\nCheckpoint loaded successfully.")

print("\nStored configuration:")

for k, v in ckpt["config"].items():
    print(f"{k:20}: {v}")

print("\nBest Validation Loss:",
      ckpt["best_val_loss"])

model = GPT(ckpt["config"])

model.load_state_dict(
    ckpt["model_state_dict"],
    strict=True
)

print("\nWeights loaded successfully.")

model = model.to(DEVICE)

print("Model moved to", DEVICE)

model.eval()

print("\nEverything works!")

# CELL 4
# ============================================================
# STEP 4 : Deterministic Validation Dataset
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DeterministicValDataset(Dataset):

    def __init__(self, data_path, block_size, max_samples=784):

        self.data = np.load(data_path)

        self.block_size = block_size

        self.starts = list(
            range(
                0,
                len(self.data) - block_size - 1,
                block_size
            )
        )

        # Keep same number of windows for every experiment
        self.starts = self.starts[:max_samples]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):

        s = self.starts[idx]

        x = self.data[s:s+self.block_size]
        y = self.data[s+1:s+self.block_size+1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


def create_loader(block_size):

    return DataLoader(
        DeterministicValDataset(
            "data/val.npy",
            block_size
        ),
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )

print("✓ Validation loader ready.")

# CELL 5
for L in [64,128,192,256]:

    loader = create_loader(L)

    x,y = next(iter(loader))

    print("="*50)
    print("Context:",L)
    print("Batch:",x.shape)
    print("Samples:",len(loader.dataset))

# CELL 6
# ============================================================
# STEP 6 : Evaluate RoPE @ 128
# ============================================================

import math

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss = 0
    total_tokens = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        _, loss, _ = model(x, y)

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens

    return avg_loss, math.exp(avg_loss)


loader = create_loader(128)

loss, ppl = evaluate(model, loader)

print("="*60)
print("RoPE @ 128")
print("="*60)

print("Validation Loss :", loss)
print("Perplexity      :", ppl)

# CELL 7
# ============================================================
# STEP 6 : Runtime Evaluation Patch (No File Modification)
# ============================================================

import math
import torch
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.positional_encoding import LearnedPositionalEncoding

# ------------------------------------------------------------
# Save original methods
# ------------------------------------------------------------

_original_attention_forward = MultiHeadAttention.forward
_original_learned_forward = LearnedPositionalEncoding.forward

# ------------------------------------------------------------
# Dynamic Attention Mask
# ------------------------------------------------------------

def attention_forward_eval(self, x):

    B, T, C = x.size()

    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

    q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
    k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
    v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)

    if self.rotary_emb is not None:
        q, k = self.rotary_emb(q, k)

    att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)

    # -------- Dynamic causal mask --------
    mask = torch.tril(
        torch.ones(T, T, device=x.device, dtype=torch.bool)
    ).view(1,1,T,T)

    att = att.masked_fill(~mask, float("-inf"))

    att = F.softmax(att, dim=-1)

    att = self.attn_dropout(att)

    y = att @ v

    y = (
        y.transpose(1,2)
         .contiguous()
         .view(B,T,C)
    )

    y = self.resid_dropout(self.c_proj(y))

    return y, att


# ------------------------------------------------------------
# Learned Position Clamp
# ------------------------------------------------------------

def learned_forward_eval(self, x):

    seq_len = x.size(1)

    positions = torch.arange(
        seq_len,
        device=x.device,
        dtype=torch.long
    )

    positions = positions.clamp(
        max=self.pos_embed.num_embeddings - 1
    )

    positions = positions.unsqueeze(0)

    return x + self.pos_embed(positions)


# ------------------------------------------------------------
# Activate evaluation patch
# ------------------------------------------------------------

MultiHeadAttention.forward = attention_forward_eval
LearnedPositionalEncoding.forward = learned_forward_eval

print("="*60)
print("Evaluation patch activated.")
print("✓ Dynamic Attention Mask")
print("✓ Learned Position Clamp")
print("="*60)

# CELL 8
# ============================================================
# STEP 7 : Universal Model Loader
# ============================================================

import gc
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(name):

    checkpoint = torch.load(
        f"checkpoints/{name}_best.pt",
        map_location="cpu"          # Always load on CPU first
    )

    config = checkpoint["config"]

    model = GPT(config)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def unload_model(model):

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("✓ Universal loader ready.")

# CELL 9
# ============================================================
# STEP 8 : Universal Evaluation
# ============================================================

import math

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        _, loss, _ = model(x, y)

        tokens = y.numel()

        total_loss += loss.item() * tokens
        total_tokens += tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity

print("✓ Evaluation function ready.")

# CELL 10
print("Attention Forward :", MultiHeadAttention.forward.__name__)
print("Learned Forward  :", LearnedPositionalEncoding.forward.__name__)

# CELL 11
# ============================================================
# PATCH 1 : Dynamic Causal Mask
# ============================================================

from pathlib import Path

path = Path("model/attention.py")

text = path.read_text()

old = """        # Causal mask (upper triangle set to -inf)
        # self.bias is (1, 1, block_size, block_size), we slice to T
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))"""

new = """        # Dynamic causal mask
        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        ).view(1, 1, T, T)

        att = att.masked_fill(~mask, float("-inf"))"""

if old in text:
    text = text.replace(old, new)
    path.write_text(text)
    print("✅ attention.py patched successfully.")
else:
    print("⚠️ Pattern not found. File may already be patched.")

# CELL 12
with open("model/attention.py") as f:
    code = f.read()

print("Dynamic mask present:", "mask = torch.tril" in code)

# CELL 13
# ============================================================
# PATCH 2 : Learned PE Clamp
# ============================================================

from pathlib import Path

path = Path("model/positional_encoding.py")

text = path.read_text()

old = """        positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
        return x + self.pos_embed(positions)"""

new = """        positions = torch.arange(
            seq_len,
            device=x.device,
            dtype=torch.long
        )

        positions = positions.clamp(
            max=self.pos_embed.num_embeddings - 1
        )

        positions = positions.unsqueeze(0)

        return x + self.pos_embed(positions)"""

if old in text:
    text = text.replace(old, new)
    path.write_text(text)
    print("✅ positional_encoding.py patched successfully.")
else:
    print("⚠️ Pattern not found. File may already be patched.")

# CELL 14
with open("model/positional_encoding.py") as f:
    code = f.read()

print("Clamp present:", "clamp(" in code)

# CELL 15
import importlib

import model.attention
import model.positional_encoding
import model.block
import model.gpt

importlib.reload(model.attention)
importlib.reload(model.positional_encoding)
importlib.reload(model.block)
importlib.reload(model.gpt)

from model.gpt import GPT

print("✅ Modules reloaded.")

# CELL 16
# ============================================================
# Reload Model After Patching
# ============================================================

import gc
import torch

# Clear old model if it exists
try:
    del rope_model
except:
    pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("GPU cache cleared.")

# Reload fresh model
rope_model = load_model("rope")

print("Fresh RoPE model loaded.")

# CELL 17
# ============================================================
# Inspect checkpoint state_dict shapes
# ============================================================

ckpt = torch.load("checkpoints/rope_best.pt", map_location="cpu")

for k, v in ckpt["model_state_dict"].items():
    if "bias" in k or "pos_embed" in k:
        print(k, tuple(v.shape))

# CELL 18
# ============================================================
# Build 256-context Evaluation Model
# ============================================================

import copy
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_eval_model(model_name, context_length):

    ckpt = torch.load(
        f"checkpoints/{model_name}_best.pt",
        map_location="cpu"
    )

    config = copy.deepcopy(ckpt["config"])
    config["block_size"] = context_length

    model = GPT(config)

    state = ckpt["model_state_dict"]

    # ---------------------------------------------------------
    # Remove attention bias buffers (fixed causal masks)
    # ---------------------------------------------------------
    keys_to_remove = []

    for k in state.keys():
        if k.endswith("attn.bias"):
            keys_to_remove.append(k)

    for k in keys_to_remove:
        del state[k]

    missing, unexpected = model.load_state_dict(
        state,
        strict=False
    )

    print("="*60)
    print(model_name.upper())
    print("="*60)
    print("Missing keys:")
    print(missing)
    print("\nUnexpected keys:")
    print(unexpected)

    model = model.to(DEVICE)
    model.eval()

    return model

print("Ready.")

# CELL 19
rope256 = load_eval_model("rope",256)

# CELL 20
# ============================================================
# Evaluate ONE loaded model on all context lengths
# ============================================================

import pandas as pd

lengths = [64, 128, 192, 256]

results = []

for L in lengths:

    print(f"Evaluating @ {L}")

    loader = create_loader(L)

    loss, ppl = evaluate(rope256, loader)

    results.append({
        "Context Length": L,
        "Validation Loss": loss,
        "Perplexity": ppl
    })

rope_df = pd.DataFrame(results)

display(rope_df)

# CELL 21
# ============================================================
# Remove RoPE model
# ============================================================

del rope256

import gc
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("RoPE model removed.")

# CELL 22
learned256 = load_eval_model("learned", 256)

# CELL 23
# ============================================================
# Evaluate Learned at all context lengths
# ============================================================

import pandas as pd

lengths = [64, 128, 192, 256]

results = []

for L in lengths:

    print(f"Evaluating @ {L}")

    loader = create_loader(L)

    loss, ppl = evaluate(learned256, loader)

    results.append({
        "Context Length": L,
        "Validation Loss": loss,
        "Perplexity": ppl
    })

learned_df = pd.DataFrame(results)

display(learned_df)

# CELL 24
# ============================================================
# Remove Learned Model
# ============================================================

del learned256

import gc
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Learned model removed.")

# CELL 25
sin256 = load_eval_model("sinusoidal", 256)

# CELL 26
# ============================================================
# Evaluate Sinusoidal
# ============================================================

import pandas as pd

lengths = [64, 128, 192, 256]

results = []

for L in lengths:

    print(f"Evaluating @ {L}")

    loader = create_loader(L)

    loss, ppl = evaluate(sin256, loader)

    results.append({
        "Context Length": L,
        "Validation Loss": loss,
        "Perplexity": ppl
    })

sin_df = pd.DataFrame(results)

display(sin_df)

# CELL 27
# ============================================================
# Combine All Results
# ============================================================

rope_df["Model"] = "RoPE"
learned_df["Model"] = "Learned"
sin_df["Model"] = "Sinusoidal"

results_df = pd.concat(
    [rope_df, learned_df, sin_df],
    ignore_index=True
)

display(results_df)

# CELL 28
# ============================================================
# Perplexity vs Context Length
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

for model in ["RoPE", "Learned", "Sinusoidal"]:

    data = results_df[results_df["Model"] == model]

    plt.plot(
        data["Context Length"],
        data["Perplexity"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=model
    )

plt.xlabel("Context Length", fontsize=13)
plt.ylabel("Perplexity", fontsize=13)

plt.title(
    "Perplexity vs Context Length",
    fontsize=15,
    weight="bold"
)

plt.xticks([64,128,192,256])

plt.grid(True, alpha=0.3)

plt.legend(fontsize=11)

plt.tight_layout()

plt.savefig(
    "perplexity_vs_context_length.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# CELL 29
results_df.to_csv(
    "perplexity_vs_context_length.csv",
    index=False
)

print("Saved:")
print("  perplexity_vs_context_length.csv")
print("  perplexity_vs_context_length.png")

# CELL 30
# ============================================================
# Find tokenizer implementation
# ============================================================

from pathlib import Path

print(Path("tokeniser/bpe_tokeniser.py").read_text())

# CELL 31
# ============================================================
# Load Tokenizer
# ============================================================

from tokeniser.bpe_tokeniser import BPETokenizer

tokenizer = BPETokenizer()

tokenizer.load(
    "vocab.json",
    "merges.json"
)

print("Vocabulary Size:", len(tokenizer.token_to_id))

# CELL 32
import inspect
from model.gpt import GPT

print(inspect.getsource(GPT.forward))

# CELL 33
# ============================================================
# Encode a Sample Sentence
# ============================================================

sentence = "Once upon a time there was a"

tokens = tokenizer.encode(sentence)

print("Tokens:")
print(tokens)

import torch

x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

print("Input Shape:", x.shape)

# CELL 34
# ============================================================
# Reload RoPE Evaluation Model
# ============================================================

import gc

try:
    del learned256
except:
    pass

try:
    del sin256
except:
    pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

rope256 = load_eval_model("rope", 256)

print("RoPE model loaded successfully.")

# CELL 35
with torch.no_grad():

    logits, loss, attention = rope256(x)

print("Number of Layers :", len(attention))

print("Layer 1 Shape :", attention[0].shape)

# CELL 36
# ============================================================
# Attention Heatmaps (All Layers × All Heads)
# ============================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    nrows=6,
    ncols=6,
    figsize=(18,18)
)

for layer in range(6):

    for head in range(6):

        ax = axes[layer][head]

        att = attention[layer][0, head].cpu().numpy()

        im = ax.imshow(att, aspect="auto", origin="lower")

        ax.set_xticks([])
        ax.set_yticks([])

        if layer == 0:
            ax.set_title(f"H{head+1}", fontsize=10)

        if head == 0:
            ax.set_ylabel(f"L{layer+1}", fontsize=10)

fig.suptitle(
    "RoPE Attention Heatmaps",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "rope_attention_heatmaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# CELL 37
# ============================================================
# Use a real validation sample
# ============================================================

loader = create_loader(64)

x, y = next(iter(loader))

# Take only the first example
x = x[:1].to(DEVICE)

tokens = x[0].cpu().tolist()

print("Number of tokens:", len(tokens))

print("\nDecoded text:\n")
print(tokenizer.decode(tokens))

# CELL 38
# ============================================================
# Extract attention for a real sequence
# ============================================================

with torch.no_grad():
    logits, loss, attention = rope256(x)

print("Layers :", len(attention))
print("Shape  :", attention[0].shape)

# CELL 39
# ============================================================
# RoPE Attention Heatmaps
# ============================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(6, 6, figsize=(18,18))

for layer in range(6):

    for head in range(6):

        att = attention[layer][0, head].cpu().numpy()

        ax = axes[layer, head]

        im = ax.imshow(
            att,
            cmap="viridis",
            interpolation="nearest",
            origin="lower"
        )

        ax.set_xticks([])
        ax.set_yticks([])

        if layer == 0:
            ax.set_title(f"H{head+1}", fontsize=10)

        if head == 0:
            ax.set_ylabel(f"L{layer+1}", fontsize=10)

fig.suptitle(
    "RoPE Attention Patterns",
    fontsize=20,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "rope_attention_heatmaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# CELL 40
# ============================================================
# Load Learned Model
# ============================================================

import gc

try:
    del rope256
except:
    pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

learned256 = load_eval_model("learned", 256)

print("Learned model loaded.")

# CELL 41
# ============================================================
# Extract Learned Attention
# ============================================================

with torch.no_grad():
    _, _, attention = learned256(x)

print("Layers :", len(attention))
print("Shape  :", attention[0].shape)

# CELL 42
# ============================================================
# Learned Attention Heatmaps
# ============================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(6, 6, figsize=(18,18))

for layer in range(6):

    for head in range(6):

        att = attention[layer][0, head].cpu().numpy()

        ax = axes[layer, head]

        ax.imshow(
            att,
            cmap="viridis",
            origin="lower",
            interpolation="nearest"
        )

        ax.set_xticks([])
        ax.set_yticks([])

        if layer == 0:
            ax.set_title(f"H{head+1}", fontsize=10)

        if head == 0:
            ax.set_ylabel(f"L{layer+1}", fontsize=10)

fig.suptitle(
    "Learned Positional Encoding Attention",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "learned_attention_heatmaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# CELL 43
# ============================================================
# Load Sinusoidal Model
# ============================================================

del learned256

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

sin256 = load_eval_model("sinusoidal", 256)

print("Sinusoidal model loaded.")

# CELL 44
# ============================================================
# Extract Sinusoidal Attention
# ============================================================

with torch.no_grad():
    _, _, attention = sin256(x)

print("Layers :", len(attention))
print("Shape  :", attention[0].shape)


# CELL 45
# ============================================================
# Sinusoidal Attention Heatmaps
# ============================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(6, 6, figsize=(18,18))

for layer in range(6):

    for head in range(6):

        att = attention[layer][0, head].cpu().numpy()

        ax = axes[layer, head]

        ax.imshow(
            att,
            cmap="viridis",
            origin="lower",
            interpolation="nearest"
        )

        ax.set_xticks([])
        ax.set_yticks([])

        if layer == 0:
            ax.set_title(f"H{head+1}", fontsize=10)

        if head == 0:
            ax.set_ylabel(f"L{layer+1}", fontsize=10)

fig.suptitle(
    "Sinusoidal Positional Encoding Attention",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "sinusoidal_attention_heatmaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# CELL 46
# ============================================================
# Compare Saved Heatmaps
# ============================================================

print("Saved files:")
print(" - rope_attention_heatmaps.png")
print(" - learned_attention_heatmaps.png")
print(" - sinusoidal_attention_heatmaps.png")

# CELL 47
# ============================================================
# Reload Learned Model
# ============================================================

import gc

try:
    del sin256
except:
    pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

learned256 = load_eval_model("learned", 256)

print("Learned model loaded.")

# CELL 48
# ============================================================
# Learned Position Embedding Similarity
# ============================================================

import torch.nn.functional as F
import matplotlib.pyplot as plt

emb = learned256.pos_encoding.pos_embed.weight.detach().cpu()

emb = F.normalize(emb, dim=1)

similarity = emb @ emb.T

plt.figure(figsize=(8,8))

plt.imshow(similarity, cmap="viridis", origin="lower")

plt.colorbar()

plt.title("Learned Position Embedding Similarity")

plt.xlabel("Position")

plt.ylabel("Position")

plt.tight_layout()

plt.savefig(
    "learned_position_similarity.png",
    dpi=300
)

plt.show()

# CELL 49
del learned256

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

sin256 = load_eval_model("sinusoidal", 256)

# CELL 50
# ============================================================
# Check what variables currently exist
# ============================================================

print("learned256" in globals())
print("rope256" in globals())
print("sin256" in globals())

# CELL 51
# ============================================================
# Reload the Learned model
# ============================================================

import gc
import torch

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

learned256 = load_eval_model("learned", 256)

print("\nVariable exists:", "learned256" in globals())
print("Model type:", type(learned256).__name__)
print("Parameter count:", sum(p.numel() for p in learned256.parameters()))

# CELL 52
print(type(learned256.pos_encoding))
print(learned256.pos_encoding.__class__.__name__)

# CELL 53
# ============================================================
# Learned Position Embedding Similarity
# ============================================================

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Get learned position embedding matrix
emb = learned256.pos_encoding.pos_embed.weight.detach().cpu()

print("Embedding shape:", emb.shape)

# Normalize embeddings
emb = F.normalize(emb, p=2, dim=1)

# Cosine similarity matrix
similarity = emb @ emb.T

plt.figure(figsize=(8, 8))

plt.imshow(
    similarity.numpy(),
    cmap="viridis",
    origin="lower"
)

plt.colorbar()

plt.title("Learned Position Embedding Similarity")

plt.xlabel("Position")
plt.ylabel("Position")

plt.tight_layout()

plt.savefig(
    "learned_position_similarity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# CELL 54
# ============================================================
# Load Sinusoidal Model
# ============================================================

del learned256

import gc
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

sin256 = load_eval_model("sinusoidal", 256)

print("Sinusoidal model loaded.")

# CELL 55
# ============================================================
# Sinusoidal Position Embedding Similarity
# ============================================================

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Get sinusoidal position embeddings
emb = sin256.pos_encoding.pe[0].detach().cpu()

print("Embedding shape:", emb.shape)

# Normalize embeddings
emb = F.normalize(emb, p=2, dim=1)

# Cosine similarity matrix
similarity = emb @ emb.T

plt.figure(figsize=(8,8))

plt.imshow(
    similarity.numpy(),
    cmap="viridis",
    origin="lower"
)

plt.colorbar()

plt.title("Sinusoidal Position Embedding Similarity")

plt.xlabel("Position")
plt.ylabel("Position")

plt.tight_layout()

plt.savefig(
    "sinusoidal_position_similarity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# CELL 56
# ============================================================
# Load All Three Models
# ============================================================

import gc

try:
    del sin256
except:
    pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

rope = load_eval_model("rope", 256)
learned = load_eval_model("learned", 256)
sin = load_eval_model("sinusoidal", 256)

print("All models loaded.")

# CELL 57
# ============================================================
# Better Text Generation
# ============================================================

@torch.no_grad()
def generate_text(
    model,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
):

    model.eval()

    ids = tokenizer.encode(prompt)

    x = torch.tensor(ids).unsqueeze(0).to(DEVICE)

    for _ in range(max_new_tokens):

        x_cond = x[:, -256:]

        logits, _, _ = model(x_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:

            values, _ = torch.topk(logits, top_k)

            logits[logits < values[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, 1)

        x = torch.cat([x, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(x[0].tolist())

# CELL 58
prompt = "Once upon a time"

print("="*80)
print("RoPE")
print("="*80)
print(generate_text(rope, prompt))

print("\n")

print("="*80)
print("Learned")
print("="*80)
print(generate_text(learned, prompt))

print("\n")

print("="*80)
print("Sinusoidal")
print("="*80)
print(generate_text(sin, prompt))
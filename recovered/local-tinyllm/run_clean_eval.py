import torch
import numpy as np
import math
import copy
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(".")
from model.gpt import GPT
from model.attention import MultiHeadAttention
from model.positional_encoding import LearnedPositionalEncoding
import torch.nn.functional as F

# Patch Attention for Evaluation
def attention_forward_eval(self, x):
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
    k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
    v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)
    if self.rotary_emb is not None:
        q, k = self.rotary_emb(q, k)
    att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1,1,T,T)
    att = att.masked_fill(~mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
    y = y.transpose(1,2).contiguous().view(B,T,C)
    y = self.resid_dropout(self.c_proj(y))
    return y, att

def learned_forward_eval(self, x):
    seq_len = x.size(1)
    positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
    positions = positions.clamp(max=self.pos_embed.num_embeddings - 1)
    positions = positions.unsqueeze(0)
    return x + self.pos_embed(positions)

MultiHeadAttention.forward = attention_forward_eval
LearnedPositionalEncoding.forward = learned_forward_eval

class CleanValDataset(Dataset):
    def __init__(self, data_path, block_size, total_tokens):
        self.data = np.load(data_path)[:total_tokens+1] # +1 for targets
        self.block_size = block_size
        self.starts = list(range(0, total_tokens, block_size))
        if len(self.starts) > 0 and self.starts[-1] + block_size > total_tokens:
            self.starts = self.starts[:-1]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.data[s:s+self.block_size]
        y = self.data[s+1:s+self.block_size+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def load_eval_model(model_name, device):
    ckpt = torch.load(f"backup/checkpoints/{model_name}_best.pt", map_location="cpu")
    config = copy.deepcopy(ckpt["config"])
    # DO NOT overwrite block_size. Let it use the original config so weight shapes match exactly!
    model = GPT(config)
    state = ckpt["model_state_dict"]
    keys_to_remove = [k for k in state.keys() if k.endswith("attn.bias")]
    for k in keys_to_remove:
        del state[k]
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss, _ = model(x, y)
        tokens = y.numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    val_data = np.load("backup/encoded_data/val.npy")
    max_tokens = len(val_data) - 1
    total_tokens = (max_tokens // 768) * 768
    print(f"Using EXACTLY {total_tokens} tokens for clean evaluation.")

    lengths = [64, 128, 192, 256]
    models = ["rope", "learned", "sinusoidal"]
    
    results = []
    
    for m in models:
        model = load_eval_model(m, device)
        for L in lengths:
            dataset = CleanValDataset("backup/encoded_data/val.npy", L, total_tokens)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            loss, ppl = evaluate(model, loader, device)
            
            results.append({
                "model": m,
                "context_length": L,
                "validation_loss": loss,
                "perplexity": ppl,
                "num_sequences": len(dataset),
                "num_tokens": len(dataset) * L
            })
            print(f"Model: {m}, Length: {L}, Loss: {loss:.4f}, PPL: {ppl:.4f}, Seqs: {len(dataset)}, Tokens: {len(dataset) * L}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import json
    with open("clean_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    main()

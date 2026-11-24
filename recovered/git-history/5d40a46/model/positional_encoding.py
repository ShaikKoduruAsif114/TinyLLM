import torch
import torch.nn as nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class SinusoidalEncoding(nn.Module):
    def __init__(self, n_embd, max_seq_len=512):
        super().__init__()
        self.register_buffer("pe", self._build_pe(n_embd, max_seq_len), persistent=False)

    def _build_pe(self, n_embd, max_seq_len):
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / n_embd))
        pe = torch.zeros(max_seq_len, n_embd)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.dtype)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, n_embd):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, n_embd)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        # Positions beyond num_embeddings - 1 lose distinct positional information and collapse to the same vector.
        positions = positions.clamp(max=self.pos_embed.num_embeddings - 1)
        positions = positions.unsqueeze(0)
        return x + self.pos_embed(positions)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, q, k):
        seq_len = q.size(-2)
        position = torch.arange(seq_len, device=q.device, dtype=q.dtype)
        freqs = torch.einsum("i,j->ij", position, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

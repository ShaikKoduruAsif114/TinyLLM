import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, block_size=128, rotary_emb=None):
        """
        Multi-Head Self Attention with optional RoPE
        
        Args:
            n_embd: embedding dimension (384)
            n_head: number of heads (6)
            dropout: dropout rate
            block_size: max sequence length (for causal mask)
            rotary_emb: Optional RoPE instance (if None, no rotation applied)
        """
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head  # 64
        self.rotary_emb = rotary_emb
        
        # Key, Query, Value projections (all in one linear layer for efficiency, or separate)
        # Using separate is clearer for learning
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # Combined Q, K, V
        self.c_proj = nn.Linear(n_embd, n_embd)      # Output projection
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask (lower triangular)
        # register_buffer means it's saved with model but not trained
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        
        # Flash attention not implemented here for simplicity (we'll use standard)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C) where B=batch, T=seq_len, C=n_embd
            
        Returns:
            output: (B, T, C)
            attention_weights: (B, H, T, T) for visualization if needed
        """
        B, T, C = x.size()
        
        # Calculate Q, K, V
        # x @ c_attn.T -> (B, T, 3*C) -> split into 3 tensors of (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided (for RoPE experiment)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)
        
        # Attention scores: (B, H, T, head_dim) @ (B, H, head_dim, T) -> (B, H, T, T)
        # Scale by sqrt(head_dim)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Causal mask (upper triangle set to -inf)
        # self.bias is (1, 1, block_size, block_size), we slice to T
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax over last dimension (keys)
        att = F.softmax(att, dim=-1)
        
        # Attention dropout
        att = self.attn_dropout(att)
        
        # Weighted sum of values: (B, H, T, T) @ (B, H, T, head_dim) -> (B, H, T, head_dim)
        y = att @ v
        
        # Reshape back: (B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y, att  # Return att for visualization if needed
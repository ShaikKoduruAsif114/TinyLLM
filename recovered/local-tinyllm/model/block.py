import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, block_size=128, rotary_emb=None):
        """
        Single Transformer Block with Pre-LN
        
        Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
        """
        super().__init__()
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout, block_size, rotary_emb)
        self.ffn = FeedForward(n_embd, dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            x: (B, T, C)
            att: attention weights for visualization
        """
        # Pre-LN + Attention + Residual
        attn_out, att = self.attn(self.ln1(x))
        x = x + attn_out
        
        # Pre-LN + FFN + Residual
        x = x + self.ffn(self.ln2(x))
        
        return x, att
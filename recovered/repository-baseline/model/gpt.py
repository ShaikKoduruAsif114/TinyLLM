import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from feedforward import FeedForward
from block import TransformerBlock
from positional_encoding import (
    SinusoidalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEmbedding
)


class GPT(nn.Module):
    def __init__(self, config):
        """
        Full GPT Language Model.

        Args:
            config: dict or object with fields:
                - vocab_size: 4096
                - n_embd: 384
                - n_head: 6
                - n_layer: 6
                - block_size: 128
                - dropout: 0.1
                - pos_encoding: 'sinusoidal' | 'learned' | 'rope'
        """
        super().__init__()

        self.config = config

        # ─────────────────────────────────────────
        # Token Embedding Table
        # Shape: (vocab_size, n_embd)
        # Maps integer token ids to dense vectors
        # ─────────────────────────────────────────
        self.token_embedding = nn.Embedding(
            config['vocab_size'],
            config['n_embd']
        )

        # ─────────────────────────────────────────
        # Positional Encoding
        # Three versions for the experiment
        # ─────────────────────────────────────────
        pos_type = config['pos_encoding']

        if pos_type == 'sinusoidal':
            self.pos_encoding = SinusoidalEncoding(
                n_embd=config['n_embd'],
                max_seq_len=config['block_size'] * 2  # *2 for extrapolation test
            )
            rotary_emb = None

        elif pos_type == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(
                max_seq_len=config['block_size'],
                n_embd=config['n_embd']
            )
            rotary_emb = None

        elif pos_type == 'rope':
            # RoPE does not add to embeddings
            # It is passed into attention directly
            self.pos_encoding = None
            rotary_emb = RotaryPositionalEmbedding(
                head_dim=config['n_embd'] // config['n_head'],
                max_seq_len=config['block_size'] * 2
            )

        else:
            raise ValueError(f"Unknown pos_encoding: {pos_type}")

        # ─────────────────────────────────────────
        # Transformer Blocks
        # Stack of n_layer blocks
        # ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=config['n_embd'],
                n_head=config['n_head'],
                dropout=config['dropout'],
                block_size=config['block_size'],
                rotary_emb=rotary_emb
            )
            for _ in range(config['n_layer'])
        ])

        # ─────────────────────────────────────────
        # Final Layer Norm
        # Applied after all transformer blocks
        # ─────────────────────────────────────────
        self.ln_f = nn.LayerNorm(config['n_embd'])

        # ─────────────────────────────────────────
        # Output Projection Head
        # Maps n_embd -> vocab_size
        # Shape: (n_embd, vocab_size)
        # ─────────────────────────────────────────
        self.lm_head = nn.Linear(
            config['n_embd'],
            config['vocab_size'],
            bias=False
        )

        # ─────────────────────────────────────────
        # Weight Tying
        # Share weights between token embedding
        # and output projection
        # Saves ~1.5M parameters
        # Empirically improves performance
        # ─────────────────────────────────────────
        self.token_embedding.weight = self.lm_head.weight

        # ─────────────────────────────────────────
        # Initialize weights
        # ─────────────────────────────────────────
        self.apply(self._init_weights)

        # Print number of parameters
        print(f"Model parameters: {self.count_parameters():,}")

    def _init_weights(self, module):
        """
        Initialize weights using normal distribution.
        Standard practice for transformer models.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx: (B, T) integer token ids
            targets: (B, T) integer token ids shifted by 1
                     If provided, compute and return loss

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross entropy loss (or None if targets not provided)
        """
        B, T = idx.size()

        # ── Token Embeddings ──────────────────────
        # (B, T) -> (B, T, n_embd)
        x = self.token_embedding(idx)

        # ── Positional Encoding ───────────────────
        # Add positional info (not for RoPE — handled inside attention)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # ── Transformer Blocks ────────────────────
        # Collect attention weights from all layers
        # for visualization later
        all_attention_weights = []

        for block in self.blocks:
            x, att = block(x)
            all_attention_weights.append(att)

        # ── Final Layer Norm ──────────────────────
        x = self.ln_f(x)

        # ── Output Projection ─────────────────────
        # (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        # ── Loss Calculation ──────────────────────
        loss = None
        if targets is not None:
            # Flatten for cross entropy
            # logits: (B, T, vocab_size) -> (B*T, vocab_size)
            # targets: (B, T) -> (B*T,)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0  # ignore PAD token (id=0)
            )

        return logits, loss, all_attention_weights

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.

        Args:
            idx: (B, T) context token ids
            max_new_tokens: how many tokens to generate
            temperature: controls randomness
                         lower = more deterministic
                         higher = more random
            top_k: if set, only sample from top k tokens

        Returns:
            idx: (B, T + max_new_tokens) extended token ids
        """
        for _ in range(max_new_tokens):

            # Crop context to block_size
            idx_cond = idx[:, -self.config['block_size']:]

            # Forward pass (no targets, no loss)
            logits, _, _ = self(idx_cond)

            # Take logits at last position only
            # (B, T, vocab_size) -> (B, vocab_size)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Softmax to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


###############################################################
# Quick test
###############################################################
if __name__ == "__main__":

    config = {
        'vocab_size': 4096,
        'n_embd': 384,
        'n_head': 6,
        'n_layer': 6,
        'block_size': 128,
        'dropout': 0.1,
        'pos_encoding': 'rope'  # try 'learned' and 'rope' too
    }

    model = GPT(config)

    # Dummy input
    x = torch.randint(0, 4096, (2, 128))  # batch=2, seq_len=128
    targets = torch.randint(0, 4096, (2, 128))

    logits, loss, att_weights = model(x, targets)

    print(f"Input shape:   {x.shape}")
    print(f"Logits shape:  {logits.shape}")
    print(f"Loss:          {loss.item():.4f}")
    print(f"Attention layers: {len(att_weights)}")
    print(f"Attention shape:  {att_weights[0].shape}")
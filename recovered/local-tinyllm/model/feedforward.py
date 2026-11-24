import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        """
        Feed Forward Network
        
        Args:
            n_embd: embedding dimension (384)
            dropout: dropout rate
        """
        super().__init__()
        
        self.net = nn.Sequential(
            # Expand: n_embd -> 4 * n_embd
            nn.Linear(n_embd, 4 * n_embd),
            
            # GELU activation (smoother than ReLU, better for transformers)
            nn.GELU(),
            
            # Compress back: 4 * n_embd -> n_embd
            nn.Linear(4 * n_embd, n_embd),
            
            # Dropout for regularization
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            output: (B, T, C)
        """
        return self.net(x)
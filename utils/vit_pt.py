import torch
import torch.nn as nn
import torchvision

class PatchEmbed(nn.Module):
    """Patch embedding layer for transformer input"""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # Number of patches
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 3-channel and 1-channel images
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Patch embedding
        x = self.proj(x).flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class SmallTransformer(nn.Module):
    """Small transformer model for binary classification"""
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = 'SmallTransformer'
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embedding
        x = x + self.pos_embed[:, 1:, :]
        
        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x
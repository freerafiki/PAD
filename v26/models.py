import torch
import torch.nn as nn
from transformers import Dinov2Model, ViTModel


class BaselineScorer(nn.Module):
    """Baseline: Just RGB through ViT."""

    def __init__(self, pretrained_name="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_name)

        # Simple scoring head
        self.scorer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb):
        """
        Args:
            rgb: (B, 3, H, W) RGB images
        Returns:
            scores: (B, 1) alignment scores in [0, 1]
        """
        vit_feats = self.vit(rgb).pooler_output  # (B, 768)
        scores = self.scorer(vit_feats)
        return scores


class GeometricScorer(nn.Module):
    """RGB + 3 geometric channels."""

    def __init__(self, pretrained_name="google/vit-base-patch16-224"):
        super().__init__()

        # Project 6 channels to 3
        self.projection = nn.Conv2d(6, 3, kernel_size=1)

        # ViT backbone
        self.vit = ViTModel.from_pretrained(pretrained_name)

        # Scoring head
        self.scorer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_geometric):
        """
        Args:
            rgb_geometric: (B, 6, H, W)
                           channels 0-2: RGB
                           channels 3-5: geometric features
        Returns:
            scores: (B, 1)
        """
        x = self.projection(rgb_geometric)  # (B, 3, H, W)
        vit_feats = self.vit(x).pooler_output  # (B, 768)
        scores = self.scorer(vit_feats)
        return scores


class MultiModalScorer(nn.Module):
    """RGB + Geometry + DINO semantic features."""

    def __init__(
        self,
        geometric_vit="google/vit-base-patch16-224",
        dino_model="facebook/dinov2-base",
    ):
        super().__init__()

        # Branch 1: Geometric ViT
        self.projection = nn.Conv2d(6, 3, kernel_size=1)
        self.geometric_vit = ViTModel.from_pretrained(geometric_vit)

        # Branch 2: DINO (frozen)
        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
        print(f"DINO model loaded. Output dimension: {self.dino.config.hidden_size}")

        # Fusion head
        dino_dim = self.dino.config.hidden_size  # 768 for dinov2-base
        geom_dim = self.geometric_vit.config.hidden_size  # 768 for vit-base

        self.fusion = nn.Sequential(
            nn.Linear(geom_dim + dino_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb, rgb_geometric):
        """
        Args:
            rgb: (B, 3, H, W) - for DINO
            rgb_geometric: (B, 6, H, W) - for geometric branch
        Returns:
            scores: (B, 1)
        """
        # Geometric branch
        x = self.projection(rgb_geometric)
        geom_feats = self.geometric_vit(x).pooler_output  # (B, 768)

        # DINO branch (no gradients)
        with torch.no_grad():
            dino_feats = self.dino(rgb).pooler_output  # (B, 768)

        # Fuse
        combined = torch.cat([geom_feats, dino_feats], dim=1)  # (B, 1536)
        scores = self.fusion(combined)

        return scores

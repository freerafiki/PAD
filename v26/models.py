import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CrossModalAttention(nn.Module):
    """Cross-attention between geometric and visual features."""

    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),  # Smaller expansion for small data
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # Cross attention: query attends to key_value
        attended, _ = self.cross_attn(query, key_value, key_value)
        x = self.norm1(query + self.dropout(attended))

        # Feed-forward
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MultiModalScorerV2(nn.Module):
    """
    Improved multi-modal scorer with:
    - Better geometric feature processing (Option B)
    - Wider fusion network (Option A)
    - Optional cross-attention (Option C)
    - Regularization for small data
    """

    def __init__(
        self,
        geometric_vit="google/vit-base-patch16-224",
        dino_model="facebook/dinov2-base",
        use_cross_attention=False,
        dropout=0.2,
    ):
        """
        Args:
            geometric_vit: Pretrained ViT for geometric branch
            dino_model: Pretrained DINO for visual branch
            use_cross_attention: If True, use cross-modal attention (Option C)
            dropout: Dropout rate (higher for smaller datasets)
        """
        super().__init__()

        self.use_cross_attention = use_cross_attention

        # ============================================
        # Geometric Branch (Option B: Better Processing)
        # ============================================

        # Learn to process geometric features before ViT
        self.geometric_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )

        # Fuse RGB with encoded geometric features
        self.rgb_geom_fusion = nn.Sequential(
            nn.Conv2d(3 + 128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

        # ViT for geometric reasoning
        self.geometric_vit = ViTModel.from_pretrained(geometric_vit)

        # ============================================
        # Visual Branch (DINO - Frozen)
        # ============================================

        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()

        print(
            f"DINO model loaded (frozen). Output dimension: {self.dino.config.hidden_size}"
        )

        # ============================================
        # Cross-Modal Attention (Option C - Optional)
        # ============================================

        if use_cross_attention:
            self.geom_to_visual = CrossModalAttention(
                dim=768, num_heads=8, dropout=dropout
            )
            self.visual_to_geom = CrossModalAttention(
                dim=768, num_heads=8, dropout=dropout
            )
            print("Using cross-modal attention")

        # ============================================
        # Fusion Head (Option A: Wider, Better Regularization)
        # ============================================

        self.fusion = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            # nn.Linear(1024, 512),
            # nn.LayerNorm(512),
            # nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(1024, 256),
            # nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Less dropout at end
            nn.Linear(256, 1),
            # No sigmoid - use BCEWithLogitsLoss
        )

    def forward(self, rgb, rgb_geometric):
        """
        Args:
            rgb: (B, 3, H, W) - RGB images for DINO
            rgb_geometric: (B, 6, H, W) - RGB + 3 geometric channels

        Returns:
            logits: (B, 1) - raw scores (apply sigmoid for probabilities)
        """
        # ============================================
        # Geometric Branch
        # ============================================

        # Split RGB and geometric channels
        rgb_only = rgb_geometric[:, :3]
        geom_only = rgb_geometric[:, 3:]

        # Encode geometric features
        geom_encoded = self.geometric_encoder(geom_only)  # (B, 128, H, W)

        # Fuse RGB with encoded geometric
        combined_input = torch.cat([rgb_only, geom_encoded], dim=1)  # (B, 131, H, W)
        vit_input = self.rgb_geom_fusion(combined_input)  # (B, 3, H, W)

        # Extract geometric features via ViT
        geom_output = self.geometric_vit(vit_input)
        geom_feats = geom_output.pooler_output  # (B, 768)

        # ============================================
        # Visual Branch (DINO)
        # ============================================

        with torch.no_grad():
            dino_output = self.dino(rgb)
            dino_feats = dino_output.pooler_output  # (B, 768)

        # ============================================
        # Cross-Modal Attention (Optional)
        # ============================================

        if self.use_cross_attention:
            # Add sequence dimension for attention
            geom_feats_seq = geom_feats.unsqueeze(1)  # (B, 1, 768)
            dino_feats_seq = dino_feats.unsqueeze(1)  # (B, 1, 768)

            # Bidirectional cross-attention
            geom_attended = self.geom_to_visual(geom_feats_seq, dino_feats_seq).squeeze(
                1
            )
            visual_attended = self.visual_to_geom(
                dino_feats_seq, geom_feats_seq
            ).squeeze(1)

            # Use attended features
            combined = torch.cat([geom_attended, visual_attended], dim=1)  # (B, 1536)
        else:
            # Simple concatenation
            combined = torch.cat([geom_feats, dino_feats], dim=1)  # (B, 1536)

        # ============================================
        # Final Fusion and Scoring
        # ============================================

        logits = self.fusion(combined)  # (B, 1)

        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import Dinov2Model, ViTModel
from transformers import DINOv3ViTModel as Dinov2Model, ViTModel

"""
Because of the fact that we compute a combination of losses 
(at this moment BCE + custom Ranking Losses)
we will use both logits and scores (after sigmoid)

so all the models here will output the logits (before sigmoids) 
and whenever the model is used, sigmoid should be applied afterwards
"""


class RGBScorer(nn.Module):
    """Baseline: Just RGB through ViT."""

    def __init__(self, 
                pretrained_vit_name="google/vit-base-patch16-224",
                freeze_vit_layers=0,
                dropout=0.3):

        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_vit_name)

        # Freeze early layers
        self.frozen_layers = freeze_vit_layers
        if self.frozen_layers > 0:
            for name, param in self.vit.named_parameters():
                layer_num = self._extract_layer_num(name)
                if layer_num is not None and layer_num < freeze_vit_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        self.dropout = dropout

        # Simple scoring head
        self.scorer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            # nn.Sigmoid(),
        )

    def _extract_layer_num(self, param_name):
        """Extract layer number from parameter name."""
        import re
        match = re.search(r'encoder\.layer\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        return None

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

    def __init__(self, pretrained_name="google/vit-base-patch16-224", geometric_channel_scale=1.0):
        super().__init__()
        self.geometric_channel_scale = geometric_channel_scale

        self.projection = nn.Conv2d(6, 3, kernel_size=1)

        self.vit = ViTModel.from_pretrained(pretrained_name)

        self.scorer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, rgb_geometric):
        """
        Args:
            rgb_geometric: (B, 6, H, W)
                           channels 0-2: RGB
                           channels 3-5: geometric features (scaled by geometric_channel_scale)
        Returns:
            scores: (B, 1)
        """
        x = torch.cat([
            rgb_geometric[:, :3],
            rgb_geometric[:, 3:] * self.geometric_channel_scale,
        ], dim=1)
        x = self.projection(x)
        vit_feats = self.vit(x).pooler_output
        scores = self.scorer(vit_feats)
        return scores


class MultiModalScorer(nn.Module):
    """RGB + Geometry + DINO semantic features."""

    def __init__(
        self,
        geometric_vit="google/vit-base-patch16-224",
        dino_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        geometric_channel_scale=1.0,
    ):
        super().__init__()
        self.geometric_channel_scale = geometric_channel_scale

        self.projection = nn.Conv2d(6, 3, kernel_size=1)
        self.geometric_vit = ViTModel.from_pretrained(geometric_vit)

        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
        print(f"DINO model loaded. Output dimension: {self.dino.config.hidden_size}")

        dino_dim = self.dino.config.hidden_size
        geom_dim = self.geometric_vit.config.hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(geom_dim + dino_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, rgb, rgb_geometric):
        rgb_geom = torch.cat([
            rgb_geometric[:, :3],
            rgb_geometric[:, 3:] * self.geometric_channel_scale,
        ], dim=1)
        x = self.projection(rgb_geom)
        geom_feats = self.geometric_vit(x).pooler_output

        with torch.no_grad():
            dino_feats = self.dino(rgb).pooler_output

        combined = torch.cat([geom_feats, dino_feats], dim=1)
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
        dino_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        use_cross_attention=False,
        dropout=0.2,
        geometric_channel_scale=1.0,
    ):
        """
        Args:
            geometric_vit: Pretrained ViT for geometric branch
            dino_model: Pretrained DINO for visual branch
            use_cross_attention: If True, use cross-modal attention (Option C)
            dropout: Dropout rate (higher for smaller datasets)
            geometric_channel_scale: Multiplier for geometric channels (3-5)
        """
        super().__init__()
        self.geometric_channel_scale = geometric_channel_scale

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

        rgb_only = rgb_geometric[:, :3]
        geom_only = rgb_geometric[:, 3:] * self.geometric_channel_scale

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

class MultiModalScorerV2_Practical(nn.Module):
    """
    Practical model for small datasets (~5K samples).
    
    Key features:
    - Frozen DINO (86M params)
    - Partially frozen ViT (14M trainable, 72M frozen)
    - Small new layers (2M params)
    - Total trainable: ~16M params
    """
    
    def __init__(self,
                 geometric_vit='google/vit-base-patch16-224',
                 dino_model='facebook/dinov2-base',
                 freeze_vit_layers=10,
                 dropout=0.4,
                 geometric_channel_scale=1.0):
        super().__init__()
        self.geometric_channel_scale = geometric_channel_scale
        
        # Geometric encoder (from scratch)
        self.geometric_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.rgb_geom_fusion = nn.Sequential(
            nn.Conv2d(3 + 128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        
        # ViT (partially frozen)
        self.geometric_vit = ViTModel.from_pretrained(geometric_vit)
        
        # Freeze early layers
        for name, param in self.geometric_vit.named_parameters():
            layer_num = self._extract_layer_num(name)
            if layer_num is not None and layer_num < freeze_vit_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        trainable_vit = sum(p.numel() for p in self.geometric_vit.parameters() if p.requires_grad)
        total_vit = sum(p.numel() for p in self.geometric_vit.parameters())
        print(f"ViT: {trainable_vit:,} trainable / {total_vit:,} total ({100*trainable_vit/total_vit:.1f}%)")
        
        # DINO (frozen)
        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
        
        # Fusion (smaller for small data)
        self.fusion = nn.Sequential(
            nn.Linear(1536, 768),     # Smaller first layer
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 1)
        )
    
    def _extract_layer_num(self, param_name):
        """Extract layer number from parameter name."""
        import re
        match = re.search(r'encoder\.layer\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        return None
    
    def forward(self, rgb, rgb_geometric):
        rgb_only = rgb_geometric[:, :3]
        geom_only = rgb_geometric[:, 3:] * self.geometric_channel_scale
        geom_encoded = self.geometric_encoder(geom_only)
        combined_input = torch.cat([rgb_only, geom_encoded], dim=1)
        vit_input = self.rgb_geom_fusion(combined_input)

        geom_feats = self.geometric_vit(vit_input).pooler_output

        with torch.no_grad():
            dino_feats = self.dino(rgb).pooler_output

        combined = torch.cat([geom_feats, dino_feats], dim=1)
        logits = self.fusion(combined)

        return logits


class MultiModalScorerWeightedVit(nn.Module):
    """
    Multi-modal scorer that explicitly injects contact-region information
    into the ViT feature representation via weighted patch pooling.

    MultiModalScorerV2_Practical architecture flow:
        rgb_geometric → geometric_encoder → rgb_geom_fusion → ViT → CLS token (768-d)
                                                                           ↓
        rgb → DINO → DINO features (768-d)  →  concat → [CLS + DINO] (1536-d) → fusion head

    The CLS token is ViT's own aggregation of all patches via self-attention.
    It may or may not focus on the contact region.

    New architecture flow:
        rgb_geometric → geometric_encoder → rgb_geom_fusion → ViT → CLS token (768-d)
                                                               ↓
                                                     patch tokens (196 × 768)
                                                               ↓
                                                  contact-weighted pooling (768-d)
                                                                           ↓
        rgb → DINO → DINO features (768-d)  →  concat → [CLS + contact_pooled + DINO] (2304-d) → fusion head

    The contact channel (rgb_geometric[:, 5], shape B×H×W) is downsampled to the
    ViT patch grid (14×14 → 196 patches), flattened, and used as weights for a
    weighted mean over the 196 patch tokens. This produces a 768-d "contact-pooled"
    feature that explicitly captures what the boundary region looks like.

    Why this is better than an attention-correlation loss:
    - No output_attentions=True needed (no performance penalty)
    - Works with frozen ViT (pooling happens after the ViT)
    - Guaranteed signal (contact region always contributes)
    - Cleaner gradients (simple multiplication + averaging, no Pearson math)
    """

    def __init__(self,
                 geometric_vit='google/vit-base-patch16-224',
                 dino_model='facebook/dinov2-base',
                 freeze_vit_layers=10,
                 dropout=0.4,
                 geometric_channel_scale=1.0):
        super().__init__()
        self.geometric_channel_scale = geometric_channel_scale

        # Geometric encoder (from scratch)
        self.geometric_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.rgb_geom_fusion = nn.Sequential(
            nn.Conv2d(3 + 128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )

        # ViT (partially frozen)
        self.geometric_vit = ViTModel.from_pretrained(geometric_vit)

        # Freeze early layers
        for name, param in self.geometric_vit.named_parameters():
            layer_num = self._extract_layer_num(name)
            if layer_num is not None and layer_num < freeze_vit_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_vit = sum(p.numel() for p in self.geometric_vit.parameters() if p.requires_grad)
        total_vit = sum(p.numel() for p in self.geometric_vit.parameters())
        print(f"ViT: {trainable_vit:,} trainable / {total_vit:,} total ({100*trainable_vit/total_vit:.1f}%)")

        # DINO (frozen)
        self.dino = Dinov2Model.from_pretrained(dino_model)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()

        # Fusion head: input is CLS (768) + contact-pooled (768) + DINO (768) = 2304
        self.fusion = nn.Sequential(
            nn.Linear(2304, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 1)
        )

    def _extract_layer_num(self, param_name):
        import re
        match = re.search(r'encoder\.layer\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        return None

    def _contact_weighted_pool(self, last_hidden_state, contact_mask):
        """
        Weighted pooling of ViT patch tokens using contact region as weights.

        Args:
            last_hidden_state: (B, 197, 768) from ViT — CLS + 196 patches
            contact_mask: (B, H, W) — channel 5 of rgb_geometric

        Returns:
            pooled: (B, 768) — weighted average of patch tokens
        """
        patch_tokens = last_hidden_state[:, 1:, :]  # (B, 196, 768)
        B, num_patches, dim = patch_tokens.shape
        grid_side = int(num_patches ** 0.5)  # 14

        contact_resized = F.interpolate(
            contact_mask.unsqueeze(1),  # (B, 1, H, W)
            size=(grid_side, grid_side),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)  # (B, 14, 14)

        weights = contact_resized.reshape(B, num_patches)  # (B, 196)
        weights = F.relu(weights) + 1e-6  # ensure positive
        weights_sum = weights.sum(dim=1, keepdim=True)  # (B, 1)

        pooled = torch.bmm(weights.unsqueeze(1), patch_tokens).squeeze(1) / weights_sum  # (B, 768)
        return pooled

    def forward(self, rgb, rgb_geometric):
        rgb_only = rgb_geometric[:, :3]
        geom_only = rgb_geometric[:, 3:] * self.geometric_channel_scale
        geom_encoded = self.geometric_encoder(geom_only)
        combined_input = torch.cat([rgb_only, geom_encoded], dim=1)
        vit_input = self.rgb_geom_fusion(combined_input)

        # Full ViT output with all patch tokens
        vit_output = self.geometric_vit(vit_input)
        cls_feats = vit_output.last_hidden_state[:, 0, :]  # (B, 768)

        # Contact-weighted pooling of patch tokens
        contact_mask = rgb_geometric[:, 5]  # (B, H, W)
        contact_pooled = self._contact_weighted_pool(vit_output.last_hidden_state, contact_mask)

        with torch.no_grad():
            dino_feats = self.dino(rgb).pooler_output

        combined = torch.cat([cls_feats, contact_pooled, dino_feats], dim=1)
        logits = self.fusion(combined)

        return logits


class FiLMViTBlock(nn.Module):
    """Wraps a single ViT block with FiLM conditioning."""
    def __init__(self, vit_block, t_dim=64, hidden_dim=768):
        super().__init__()
        self.block = vit_block
        self.film_mlp = nn.Sequential(
            nn.Linear(t_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim * 2)
        )

    def forward(self, hidden_states, t_emb=None, head_mask=None, output_attentions=False):
        x = self.block(hidden_states, head_mask=head_mask)

        if t_emb is not None:
            gamma, beta = self.film_mlp(t_emb).chunk(2, dim=-1)
            x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)

        return x


class MultiModalScorerWeightedViTFiLM(MultiModalScorerWeightedVit):
    """
    Multi-modal scorer with FiLM conditioning + contact-weighted pooling.

    Architecture:
        rgb_geometric → geometric_encoder → rgb_geom_fusion
            → FiLMViT ─┬→ CLS token
                        └→ patch tokens → contact-weighted pooling
                                                   ↓
        rgb → DINO → concat → fusion → score

    FiLM conditioning vector is computed by global average pooling of the
    geometric channels (proximity_A, proximity_B, contact) and projecting to t_dim.
    This tells each FiLM-wrapped ViT block what alignment configuration to expect,
    allowing dynamic modulation of self-attention and FFN computations.
    """

    def __init__(self, t_dim=64, film_layers=(8, 9, 10, 11), **kwargs):
        super().__init__(**kwargs)
        self.t_dim = t_dim
        self.film_layers = film_layers

        self.t_emb_proj = nn.Sequential(
            nn.Linear(3, t_dim),
            nn.GELU(),
        )

        for i in film_layers:
            self.geometric_vit.encoder.layer[i] = FiLMViTBlock(
                self.geometric_vit.encoder.layer[i], t_dim
            )

        trainable_film = sum(p.numel() for p in self.t_emb_proj.parameters())
        for i in film_layers:
            trainable_film += sum(p.numel() for p in self.geometric_vit.encoder.layer[i].film_mlp.parameters())
        print(f"FiLM: wrapped layers {film_layers}, t_dim={t_dim}, +{trainable_film:,} params")

    def forward(self, rgb, rgb_geometric):
        rgb_only = rgb_geometric[:, :3]
        geom_only = rgb_geometric[:, 3:] * self.geometric_channel_scale
        geom_encoded = self.geometric_encoder(geom_only)
        combined_input = torch.cat([rgb_only, geom_encoded], dim=1)
        vit_input = self.rgb_geom_fusion(combined_input)

        geom_pooled = rgb_geometric[:, 3:].mean(dim=[2, 3])  # (B, 3)
        t_emb = self.t_emb_proj(geom_pooled)                  # (B, t_dim)

        x = self.geometric_vit.embeddings(vit_input)
        for block in self.geometric_vit.encoder.layer:
            if isinstance(block, FiLMViTBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
        x = self.geometric_vit.layernorm(x)  # (B, 197, 768)

        cls_feats = x[:, 0, :]
        contact_pooled = self._contact_weighted_pool(x, rgb_geometric[:, 5])

        with torch.no_grad():
            dino_feats = self.dino(rgb).pooler_output

        combined = torch.cat([cls_feats, contact_pooled, dino_feats], dim=1)
        logits = self.fusion(combined)

        return logits

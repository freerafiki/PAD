"""
guided_modules.py
-----------------
Two drop-in PyTorch modules for guidance-map-conditioned CNNs.

    SPADE               — spatially-adaptive normalization (Park et al., CVPR 2019)
                          adapted from https://github.com/NVlabs/SPADE (CC BY-NC-SA 4.0)
                          academic / non-commercial use only

    GuidanceGatedConv2d — soft-gated convolution whose gate is informed by an
                          external spatial guidance map (Yu et al., ICCV 2019)

Both expect a guidance map of shape [B, guidance_nc, H, W] with values in [0, 1].
Guidance is automatically resized (bilinear) to match the feature map at each call.

Tested with PyTorch >= 2.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SPADE — Spatially-Adaptive Normalization
# ---------------------------------------------------------------------------

class SPADE(nn.Module):
    """
    Spatially-Adaptive (De)Normalization layer.

    Replaces the affine parameters of a normalization layer with spatially-
    varying gamma and beta predicted from a guidance map. The base norm step
    uses affine=False; ALL learned scale/shift comes from the guidance branch.

    Args:
        norm_nc      (int)  : channels in the feature map to be normalized.
        guidance_nc  (int)  : channels in the guidance map (1 for a float mask).
        norm_type    (str)  : 'group' (default) | 'instance' | 'batch'.
                              'group' is recommended for discriminative encoders —
                              batch-size independent and avoids cross-sample mixing.
        num_groups   (int)  : groups for GroupNorm (only used if norm_type='group').
                              norm_nc must be divisible by num_groups.
        hidden_nc    (int)  : intermediate channels in the guidance MLP.

    Forward:
        x        : [B, norm_nc, H, W]  — feature map to modulate.
        guidance : [B, guidance_nc, H_g, W_g]  — spatial guidance map.
                   Resized bilinearly to (H, W) inside forward().
    Returns:
        [B, norm_nc, H, W]  — modulated feature map.

    Usage note — (1 + gamma) vs gamma:
        The implementation uses out = x_norm * (1 + gamma) + beta.
        This is an initialisation trick confirmed by the original authors (issue #4):
        since conv weights start near zero, gamma ≈ 0 → (1 + gamma) ≈ 1,
        so SPADE is near-identity at init, which stabilises early training.
        Mathematically equivalent to plain gamma once training begins.
    """

    def __init__(
        self,
        norm_nc: int,
        guidance_nc: int = 1,
        norm_type: str = "group",
        num_groups: int = 32,
        hidden_nc: int = 64,
    ):
        super().__init__()

        if norm_type == "group":
            # Batch-size independent; best for discriminative encoders.
            # Falls back gracefully when norm_nc < num_groups.
            g = min(num_groups, norm_nc)
            assert norm_nc % g == 0, (
                f"norm_nc ({norm_nc}) must be divisible by num_groups ({g}). "
                f"Try num_groups=16 or a power of 2 that divides norm_nc."
            )
            self.norm = nn.GroupNorm(g, norm_nc, affine=False)

        elif norm_type == "instance":
            # Per-sample, no cross-sample mixing → cleanest for per-sample guidance.
            # Slightly weaker discriminative statistics than BN/GN.
            self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        elif norm_type == "batch":
            # Requires batch >= ~32 for stable statistics.
            # Conceptually fights per-sample SPADE conditioning at small batches.
            # Use only if batch size is reliably large (>= 32) and GroupNorm
            # underperforms in ablations.
            self.norm = nn.BatchNorm2d(norm_nc, affine=False)

        else:
            raise ValueError(
                f"Unknown norm_type '{norm_type}'. Choose 'group', 'instance', or 'batch'."
            )

        # Lightweight guidance MLP: guidance_map → shared embedding → (gamma, beta)
        # Two 3×3 convs; kernel size 3 preserves spatial structure at all resolutions.
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(guidance_nc, hidden_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        # 1. Normalize (no learned affine — that comes entirely from guidance below)
        x_norm = self.norm(x)

        # 2. Resize guidance to match current feature spatial resolution
        if guidance.shape[2:] != x.shape[2:]:
            guidance = F.interpolate(
                guidance, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        # 3. Predict spatially-varying gamma and beta from guidance
        shared = self.mlp_shared(guidance)
        gamma  = self.mlp_gamma(shared)   # [B, norm_nc, H, W]
        beta   = self.mlp_beta(shared)    # [B, norm_nc, H, W]

        # 4. Modulate — (1 + gamma) initialises as near-identity; see docstring
        return x_norm * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# GuidanceGatedConv2d — guidance-conditioned soft-gated convolution
# ---------------------------------------------------------------------------

class GuidanceGatedConv2d(nn.Module):
    """
    Soft-gated convolution whose gate branch is conditioned on an external
    spatial guidance map (Yu et al., ICCV 2019 — extended with explicit guidance).

    Standard gated conv:  out = phi(W_f * x)  *  sigmoid(W_g * x)
    Guided  gated conv:   out = phi(W_f * x)  *  sigmoid(W_g * cat(x, guidance))

    The feature branch operates on x alone (no guidance) to preserve normal
    CNN semantics. The gate branch sees cat(x, guidance) so it learns where
    in the image the guidance says "pay attention" — and can refine that via
    the feature context — without polluting the feature computation itself.

    Args:
        in_nc        (int)  : input feature channels.
        out_nc       (int)  : output feature channels.
        guidance_nc  (int)  : guidance map channels (1 for a float mask).
        kernel_size  (int)  : conv kernel size (default 3).
        stride       (int)  : stride (default 1).
        padding      (int)  : padding (default 1, keeps spatial size for kernel 3).
        dilation     (int)  : dilation (default 1).
        activation          : activation on the feature branch (default ELU).
                              ELU is preferred over ReLU here: it has non-zero
                              gradient for negative inputs, helping the gate not
                              to kill information in low-guidance regions.

    Forward:
        x        : [B, in_nc, H, W]
        guidance : [B, guidance_nc, H_g, W_g]  — resized inside forward().
    Returns:
        [B, out_nc, H, W]
    """

    def __init__(
        self,
        in_nc: int,
        out_nc: int,
        guidance_nc: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        activation: nn.Module = None,
    ):
        super().__init__()

        self.activation = activation if activation is not None else nn.ELU(inplace=True)

        # Feature branch — sees only x
        self.feature_conv = nn.Conv2d(
            in_nc, out_nc, kernel_size, stride, padding, dilation
        )

        # Gate branch — sees cat(x, guidance), one extra input channel group
        self.gate_conv = nn.Conv2d(
            in_nc + guidance_nc, out_nc, kernel_size, stride, padding, dilation
        )

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        # Resize guidance to match x if needed (e.g. after stride-2 pooling upstream)
        if guidance.shape[2:] != x.shape[2:]:
            guidance = F.interpolate(
                guidance, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        # Feature path
        features = self.activation(self.feature_conv(x))

        # Gate path: guidance informs where to open/close gates
        gate_input = torch.cat([x, guidance], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))

        return features * gate


# ---------------------------------------------------------------------------
# Minimal example: how to use both modules in a ResNet-style encoder
# ---------------------------------------------------------------------------

class GatedResBlock(nn.Module):
    """
    Standard ResBlock where Conv2d → BN → ReLU is replaced with
    GuidanceGatedConv2d, letting the contact-region map gate each convolution.
    BatchNorm (on the residual) is kept outside the gating for discriminative power.
    """
    def __init__(self, nc: int, guidance_nc: int = 1):
        super().__init__()
        self.gconv1 = GuidanceGatedConv2d(nc, nc, guidance_nc)
        self.bn1    = nn.BatchNorm2d(nc)
        self.gconv2 = GuidanceGatedConv2d(nc, nc, guidance_nc)
        self.bn2    = nn.BatchNorm2d(nc)

    def forward(self, x, guidance):
        h = self.bn1(self.gconv1(x, guidance))
        h = self.bn2(self.gconv2(h, guidance))
        return x + h


class SPADEResBlock(nn.Module):
    """
    Standard ResBlock where BatchNorm is replaced by SPADE normalization.
    Convolutions are plain nn.Conv2d; the guidance modulates the norm, not the conv.
    This is the 'newer' approach (no SPADEResBlock class from the original repo):
    SPADE is used as a drop-in norm layer inside any block you want.
    """
    def __init__(self, nc: int, guidance_nc: int = 1):
        super().__init__()
        self.conv1  = nn.Conv2d(nc, nc, 3, padding=1)
        self.spade1 = SPADE(nc, guidance_nc)
        self.conv2  = nn.Conv2d(nc, nc, 3, padding=1)
        self.spade2 = SPADE(nc, guidance_nc)

    def forward(self, x, guidance):
        h = self.conv1(F.relu(self.spade1(x,  guidance), inplace=True))
        h = self.conv2(F.relu(self.spade2(h,  guidance), inplace=True))
        return x + h


# ---------------------------------------------------------------------------
# Minimal training loop sketch
# ---------------------------------------------------------------------------

def _training_loop_example():
    """
    Illustrates how guidance flows through the network during training.
    Replace with your actual dataset, loss, and optimiser.
    """
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build a tiny dual-stream encoder (one stream shown) ----------------
    # Input: 3-channel RGB patch
    # Guidance: 1-channel contact region map, same spatial size as input

    class TinyPuzzleEncoder(nn.Module):
        def __init__(self, use_spade: bool = False):
            super().__init__()
            Block = SPADEResBlock if use_spade else GatedResBlock
            self.stem   = nn.Conv2d(3, 64, 7, stride=2, padding=3)  # 64×H/2×W/2
            self.block1 = Block(64)
            self.down1  = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 128×H/4×W/4
            self.block2 = Block(128)
            self.pool   = nn.AdaptiveAvgPool2d(1)
            self.head   = nn.Linear(128, 1)   # binary compatibility score

        def forward(self, x, guidance):
            f = F.relu(self.stem(x), inplace=True)
            f = self.block1(f, guidance)
            f = F.relu(self.down1(f), inplace=True)
            f = self.block2(f, guidance)
            return self.head(self.pool(f).flatten(1))

    # ---- Instantiate --------------------------------------------------------
    model = TinyPuzzleEncoder(use_spade=False).to(device)   # swap to True for SPADE
    optimiser = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn   = nn.BCEWithLogitsLoss()

    # ---- One training step --------------------------------------------------
    B = 32
    x_a    = torch.randn(B, 3, 64, 64, device=device)  # piece A patch
    guide  = torch.rand( B, 1, 64, 64, device=device)  # contact region map [0,1]
    labels = torch.randint(0, 2, (B, 1), dtype=torch.float, device=device)

    optimiser.zero_grad()
    logits = model(x_a, guide)
    loss   = loss_fn(logits, labels)
    loss.backward()
    optimiser.step()

    print(f"loss={loss.item():.4f}  |  logits range [{logits.min():.2f}, {logits.max():.2f}]")


if __name__ == "__main__":
    _training_loop_example()

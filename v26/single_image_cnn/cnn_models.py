# Adapted from NVlabs/SPADE (Park et al., CVPR 2019)
# https://github.com/NVlabs/SPADE — CC BY-NC-SA 4.0, academic use only

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """
    Spatially-Adaptive Normalization (Park et al., CVPR 2019).
    Modulates BatchNorm/InstanceNorm parameters using a spatial guidance map.

    Args:
        norm_nc:      number of channels in the feature map to normalize
        guidance_nc:  number of channels in the guidance map (1 for your contact map)
        norm_type:    'batch' | 'instance'
        hidden_nc:    intermediate channels in the modulation MLP
    """
    def __init__(self, norm_nc, guidance_nc=1, norm_type='batch', hidden_nc=64):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(f'Unknown norm type: {norm_type}')

        # Small conv MLP: guidance_map → (gamma, beta) at feature resolution
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(guidance_nc, hidden_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, guidance):
        # 1. Normalize (without learned affine — that's the whole point)
        x_norm = self.param_free_norm(x)

        # 2. Resize guidance map to match current feature map resolution
        guidance = F.interpolate(guidance, size=x.shape[2:], mode='bilinear',
                                 align_corners=False)

        # 3. Predict spatially-adaptive scale (gamma) and shift (beta)
        shared = self.mlp_shared(guidance)
        gamma  = self.mlp_gamma(shared)   # [B, C, H, W]
        beta   = self.mlp_beta(shared)    # [B, C, H, W]

        # 4. Modulate — note: (1 + gamma) follows the original paper convention
        return x_norm * (1 + gamma) + beta


class SPADEResBlock(nn.Module):
    """
    ResNet block where BN is replaced by SPADE.
    Can be dropped into any ResNet encoder in place of a standard ResBlock.
    """
    def __init__(self, in_nc, out_nc, guidance_nc=1, norm_type='batch'):
        super().__init__()
        mid_nc = min(in_nc, out_nc)

        self.conv_0 = nn.Conv2d(in_nc,  mid_nc, 3, padding=1)
        self.conv_1 = nn.Conv2d(mid_nc, out_nc, 3, padding=1)

        self.norm_0 = SPADE(in_nc,  guidance_nc, norm_type)
        self.norm_1 = SPADE(mid_nc, guidance_nc, norm_type)

        # Learned shortcut if channel dims differ
        self.shortcut = nn.Conv2d(in_nc, out_nc, 1, bias=False) \
                        if in_nc != out_nc else nn.Identity()
        self.norm_s   = SPADE(in_nc, guidance_nc, norm_type) \
                        if in_nc != out_nc else None

    def forward(self, x, guidance):
        # Shortcut path
        if self.norm_s is not None:
            x_s = self.shortcut(self.norm_s(x, guidance))
        else:
            x_s = self.shortcut(x)

        # Main path
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, guidance), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, guidance), 0.2))

        return x_s + dx


class SPADE(nn.Module):
    def __init__(self, norm_nc, guidance_nc=1,
                 norm_type='instance',   # 'batch' | 'instance' | 'group'
                 num_groups=32,          # only used if norm_type='group'
                 hidden_nc=64):
        super().__init__()

        if norm_type == 'instance':
            # Best for SPADE: per-sample, no cross-sample mixing
            # Slightly weaker discriminability
            self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        elif norm_type == 'batch':
            # Requires large batch (≥32); fights per-sample SPADE conditioning
            # but offers strongest discriminative statistics
            self.norm = nn.BatchNorm2d(norm_nc, affine=False,
                                       track_running_stats=True)

        elif norm_type == 'group':
            # Best compromise: batch-size independent, partial per-sample isolation
            # num_groups=32 is standard; norm_nc must be divisible by num_groups
            g = min(num_groups, norm_nc)  # guard against small channel counts
            self.norm = nn.GroupNorm(g, norm_nc, affine=False)

        else:
            raise ValueError(f'Unknown norm_type: {norm_type}')

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(guidance_nc, hidden_nc, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)
        self.mlp_beta  = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)

    def forward(self, x, guidance):
        x_norm = self.norm(x)
        g = F.interpolate(guidance, size=x.shape[2:],
                          mode='bilinear', align_corners=False)
        shared = self.mlp_shared(g)
        gamma  = self.mlp_gamma(shared)
        beta   = self.mlp_beta(shared)
        return x_norm * (1 + gamma) + beta

# Replace standard ResBlocks with SPADEResBlocks
# guidance shape: [B, 1, H, W]  (your contact region map, 0→1 float)

class PuzzleStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem  = nn.Conv2d(3, 64, 7, stride=2, padding=3)   # RGB in
        self.block1 = SPADEResBlock(64,  128, guidance_nc=1)
        self.block2 = SPADEResBlock(128, 256, guidance_nc=1)
        self.pool   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, contact_map):
        f = self.stem(x)
        f = self.block1(f, contact_map)   # guidance auto-resized inside SPADE
        f = F.max_pool2d(f, 2)
        f = self.block2(f, contact_map)
        return self.pool(f).flatten(1)


# Adapted from Yu et al., ICCV 2019 — Free-Form Image Inpainting with Gated Convolution
# Reference impl: github.com/avalonstrel/GatedConvolution_pytorch

class GatedConv2d(nn.Module):
    """
    Standard gated convolution (Yu et al. 2019).
    Gate is computed from the input features alone — the original formulation.
    """
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1,
                 padding=1, dilation=1, activation=nn.ELU(inplace=True)):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_nc, out_nc, kernel_size,
                                      stride, padding, dilation)
        self.gate_conv    = nn.Conv2d(in_nc, out_nc, kernel_size,
                                      stride, padding, dilation)
        self.activation   = activation

    def forward(self, x):
        features = self.activation(self.feature_conv(x))
        gate     = torch.sigmoid(self.gate_conv(x))
        return features * gate


class GuidanceGatedConv2d(nn.Module):
    """
    Guidance-conditioned gated convolution.
    The gate branch takes concat(features, guidance_map) as input,
    so the network learns to open/close gates based on the contact region.
    This is the variant directly relevant to the puzzle task.
    """
    def __init__(self, in_nc, out_nc, guidance_nc=1, kernel_size=3,
                 stride=1, padding=1, activation=nn.ELU(inplace=True)):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_nc, out_nc,
                                      kernel_size, stride, padding)
        # Gate input = feature channels + guidance channels
        self.gate_conv    = nn.Conv2d(in_nc + guidance_nc, out_nc,
                                      kernel_size, stride, padding)
        self.activation   = activation

    def forward(self, x, guidance):
        # Resize guidance to match x spatial dims
        g = F.interpolate(guidance, size=x.shape[2:],
                          mode='bilinear', align_corners=False)
        features  = self.activation(self.feature_conv(x))
        gate_in   = torch.cat([x, g], dim=1)
        gate      = torch.sigmoid(self.gate_conv(gate_in))
        return features * gate

        
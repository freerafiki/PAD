"""
# This version projects the geometric + RGB information (6 channels)
# into 3 channels

Pros:
- Simple, clean
- Leverages fully pretrained ViT weights
- Fast to train (only projection layer starts random)
-Less prone to overfitting

Cons:
- Bottleneck: 6 channels compressed to 3 before ViT sees them
- Information loss: geometric features mixed into RGB space
- ViT can't learn channel-specific processing
"""
class ProjectionViT(nn.Module):
    def __init__(self, pretrained_name='google/vit-base-patch16-224'):
        super().__init__()
        self.projection = nn.Conv2d(6, 3, kernel_size=1)
        self.vit = ViTModel.from_pretrained(pretrained_name)
    
    def forward(self, x):
        # x: (B, 6, H, W)
        x_3ch = self.projection(x)  # (B, 3, H, W)
        outputs = self.vit(x_3ch)
        return outputs.pooler_output  # (B, 768)

"""
This version modifies the first layer to accept 6-channels input

Pros:
- No bottleneck: ViT processes all 6 channels natively
- Channel-specific learning: each channel gets dedicated filters
- Better theoretical capacity

Cons:
- 3 channels start random (geometric features)
- Might need more data to train effectively
- Slightly higher risk of overfitting
"""
class ModifiedViT(nn.Module):
    def __init__(self, pretrained_name='google/vit-base-patch16-224'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_name)
        
        # Get original patch embedding
        original_proj = self.vit.embeddings.patch_embeddings.projection
        
        # Create new one with 6 input channels
        self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=6,
            out_channels=original_proj.out_channels,  # 768 for base
            kernel_size=original_proj.kernel_size,     # (16, 16)
            stride=original_proj.stride                # (16, 16)
        )
        
        # Initialize: copy RGB weights, random for geometric channels
        with torch.no_grad():
            new_weight = self.vit.embeddings.patch_embeddings.projection.weight
            new_weight[:, :3, :, :] = original_proj.weight.data
            # Channels 3-5 stay randomly initialized
    
    def forward(self, x):
        # x: (B, 6, H, W)
        outputs = self.vit(x)
        return outputs.pooler_output  # (B, 768)
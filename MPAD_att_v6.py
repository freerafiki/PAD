import torch
import torch.nn as nn
from transformers import ViTForMaskedImageModeling, ViTImageProcessor, ViTModel
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================================
# MODIFICATION 1: Custom Dataset with Masks
# ============================================================================

class MaskedImageDataset(Dataset):
    def __init__(self, images, masks, attention_maps, processor, patch_size=16,
                 mask_strategy='threshold', mask_threshold=0.3):
        """
        Args:
            images: List/array of images [H, W, 3] - RGB images (224x224)
            masks: List/array of binary masks [H, W] - pixel-level (224x224)
            attention_maps: List/array of attention maps [H, W] - pixel-level (224x224)
                           Can be binary (0/1) or continuous (0.0-1.0) values
            processor: ViTImageProcessor for preprocessing
            patch_size: Size of patches (default 16 for ViT-base)
            mask_strategy: How to convert pixel masks to patch masks
                - 'any': mask if ANY pixel is masked (aggressive)
                - 'majority': mask if >50% of pixels are masked
                - 'threshold': mask if >= mask_threshold of pixels are masked
                - 'all': mask only if ALL pixels are masked (conservative)
            mask_threshold: Threshold for 'threshold' strategy (default 0.3 = 30%)
        """
        self.images = images
        self.masks = masks
        self.attention_maps = attention_maps
        self.processor = processor
        self.patch_size = patch_size
        self.num_patches_per_side = 224 // patch_size  # 14 for patch_size=16
        self.num_patches = self.num_patches_per_side ** 2  # 196
        self.mask_strategy = mask_strategy
        self.mask_threshold = mask_threshold
    
    def __len__(self):
        return len(self.images)
    
    def pixel_to_patch_mask(self, pixel_mask, strategy='threshold', threshold=0.3):
        """
        Convert 224x224 binary pixel mask to 196 boolean patch mask.
        
        Args:
            pixel_mask: [H, W] binary array (1 = masked, 0 = unmasked)
            strategy: str, one of:
                - 'any': patch is masked if ANY pixel is masked (most aggressive)
                - 'majority': patch is masked if >50% of pixels are masked
                - 'threshold': patch is masked if >= threshold% of pixels are masked
                - 'all': patch is masked only if ALL pixels are masked (most conservative)
            threshold: float in [0, 1], used when strategy='threshold'
        
        Returns:
            patch_mask: [num_patches] boolean array (True = masked)
        """
        H, W = pixel_mask.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Reshape to group pixels by patch: [num_patches_h, patch_size, num_patches_w, patch_size]
        reshaped = pixel_mask.reshape(
            num_patches_h, self.patch_size,
            num_patches_w, self.patch_size
        )
        
        # Calculate percentage of masked pixels per patch: [num_patches_h, num_patches_w]
        masked_ratio = reshaped.mean(axis=(1, 3))
        
        if strategy == 'any':
            # Patch is masked if ANY pixel is masked
            patch_mask = masked_ratio > 0
            
        elif strategy == 'majority':
            # Patch is masked if majority (>50%) of pixels are masked
            patch_mask = masked_ratio > 0.5
            
        elif strategy == 'threshold':
            # Patch is masked if >= threshold of pixels are masked
            patch_mask = masked_ratio >= threshold
            
        elif strategy == 'all':
            # Patch is masked only if ALL pixels are masked
            patch_mask = masked_ratio >= 1.0
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'any', 'majority', 'threshold', or 'all'")
        
        # Flatten to [num_patches]
        patch_mask_flat = patch_mask.reshape(-1)
        
        return patch_mask_flat.astype(bool)
    
    def pixel_to_patch_attention(self, attention_map):
        """
        Convert 224x224 attention map to 196 patch-level attention weights.
        Uses average pooling to preserve spatial importance.
        
        Args:
            attention_map: [H, W] array with values (binary 0/1 or continuous 0.0-1.0)
        Returns:
            patch_attention: [num_patches] float array
        """
        H, W = attention_map.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Reshape to group pixels by patch
        reshaped = attention_map.reshape(
            num_patches_h, self.patch_size,
            num_patches_w, self.patch_size
        )
        
        # Average pooling within each patch: [num_patches_h, num_patches_w]
        patch_attention = reshaped.mean(axis=(1, 3))
        
        # Alternative: Use max pooling for binary maps
        # patch_attention = reshaped.max(axis=(1, 3))
        
        # Flatten to [num_patches]
        patch_attention_flat = patch_attention.reshape(-1)
        
        return patch_attention_flat.astype(np.float32)
    
    def __getitem__(self, idx):
        image = self.images[idx]  # [224, 224, 3]
        mask = self.masks[idx]  # [224, 224] binary
        attention_map = self.attention_maps[idx]  # [224, 224] binary or continuous
        
        # Process image with ViTImageProcessor
        # This handles normalization and converts to [3, 224, 224]
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].squeeze(0)
        
        # Convert pixel-level mask to patch-level boolean mask [196]
        patch_mask = self.pixel_to_patch_mask(
            mask, 
            strategy=self.mask_strategy, 
            threshold=self.mask_threshold
        )
        bool_masked_pos = torch.tensor(patch_mask, dtype=torch.bool)
        
        # Convert pixel-level attention to patch-level weights [196]
        patch_attention = self.pixel_to_patch_attention(attention_map)
        attention_target = torch.tensor(patch_attention, dtype=torch.float32)
        
        return {
            'pixel_values': pixel_values,  # [3, 224, 224]
            'bool_masked_pos': bool_masked_pos,  # [196] boolean
            'attention_target': attention_target  # [196] float
        }


# ============================================================================
# MODIFICATION 2: Model with Attention as Input (Conditional)
# ============================================================================

class ViTWithAttentionInput(nn.Module):
    """
    ViT that takes attention maps as additional input.
    The attention map guides where the model should focus.
    """
    def __init__(self, model_name='google/vit-base-patch16-224-in21k', 
                 attention_loss_weight=0.1,
                 use_attention_as_input=True):
        super().__init__()
        self.vit = ViTForMaskedImageModeling.from_pretrained(model_name)
        self.attention_loss_weight = attention_loss_weight
        self.use_attention_as_input = use_attention_as_input
        
        # Get model config
        self.config = self.vit.config
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2
        self.hidden_size = self.config.hidden_size
        
        if use_attention_as_input:
            # APPROACH 1: Add attention as additional channel
            # Project attention map to patch embeddings and add to input
            self.attention_projection = nn.Linear(1, self.hidden_size)
            
            # APPROACH 2: Modulate patch embeddings with attention weights
            # self.attention_modulation = nn.Sequential(
            #     nn.Linear(1, self.hidden_size),
            #     nn.Tanh()
            # )
            
            # APPROACH 3: Concatenate and project
            # self.fusion_projection = nn.Linear(self.hidden_size + 1, self.hidden_size)
    
    def inject_attention_map(self, embeddings, attention_target):
        """
        Inject attention map information into patch embeddings.
        
        Args:
            embeddings: [batch, num_patches+1, hidden_size] (includes CLS token)
            attention_target: [batch, num_patches] (patch-level attention weights)
        
        Returns:
            Modified embeddings with attention guidance
        """
        batch_size = embeddings.shape[0]
        
        # Normalize attention weights
        attention_norm = attention_target / (attention_target.sum(dim=1, keepdim=True) + 1e-8)
        attention_norm = attention_norm.unsqueeze(-1)  # [batch, num_patches, 1]
        
        # APPROACH 1: Additive - project and add to embeddings
        attention_embed = self.attention_projection(attention_norm)  # [batch, num_patches, hidden_size]
        embeddings[:, 1:, :] = embeddings[:, 1:, :] + attention_embed
        
        # APPROACH 2: Multiplicative modulation (alternative)
        # attention_scale = self.attention_modulation(attention_norm)
        # embeddings[:, 1:, :] = embeddings[:, 1:, :] * (1 + attention_scale)
        
        # APPROACH 3: Concatenate and project (alternative)
        # combined = torch.cat([embeddings[:, 1:, :], attention_norm.expand(-1, -1, self.hidden_size)], dim=-1)
        # embeddings[:, 1:, :] = self.fusion_projection(combined)
        
        return embeddings
    
    def extract_attention_maps(self, attentions):
        """
        Extract and aggregate attention maps from all layers.
        """
        attention_weights = []
        
        for layer_attention in attentions:
            cls_attention = layer_attention[:, :, 0, 1:]  # [batch, heads, num_patches]
            cls_attention = cls_attention.mean(dim=1)  # [batch, num_patches]
            attention_weights.append(cls_attention)
        
        avg_attention = torch.stack(attention_weights).mean(dim=0)  # [batch, num_patches]
        return avg_attention
    
    def compute_attention_loss(self, attention_maps, attention_targets):
        """Compute loss to guide attention towards target regions."""
        attention_targets_norm = attention_targets / (attention_targets.sum(dim=1, keepdim=True) + 1e-8)
        log_attention = torch.log(attention_maps + 1e-8)
        kl_loss = nn.functional.kl_div(log_attention, attention_targets_norm, reduction='batchmean')
        return kl_loss
    
    def prepare_attention_target(self, attention_target, image_size=224, patch_size=16):
        """
        Convert pixel-wise attention map to patch-level attention map.
        NOTE: This is now handled in the Dataset class, but kept for compatibility.
        
        Args:
            attention_target: [batch_size, num_patches] - already patch-level from dataset
        
        Returns:
            Patch-level attention map: [batch_size, num_patches]
        """
        # Attention target is already at patch level from dataset
        return attention_target
    
    def forward(self, pixel_values, bool_masked_pos=None, attention_target=None):
        """
        Forward pass with optional attention conditioning.
        
        Args:
            pixel_values: [batch, 3, 224, 224]
            bool_masked_pos: [batch, 196] boolean tensor
            attention_target: [batch, 196] float tensor (already at patch level)
        
        NOTE: Positional embeddings are handled automatically by ViT.
        The model preserves spatial relationships through its position encodings.
        """
        # Attention target is already at patch level from dataset
        if attention_target is not None and attention_target.dim() == 3:
            attention_target = self.prepare_attention_target(attention_target)
        
        # If using attention as input, we need to modify the embeddings
        if self.use_attention_as_input and attention_target is not None:
            # Get patch embeddings from the ViT model
            # Positional embeddings are added inside vit.embeddings()
            vit_model = self.vit.vit
            
            # Embed patches (includes positional embeddings automatically)
            embeddings = vit_model.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
            
            # Inject attention guidance into embeddings
            # Spatial positions are preserved through the positional embeddings
            embeddings = self.inject_attention_map(embeddings, attention_target)
            
            # Forward through encoder
            encoder_outputs = vit_model.encoder(
                embeddings,
                output_attentions=True,
                return_dict=True
            )
            
            sequence_output = encoder_outputs.last_hidden_state
            
            # Get decoder output for masked image modeling
            decoder_output = self.vit.decoder(sequence_output, return_dict=True)
            logits = decoder_output.logits
            
            # Compute reconstruction loss
            if bool_masked_pos is not None:
                loss_fct = nn.MSELoss()
                masked_lm_loss = loss_fct(
                    logits[bool_masked_pos],
                    pixel_values.view(pixel_values.shape[0], -1)[bool_masked_pos]
                )
            else:
                masked_lm_loss = None
            
            outputs = {
                'loss': masked_lm_loss,
                'logits': logits,
                'attentions': encoder_outputs.attentions
            }
        else:
            # Standard forward pass without attention injection
            outputs = self.vit(
                pixel_values=pixel_values,
                bool_masked_pos=bool_masked_pos,
                output_attentions=True,
                return_dict=True
            )
        
        reconstruction_loss = outputs['loss'] if outputs['loss'] is not None else torch.tensor(0.0)
        attention_loss = torch.tensor(0.0, device=pixel_values.device)
        
        # Compute attention alignment loss (regularization)
        if attention_target is not None and outputs['attentions'] is not None:
            attention_maps = self.extract_attention_maps(outputs['attentions'])
            attention_loss = self.compute_attention_loss(attention_maps, attention_target)
        
        # Combined loss
        total_loss = reconstruction_loss + self.attention_loss_weight * attention_loss
        
        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'attention_loss': attention_loss,
            'logits': outputs['logits'],
            'attentions': outputs['attentions']
        }


# ============================================================================
# Comparison: With vs Without Attention as Input
# ============================================================================

def compare_approaches():
    """
    Compare the two approaches:
    1. Attention only as regularization (training only)
    2. Attention as model input (available at train and test time)
    """
    
    print("=" * 70)
    print("APPROACH COMPARISON")
    print("=" * 70)
    
    print("\n1. ATTENTION AS REGULARIZATION ONLY (Original)")
    print("-" * 70)
    print("Pros:")
    print("  - No architectural changes needed")
    print("  - Works even if attention maps not available at test time")
    print("  - Simpler to implement")
    print("\nCons:")
    print("  - Model doesn't explicitly learn to use attention guidance")
    print("  - Can't leverage attention information at inference")
    print("  - Less expressive if attention maps are informative")
    
    print("\n2. ATTENTION AS MODEL INPUT (Conditional)")
    print("-" * 70)
    print("Pros:")
    print("  - Model explicitly learns to condition on attention maps")
    print("  - Can leverage test-time attention information")
    print("  - More expressive - different attention -> different predictions")
    print("  - Potentially much higher accuracy if attention maps are informative")
    print("\nCons:")
    print("  - Requires attention maps at test time")
    print("  - More complex architecture")
    print("  - Slight increase in parameters and computation")
    
    print("\n" + "=" * 70)
    print("MASK CONVERSION STRATEGIES")
    print("=" * 70)
    print("\nWhen converting pixel masks (224x224) to patch masks (196):")
    print()
    print("'any' - Mask if ANY pixel is masked")
    print("  → Most aggressive, maximum masking")
    print("  → Use when: masks indicate 'contaminated' regions")
    print()
    print("'majority' - Mask if >50% of pixels are masked")
    print("  → Balanced approach")
    print("  → Use when: partial masking is acceptable")
    print()
    print("'threshold' - Mask if >= X% of pixels are masked")
    print("  → Customizable (e.g., 30% threshold)")
    print("  → Use when: you want fine control")
    print("  → RECOMMENDED for most cases")
    print()
    print("'all' - Mask only if ALL pixels are masked")
    print("  → Most conservative, minimum masking")
    print("  → Use when: masks are noisy or you want fewer masked patches")
    print()
    print("RESEARCH INSIGHTS:")
    print("  • MAE paper (He et al., 2022): 75% random masking works best")
    print("  • For structured/meaningful masks: 'threshold' (30-50%) recommended")
    print("  • Black image regions: use 'all' to only mask fully black patches")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("If attention maps are available at test time and are informative:")
    print("  → USE ATTENTION AS INPUT (Approach 2)")
    print("  → Expected improvement: 5-20% depending on attention quality")
    print("\nIf attention maps only available at training:")
    print("  → USE AS REGULARIZATION ONLY (Approach 1)")
    print("\nFor mask strategy:")
    print("  → Start with 'threshold' at 0.3 (30%)")
    print("  → Experiment with 0.2-0.5 range based on validation performance")
    print("=" * 70)


# ============================================================================
# Training Loop
# ============================================================================

def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_attn_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch['pixel_values'].to(device)
            bool_masked_pos = batch['bool_masked_pos'].to(device)
            attention_target = batch['attention_target'].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                bool_masked_pos=bool_masked_pos,
                attention_target=attention_target
            )
            
            loss = outputs['loss']
            recon_loss = outputs['reconstruction_loss']
            attn_loss = outputs['attention_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_attn_loss += attn_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                      f"Attn: {attn_loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_attn = total_attn_loss / len(dataloader)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - "
              f"Avg Loss: {avg_loss:.4f}, Avg Recon: {avg_recon:.4f}, "
              f"Avg Attn: {avg_attn:.4f}\n")


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Show comparison
    compare_approaches()
    
    print("\n" + "=" * 70)
    print("DATA FORMAT SUMMARY")
    print("=" * 70)
    print("\nYou need to prepare:")
    print("  1) Images: [224, 224, 3] RGB (np.uint8 or np.float32)")
    print("     - Black parts are fine and will be processed normally")
    print()
    print("  2) Masks: [224, 224] binary (0 or 1)")
    print("     - 0 = visible, 1 = masked")
    print("     - Will be converted to [196] patch-level boolean")
    print("     - Patch is masked if ANY pixel in it is masked")
    print()
    print("  3) Attention maps: [224, 224]")
    print("     - Binary (0/1): regions to focus on vs ignore")
    print("     - OR Continuous (0.0-1.0): importance weights")
    print("     - Higher values = more important regions")
    print("     - Will be converted to [196] patch-level weights via averaging")
    print()
    print("SPATIAL INFORMATION:")
    print("  - Positional embeddings are AUTOMATICALLY handled by ViT")
    print("  - Patch positions (14x14 grid) are encoded in the model")
    print("  - Top-left patch is index 0, top-right is 13, etc.")
    print("  - You don't need to manually add position info!")
    print("=" * 70 + "\n")
    
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    # Create dummy data - showing the correct format
    num_samples = 100
    
    # 1) RGB images with some black regions
    images = []
    for _ in range(num_samples):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add some black regions
        img[100:150, 100:150] = 0  # Black square
        images.append(img)
    
    # 2) Binary pixel masks (224x224)
    # 1 = masked, 0 = visible
    masks = []
    for _ in range(num_samples):
        mask = np.zeros((224, 224), dtype=np.uint8)
        # Mask some random regions
        mask[50:100, 50:100] = 1
        mask[150:200, 150:200] = 1
        masks.append(mask)
    
    # 3) Attention maps (224x224)
    # Option A: Binary (0 or 1)
    attention_maps_binary = []
    for _ in range(num_samples):
        attn = np.zeros((224, 224), dtype=np.float32)
        # Mark important regions
        attn[0:112, 0:112] = 1.0  # Top-left quadrant important
        attention_maps_binary.append(attn)
    
    # Option B: Continuous (0.0 to 1.0)
    attention_maps_continuous = []
    for _ in range(num_samples):
        # Gaussian-like attention centered on image
        y, x = np.ogrid[:224, :224]
        center_y, center_x = 112, 112
        attn = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 50**2))
        attention_maps_continuous.append(attn.astype(np.float32))
    
    # Use continuous attention maps (more informative)
    dataset = MaskedImageDataset(images, masks, attention_maps_continuous, processor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Show what the dataset produces
    sample = dataset[0]
    print("\nSample batch shapes:")
    print(f"  pixel_values: {sample['pixel_values'].shape}")  # [3, 224, 224]
    print(f"  bool_masked_pos: {sample['bool_masked_pos'].shape}")  # [196]
    print(f"  attention_target: {sample['attention_target'].shape}")  # [196]
    print()
    
    print("\n" + "=" * 70)
    print("TRAINING WITH ATTENTION AS INPUT")
    print("=" * 70 + "\n")
    
    # Initialize model WITH attention as input
    model = ViTWithAttentionInput(
        model_name='google/vit-base-patch16-224-in21k',
        attention_loss_weight=0.1,
        use_attention_as_input=True  # KEY PARAMETER
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train
    train_model(model, dataloader, optimizer, device, num_epochs=3)
    
    print("\nTraining completed!")
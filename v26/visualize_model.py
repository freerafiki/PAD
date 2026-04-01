"""
visualize_model.py

Visualize model predictions, scores, and attention maps.
Usage:
    python visualize_model.py --model checkpoints/geometric_best.pth --model_type geometric
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
from sklearn.decomposition import PCA
import cv2

from dataset_v3 import PrecomposedAlignmentDataset, collate_alignment_samples, ShuffledBatchSampler
from models import BaselineScorer, GeometricScorer, MultiModalScorer #MultiModalScorerV2_Practical
from torch.utils.data import DataLoader


def load_model(checkpoint_path, model_type, device='cuda'):
    """
    Load a trained model from checkpoint.
    Handles both old checkpoints (with numpy) and new clean checkpoints.
    """
    # Initialize model
    if model_type == 'baseline':
        model = BaselineScorer()
    elif model_type == 'geometric':
        model = GeometricScorer()
    elif model_type == 'multimodal':
        model = MultiModalScorer() #MultiModalScorerV2_Practical()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint_path = Path(checkpoint_path)

    # Try to load with weights_only=True first (safer, for new checkpoints)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print("Loaded checkpoint with weights_only=True (safe mode)")
    except Exception as e:
        # Fall back to weights_only=False for old checkpoints
        print(f"Safe load failed, using weights_only=False (your own checkpoint, should be safe)")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both formats: full checkpoint dict or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

        # Print metadata if available
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_accuracy' in checkpoint:
            print(f"  Val Accuracy: {checkpoint['val_accuracy']:.3f}")
    else:
        # Assume it's just the state_dict
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Loaded {model_type} model from {checkpoint_path}")

    return model

def denormalize_image(tensor):
    """
    Denormalize image tensor for visualization.

    Args:
        tensor: (C, H, W) normalized tensor

    Returns:
        image: (H, W, C) numpy array in [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)

    return img

"""
visualize_model.py - Updated with attention map visualization
"""



# def extract_attention_maps(model, rgb, rgb_geometric, model_type='geometric'):
#     """
#     Extract attention maps from ViT model.
    
#     Args:
#         model: The trained model
#         rgb: (B, 3, H, W) RGB input
#         rgb_geometric: (B, 6, H, W) RGB+geometric input
#         model_type: 'baseline', 'geometric', or 'multimodal'
    
#     Returns:
#         attention_maps: (B, H, W) attention heatmaps
#     """
#     model.eval()
    
#     # Hook to capture attention weights
#     attention_weights = []
    
#     def attention_hook(module, input, output):
#         # output is a tuple: (attention_output, attention_weights)
#         # attention_weights shape: (B, num_heads, num_patches, num_patches)
#         if isinstance(output, tuple) and len(output) > 1:
#             attn = output[1]
#             attention_weights.append(attn)
    
#     # Register hooks on the last attention layer
#     # For ViT, we want the last encoder layer's attention
#     if model_type == 'baseline':
#         vit_model = model.vit
#     elif model_type == 'geometric':
#         vit_model = model.geometric_vit
#     elif model_type == 'multimodal':
#         vit_model = model.geometric_vit
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")
    
#     # Get last encoder layer
#     last_layer = vit_model.encoder.layer[-1].attention.attention
#     hook_handle = last_layer.register_forward_hook(attention_hook)
    
#     # Forward pass
#     with torch.no_grad():
#         if model_type == 'baseline':
#             _ = model(rgb_geometric)
#         elif model_type == 'geometric':
#             _ = model(rgb_geometric)
#         elif model_type == 'multimodal':
#             _ = model(rgb, rgb_geometric)
    
#     # Remove hook
#     hook_handle.remove()
    
#     # Process attention weights
#     if len(attention_weights) == 0:
#         # Fallback: return empty maps
#         B = rgb.shape[0]
#         return torch.zeros(B, 14, 14)
    
#     # Get attention from last layer
#     # Shape: (B, num_heads, num_patches, num_patches)
#     attn = attention_weights[-1]
    
#     # Average over heads
#     attn = attn.mean(dim=1)  # (B, num_patches, num_patches)
    
#     # Get attention from CLS token to all patches
#     # attn[:, 0, 1:] gives attention from CLS to all image patches
#     cls_attn = attn[:, 0, 1:]  # (B, num_patches - 1)
    
#     # Reshape to spatial grid
#     # For ViT-Base with 224x224 input and patch size 16: 14x14 patches
#     num_patches = int(np.sqrt(cls_attn.shape[1]))
#     attn_maps = cls_attn.reshape(-1, num_patches, num_patches)  # (B, 14, 14)
    
#     return attn_maps
def extract_attention_maps(model, rgb, rgb_geometric, model_type='geometric'):
    """
    Extract attention maps from ViT model.
    
    IMPORTANT: This modifies the forward pass temporarily to output attentions.
    """
    model.eval()
    
    # Determine which ViT to use
    if model_type == 'baseline':
        vit_model = model.vit
    elif model_type == 'geometric':    
        vit_model = model.vit
    elif model_type == 'multimodal':
        vit_model = model.geometric_vit
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Prepare input for ViT
    if model_type == 'baseline' or model_type == 'geometric':
        # These models process rgb_geometric through projection
        if hasattr(model, 'projection'):
            vit_input = model.projection(rgb_geometric)
        else:
            vit_input = rgb_geometric[:, :3]  # Just take RGB if no projection
    elif model_type == 'multimodal':
        # Multimodal processes geometric features first
        with torch.no_grad():
            rgb_only = rgb_geometric[:, :3]
            geom_only = rgb_geometric[:, 3:]
            geom_encoded = model.geometric_encoder(geom_only)
            combined_input = torch.cat([rgb_only, geom_encoded], dim=1)
            vit_input = model.rgb_geom_fusion(combined_input)
    
    # *** KEY CHANGE: Request attention outputs ***
    with torch.no_grad():
        outputs = vit_model(vit_input, output_attentions=True)
    
    # outputs.attentions is a tuple of attention tensors, one per layer
    # Each has shape: (batch_size, num_heads, sequence_length, sequence_length)
    
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        print("⚠️  Warning: Model did not output attentions!")
        B = rgb.shape[0]
        return torch.zeros(B, 14, 14, device=rgb.device)
    
    # Get attention from LAST layer
    last_layer_attn = outputs.attentions[-1]  # (B, num_heads, seq_len, seq_len)
    
    # Average over attention heads
    attn = last_layer_attn.mean(dim=1)  # (B, seq_len, seq_len)
    
    # Get attention from CLS token (index 0) to all patch tokens (index 1:)
    cls_attn = attn[:, 0, 1:]  # (B, num_patches)
    
    # Reshape to 2D grid
    # ViT-Base with 224x224 input and patch_size=16 → 14×14 patches
    num_patches_side = int(np.sqrt(cls_attn.shape[1]))
    attn_maps = cls_attn.reshape(-1, num_patches_side, num_patches_side)  # (B, 14, 14)
    
    return attn_maps


def extract_dino_features(model, rgb):
    """
    Extract DINO features from the multimodal model.

    Args:
        model: MultiModalScorer model with DINO branch
        rgb: (B, 3, H, W) RGB input

    Returns:
        feature_maps: List of (H, W, 3) numpy arrays for visualization
    """
    model.eval()
    with torch.no_grad():
        # Get DINO outputs with hidden states
        dino_output = model.dino(rgb, output_hidden_states=True)
        
        # Get the last hidden state: (B, num_patches + 1, hidden_dim)
        # num_patches = (224/16)^2 = 196 for vit-base-patch16-224
        hidden_states = dino_output.last_hidden_state
        
        # Remove CLS token, keep only patch tokens
        patch_tokens = hidden_states[:, 1:, :]  # (B, num_patches, hidden_dim)
        
        feature_maps = []
        for i in range(patch_tokens.shape[0]):
            # Get patch tokens for this image
            tokens = patch_tokens[i].cpu().numpy()  # (num_patches, hidden_dim)
            
            # Reshape to spatial grid (14x14 for 224x224 image with patch size 16)
            num_patches = tokens.shape[0]
            grid_size = int(np.sqrt(num_patches))
            
            if grid_size * grid_size != num_patches:
                # Fallback: just use PCA on all tokens
                grid_size = int(np.ceil(np.sqrt(num_patches)))
            
            # Use PCA to reduce to 3 components for RGB visualization
            pca = PCA(n_components=3)
            features_3d = pca.fit_transform(tokens)  # (num_patches, 3)
            
            # Normalize to [0, 1]
            features_3d = (features_3d - features_3d.min()) / (features_3d.max() - features_3d.min() + 1e-8)
            
            # Reshape to grid
            feature_map = features_3d.reshape(grid_size, grid_size, 3)
            
            # Upsample to match image size for better visualization
            feature_map_pil = Image.fromarray((feature_map * 255).astype(np.uint8))
            feature_map_upsampled = feature_map_pil.resize((224, 224), Image.BILINEAR)
            feature_map_np = np.array(feature_map_upsampled) / 255.0
            
            feature_maps.append(feature_map_np)
    
    return feature_maps

def extract_dino_attention(model, rgb):
    """
    Extract attention maps from frozen DINO.
    """
    model.eval()
    
    # *** KEY: Request attentions ***
    with torch.no_grad():
        outputs = model.dino(rgb, output_attentions=True)
    
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        print("⚠️  Warning: DINO did not output attentions!")
        B = rgb.shape[0]
        return torch.zeros(B, 14, 14, device=rgb.device)
    
    # Process same as ViT
    last_layer_attn = outputs.attentions[-1]
    attn = last_layer_attn.mean(dim=1)
    cls_attn = attn[:, 0, 1:]
    
    num_patches_side = int(np.sqrt(cls_attn.shape[1]))
    attn_maps = cls_attn.reshape(-1, num_patches_side, num_patches_side)
    
    return attn_maps


def overlay_attention_on_image(image, attention_map, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlay attention heatmap on image.
    
    Args:
        image: (H, W, 3) numpy array in [0, 1]
        attention_map: (H_attn, W_attn) numpy array
        alpha: Blending factor
        colormap: OpenCV colormap
    
    Returns:
        overlayed: (H, W, 3) numpy array in [0, 1]
    """
    H, W = image.shape[:2]
    
    # Resize attention map to image size
    attention_resized = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 255]
    attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
    attention_normalized = (attention_normalized * 255).astype(np.uint8)
    
    # breakpoint()

    # Apply colormap
    heatmap = cv2.applyColorMap(attention_normalized, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay on image
    overlayed = alpha * heatmap + (1 - alpha) * image
    overlayed = np.clip(overlayed, 0, 1)
    
    return overlayed


def denormalize_image(tensor):
    """Denormalize image tensor for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    return img


def visualize_batch(model, batch, device, save_dir, batch_idx=0, model_type='geometric', threshold=0.5):
    """
    Visualize predictions for one batch with attention maps.
    
    Rows:
    0. RGB image
    1. DINO attention (for multimodal) or ViT attention (for others)
    2. Contact region
    3. ViT attention maps (shows where geometric ViT is looking)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    rgb = batch['rgb'].to(device)
    rgb_geometric = batch['rgb_geometric'].to(device)
    labels = batch['labels'].cpu().numpy()
    difficulties = batch['difficulties']
    positions = batch['positions']
    group_sizes = batch['group_sizes']  
    
    # Get predictions
    with torch.no_grad():
        if model_type == 'baseline':
            scores = torch.sigmoid(model(rgb)).squeeze()
        elif model_type == 'multimodal':
            scores = torch.sigmoid(model(rgb, rgb_geometric)).squeeze()
        else:
            scores = torch.sigmoid(model(rgb_geometric)).squeeze()
    
    scores_np = scores.cpu().numpy()
    
    # Extract attention maps
    print(f"Extracting attention maps for {model_type} model...")
    vit_attention = extract_attention_maps(model, rgb, rgb_geometric, model_type)
    
    # Extract DINO features if using multimodal model
    dino_features = None
    if model_type == 'multimodal':
        dino_features = extract_dino_features(model, rgb)
    
    dino_attention = None
    if model_type == 'multimodal':
        dino_attention = extract_dino_attention(model, rgb)

    # *** Process each group using group_sizes ***
    start_idx = 0
    for group_idx, group_size in enumerate(group_sizes):
        end_idx = start_idx + group_size
        
        # Extract this group
        group_indices = range(start_idx, end_idx)
        group_size = len(group_indices)
        
        # Extract data for this group
        group_rgb = [rgb[i] for i in group_indices]
        group_rgb_geom = [rgb_geometric[i] for i in group_indices]
        group_scores = scores_np[list(group_indices)]
        group_labels = labels[list(group_indices)]
        group_difficulties = [difficulties[i] for i in group_indices]
        group_positions = [positions[i] for i in group_indices]
        group_vit_attn = [vit_attention[i] for i in group_indices]
        
        group_dino_attn = None
        if dino_attention is not None:
            group_dino_attn = [dino_attention[i] for i in group_indices]
        # Extract DINO features for this group if available
        group_dino_features = None
        if dino_features is not None:
            group_dino_features = [dino_features[i] for i in group_indices]

        # Rank by score (descending)
        rank_order = np.argsort(-group_scores)

        # Find positive
        pos_mask = group_labels == 1.0
        if pos_mask.sum() != 1:
            print(f"⚠️  Warning: Group {group_idx} has {pos_mask.sum()} positives")
            start_idx = end_idx
            continue
        
        # Check if positive is ranked first
        is_correct = (group_labels[rank_order[0]] == 1.0)
        
        # Get the best score and check if it exceeds threshold
        best_score = group_scores[rank_order[0]]
        is_accepted = best_score >= threshold
        
        # Get positive info
        positive_idx_in_group = np.where(group_labels == 1.0)[0][0]
        positive_position = group_positions[positive_idx_in_group]
        
        # Create visualization (4 rows now)
        num_rows = 7 # if dino_features is None else 5
        fig = plt.figure(figsize=(20, 7 * num_rows))
        gs = fig.add_gridspec(num_rows, group_size, hspace=0.3, wspace=0.2)
        
        for i, rank_idx in enumerate(rank_order):
            actual_idx = rank_idx
            
            # Row 0: RGB image
            ax_rgb = fig.add_subplot(gs[0, i])
            img_rgb = denormalize_image(group_rgb[actual_idx])
            ax_rgb.imshow(img_rgb)
            
            # Title
            score = group_scores[actual_idx]
            label = group_labels[actual_idx]
            diff = group_difficulties[actual_idx]
            orig_pos = group_positions[actual_idx]
            
            title_color = 'green' if label == 1.0 else 'red'
            ax_rgb.set_title(
                f"Rank {i+1} | Pos {orig_pos}\nScore: {score:.3f}\n{diff}",
                color=title_color,
                fontsize=10,
                fontweight='bold' if i == 0 else 'normal'
            )
            ax_rgb.axis('off')
            
            # Row 1: Contact region
            row_1_idx = 1
            ax_contact = fig.add_subplot(gs[row_1_idx, i])
            contact = group_rgb_geom[actual_idx][5].cpu().numpy()
            im_contact = ax_contact.imshow(contact, cmap='hot', vmin=0, vmax=1)
            ax_contact.set_title(f'Contact (max={contact.max():.2f})', fontsize=8)
            ax_contact.axis('off')  
            
            # Row 2: DINO features (only if multimodal)
            if model_type == 'multimodal' and group_dino_features is not None:
                ax_dino = fig.add_subplot(gs[2, i])
                dino_feat = group_dino_features[actual_idx]
                im_dino = ax_dino.imshow(dino_feat)
                ax_dino.set_title('DINO Features (PCA)', fontsize=8)
                ax_dino.axis('off')

            # Row 3: DINO attention (if multimodal) or ViT attention
            if dino_attention is not None:
                row_3_idx = 3
                ax_attn3 = fig.add_subplot(gs[row_3_idx, i])
                dino_attn_map = group_dino_attn[actual_idx].cpu().numpy()
                attn_overlay = overlay_attention_on_image(img_rgb, dino_attn_map, alpha=0.6)
                ax_attn3.imshow(attn_overlay)
                ax_attn3.set_title('DINO Attention', fontsize=8)

                # Row 4: Correlation between DINO attention and contact region
                row_4_idx = 4
                ax_comparison = fig.add_subplot(gs[row_4_idx, i])
                
                # Create side-by-side: contact region (left) + attention (right)
                # Normalize both to [0, 1] for comparison
                contact_normalized = (contact - contact.min()) / (contact.max() - contact.min() + 1e-8)
                dino_attn_normalized = (dino_attn_map - dino_attn_map.min()) / (dino_attn_map.max() - dino_attn_map.min() + 1e-8)
                
                # Resize to match for comparison
                from skimage.transform import resize
                dino_attn_resized = resize(dino_attn_normalized, contact.shape, order=1)
                
                # Compute correlation
                correlation = np.corrcoef(contact_normalized.flatten(), dino_attn_resized.flatten())[0, 1]
                
                # Show both
                combined = np.hstack([contact_normalized, dino_attn_resized])
                ax_comparison.imshow(combined, cmap='hot', vmin=0, vmax=1)
                ax_comparison.set_title(f'Contact | DINO Attention \n(corr={correlation:.2f})', fontsize=8)
                ax_comparison.axis('off')
            
            
            # Row 3: ViT attention (geometric branch)
            row_5_idx = 5
            ax_vit_attn = fig.add_subplot(gs[row_5_idx, i])
            vit_attn_map = group_vit_attn[actual_idx].cpu().numpy()
            
            # Overlay ViT attention on image
            vit_attn_overlay = overlay_attention_on_image(img_rgb, vit_attn_map, alpha=0.6, colormap=cv2.COLORMAP_VIRIDIS)
            ax_vit_attn.imshow(vit_attn_overlay)
            ax_vit_attn.set_title('Geometric ViT Attention', fontsize=8)
            ax_vit_attn.axis('off')
            
            # Optional Row 4: Side-by-side comparison (contact region + ViT attention)
            if vit_attn_map is not None:
                row_6_idx = 6
                ax_comparison = fig.add_subplot(gs[row_6_idx, i])
                
                # Create side-by-side: contact region (left) + attention (right)
                # Normalize both to [0, 1] for comparison
                contact_normalized = (contact - contact.min()) / (contact.max() - contact.min() + 1e-8)
                vit_attn_normalized = (vit_attn_map - vit_attn_map.min()) / (vit_attn_map.max() - vit_attn_map.min() + 1e-8)
                
                # Resize to match for comparison
                from skimage.transform import resize
                vit_attn_resized = resize(vit_attn_normalized, contact.shape, order=1)
                
                # Compute correlation
                correlation = np.corrcoef(contact_normalized.flatten(), vit_attn_resized.flatten())[0, 1]
                
                # Show both
                combined = np.hstack([contact_normalized, vit_attn_resized])
                ax_comparison.imshow(combined, cmap='hot', vmin=0, vmax=1)
                ax_comparison.set_title(f'Contact | ViT Attention \n(corr={correlation:.2f})', fontsize=8)
                ax_comparison.axis('off')
        
        # # Add colorbars
        # fig.colorbar(im_contact, ax=fig.get_axes()[group_size*2:group_size*3], 
        #             orientation='horizontal', pad=0.05, fraction=0.05)
        
        # Overall title
        correct_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        correct_color = 'green' if is_correct else 'red'
            
        # Overall title with shuffling info and threshold status
        correct_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        correct_color = 'green' if is_correct else 'red'
        
        # Threshold acceptance status
        if is_accepted:
            accepted_text = f"✓ ACCEPTED (score: {best_score:.2f} >= threshold: {threshold:.2f})"
            accepted_color = 'green'
        else:
            accepted_text = f"✗ REJECTED (best score: {best_score:.2f} < threshold: {threshold:.2f})"
            accepted_color = 'red'
            

        model_info = " [with DINO features]" if model_type == 'multimodal' else ""
        fig.suptitle(
            f"Batch {batch_idx}, Group {group_idx+1} | {correct_text} | {accepted_text}{model_info}\n"
            f"Positive Score: {group_scores[group_labels == 1.0][0]:.3f} | "
            f"Positive at position {positive_position} | Model: {model_type}",
            fontsize=14,
            fontweight='bold',
            color=correct_color
        )

        # Save
        save_path = save_dir / f'batch{batch_idx:03d}_group{group_idx:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path} | Positive at position {positive_position}")


def compare_attention_across_models(models_dict, batch, device, save_dir, batch_idx=0):
    """
    Compare attention maps across different models (baseline, geometric, multimodal).
    
    Args:
        models_dict: Dict like {'baseline': model1, 'geometric': model2, 'multimodal': model3}
        batch: Data batch
        device: Device
        save_dir: Save directory
        batch_idx: Batch index
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    rgb = batch['rgb'].to(device)
    rgb_geometric = batch['rgb_geometric'].to(device)
    labels = batch['labels'].cpu().numpy()
    
    # Find first positive (for comparison)
    pos_idx = np.where(labels == 1.0)[0][0]
    
    # Extract attention from each model
    attention_maps = {}
    for model_name, model in models_dict.items():
        model.eval()
        attn = extract_attention_maps(model, rgb, rgb_geometric, model_type=model_name)
        attention_maps[model_name] = attn[pos_idx].cpu().numpy()
    
    # Get contact region
    contact = rgb_geometric[pos_idx, 5].cpu().numpy()
    
    # Get RGB image
    img_rgb = denormalize_image(rgb[pos_idx])
    
    # Create comparison figure
    num_models = len(models_dict)
    fig, axes = plt.subplots(2, num_models + 1, figsize=(5 * (num_models + 1), 10))
    
    # First column: RGB and contact
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('RGB Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(contact, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Contact Region\n(max={contact.max():.2f})', fontsize=12)
    axes[1, 0].axis('off')
    
    # Other columns: attention from each model
    for i, (model_name, attn_map) in enumerate(attention_maps.items(), start=1):
        # Row 0: Attention overlay on image
        attn_overlay = overlay_attention_on_image(img_rgb, attn_map, alpha=0.6)
        axes[0, i].imshow(attn_overlay)
        axes[0, i].set_title(f'{model_name.capitalize()}\nAttention', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Row 1: Pure attention heatmap
        axes[1, i].imshow(attn_map, cmap='viridis')
        
        # Compute correlation with contact region
        from skimage.transform import resize
        attn_resized = resize(attn_map, contact.shape, order=1)
        contact_normalized = (contact - contact.min()) / (contact.max() - contact.min() + 1e-8)
        attn_normalized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        correlation = np.corrcoef(contact_normalized.flatten(), attn_normalized.flatten())[0, 1]
        
        axes[1, i].set_title(f'Attention Map\n(corr w/ contact: {correlation:.2f})', fontsize=12)
        axes[1, i].axis('off')
    
    plt.suptitle(f'Attention Comparison Across Models - Batch {batch_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f'attention_comparison_batch{batch_idx:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention comparison to {save_path}")

def visualize_score_distribution(model, dataloader, device, save_dir, model_type='geometric', max_batches=10):
    """
    Visualize score distributions across multiple batches.

    Args:
        model: Trained model
        dataloader: DataLoader
        device: 'cuda' or 'cpu'
        save_dir: Directory to save plots
        model_type: Type of model
        max_batches: Maximum number of batches to process
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    all_pos_scores = []
    all_neg_scores = []
    all_hard_neg_scores = []

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            rgb = batch['rgb'].to(device)
            rgb_geometric = batch['rgb_geometric'].to(device)
            labels = batch['labels'].cpu().numpy()
            difficulties = batch['difficulties']

            # Get predictions
            if model_type == 'multimodal':
                scores = model(rgb, rgb_geometric).squeeze()
            elif model_type == 'geometric':
                scores = model(rgb_geometric).squeeze()
            else:
                scores = model(rgb).squeeze()

            scores_np = scores.cpu().numpy()

            # Separate by type
            for score, label, diff in zip(scores_np, labels, difficulties):
                if label == 1.0:
                    all_pos_scores.append(score)
                elif diff == 'hard_negative':
                    all_hard_neg_scores.append(score)
                else:
                    all_neg_scores.append(score)

    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(all_pos_scores, bins=30, alpha=0.7, label='Positive', color='green', density=True)
    ax.hist(all_hard_neg_scores, bins=30, alpha=0.7, label='Hard Negative', color='orange', density=True)
    ax.hist(all_neg_scores, bins=30, alpha=0.7, label='Negative', color='red', density=True)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1]
    data = [all_pos_scores, all_hard_neg_scores, all_neg_scores]
    labels_box = ['Positive', 'Hard Negative', 'Negative']
    colors = ['green', 'orange', 'red']

    bp = ax.boxplot(data, labels=labels_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Score')
    ax.set_title('Score Distribution (Box Plot)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    stats_text = f"""
    Positive:      μ={np.mean(all_pos_scores):.3f}, σ={np.std(all_pos_scores):.3f}
    Hard Negative: μ={np.mean(all_hard_neg_scores):.3f}, σ={np.std(all_hard_neg_scores):.3f}
    Negative:      μ={np.mean(all_neg_scores):.3f}, σ={np.std(all_neg_scores):.3f}

    Separation (Pos - HardNeg): {np.mean(all_pos_scores) - np.mean(all_hard_neg_scores):.3f}
    """

    fig.text(0.5, -0.25, stats_text, ha='center', fontsize=10, family='monospace')

    # plt.tight_layout()
    save_path = save_dir / 'score_distributions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved score distribution to {save_path}")
    print(stats_text)


def analyze_failures(model, dataloader, device, save_dir, model_type='geometric', max_failures=20, threshold=0.5):
    """
    Find and visualize failure cases (where positive is not ranked first).

    Args:
        model: Trained model
        dataloader: DataLoader
        device: 'cuda' or 'cpu'
        save_dir: Directory to save visualizations
        model_type: Type of model
        max_failures: Maximum number of failures to visualize
        threshold: Threshold for determining if inference was successful
    """
    save_dir = Path(save_dir)
    failure_dir = save_dir / 'failures'
    failure_dir.mkdir(exist_ok=True, parents=True)

    model.eval()
    failures_found = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if failures_found >= max_failures:
                break

            rgb = batch['rgb'].to(device)
            rgb_geometric = batch['rgb_geometric'].to(device)
            labels = batch['labels'].cpu().numpy()
            difficulties = batch['difficulties']

            # Get predictions
            if model_type == 'multimodal':
                scores = model(rgb, rgb_geometric).squeeze()
            elif model_type == 'geometric':
                scores = model(rgb_geometric).squeeze()
            else:
                scores = model(rgb).squeeze()

            scores_np = scores.cpu().numpy()

            # Find groups
            positive_indices = np.where(labels == 1.0)[0]

            for group_idx, pos_idx in enumerate(positive_indices):
                if failures_found >= max_failures:
                    break

                # Find group boundaries
                if group_idx < len(positive_indices) - 1:
                    next_pos_idx = positive_indices[group_idx + 1]
                else:
                    next_pos_idx = len(labels)

                group_indices = range(pos_idx, next_pos_idx)
                group_scores = scores_np[list(group_indices)]
                group_labels = labels[list(group_indices)]

                # Check if this is a failure
                rank_order = np.argsort(-group_scores)
                is_failure = (group_labels[rank_order[0]] != 1.0)

                if is_failure:
                    # Visualize this failure
                    visualize_batch(
                        model,
                        batch,
                        device,
                        failure_dir,
                        batch_idx=failures_found,
                        model_type=model_type,
                        threshold=threshold
                    )
                    failures_found += 1

    print(f"Found and visualized {failures_found} failure cases in {failure_dir}")


def main(args):

    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir) / Path(args.model).stem
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir}")

        # Load multiple models for comparison
    if args.compare_models:
        models_dict = {}
        
        # Load baseline
        if args.baseline_checkpoint:
            models_dict['baseline'] = load_model(args.baseline_checkpoint, 'baseline', device)
        
        # Load geometric
        if args.geometric_checkpoint:
            models_dict['geometric'] = load_model(args.geometric_checkpoint, 'geometric', device)
        
        # Load multimodal
        if args.multimodal_checkpoint:
            models_dict['multimodal'] = load_model(args.multimodal_checkpoint, 'multimodal', device)
        
        # Compare attention maps
        for viz_idx, pair_idx in enumerate(random_indices[:5]):
            sample = dataset[pair_idx]
            
            batch = {
                'rgb': sample['rgb'].unsqueeze(0).to(device),
                'rgb_geometric': sample['rgb_geometric'].unsqueeze(0).to(device),
                'labels': sample['labels'].unsqueeze(0).to(device),
            }
            
            compare_attention_across_models(
                models_dict, 
                batch, 
                device, 
                output_dir / 'comparisons',
                batch_idx=viz_idx
            )

    # Load model
    model = load_model(args.model, args.model_type, device)

     # DATASET PARAMETERS
    RADIUS = 50
    THRESHOLD = 50

    # Load dataset (use validation set for visualization)
    full_dataset = PrecomposedAlignmentDataset(
        data_root=args.data_root,
        max_negatives_per_positive=12,
        radius=RADIUS,
        threshold=THRESHOLD,
    )

    # Create dataloader
    dataloader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for consistent visualization
        collate_fn=collate_alignment_samples,
        num_workers=0
    )

    print(f"Dataset: {len(full_dataset)} samples")

    # Visualize batches
    print(f"\nVisualizing {args.num_batches} batches...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        visualize_batch(
            model,
            batch,
            device,
            output_dir,
            batch_idx=batch_idx,
            model_type=args.model_type,
            threshold=args.threshold
        )

    # Score distribution
    print("\nAnalyzing score distributions...")
    visualize_score_distribution(
        model,
        dataloader,
        device,
        output_dir,
        model_type=args.model_type,
        max_batches=20
    )

    # Failure analysis
    if args.analyze_failures:
        print("\nAnalyzing failure cases...")
        analyze_failures(
            model,
            dataloader,
            device,
            output_dir,
            model_type=args.model_type,
            max_failures=20,
            threshold=args.threshold
        )

    print(f"\nDone! Visualizations saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['baseline', 'geometric', 'multimodal'],
                       help='Type of model')
    parser.add_argument('--data_root', type=str, default='./data', help='Path to data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_batches', type=int, default=5, help='Number of batches to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--analyze_failures', action='store_true', help='Find and visualize failures')
    parser.add_argument('--compare_models', action='store_true', help='Compare models')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for determining if inference was successful (default: 0.5)')

    args = parser.parse_args()
    main(args)

"""
python visualize_model.py \
    --compare_models \
    --baseline_checkpoint checkpoints/baseline_best.pth \
    --geometric_checkpoint checkpoints/geometric_best.pth \
    --multimodal_checkpoint checkpoints/multimodal_best.pth \
    --data_root ./data \
    --output_dir ./visualizations/comparison
"""
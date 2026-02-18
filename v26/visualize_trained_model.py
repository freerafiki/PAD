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

from dataset_v3 import PrecomposedAlignmentDataset, collate_alignment_samples
from models import BaselineScorer, GeometricScorer, MultiModalScorer
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
        model = MultiModalScorer()
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


def extract_attention_maps(model, rgb, rgb_geometric):
    """
    Extract attention maps from ViT model.

    This is a simplified version - actual implementation depends on model architecture.
    For ViT, we can hook into the attention layers.

    Args:
        model: The trained model
        rgb: (B, 3, H, W) RGB input
        rgb_geometric: (B, 6, H, W) RGB+geometric input

    Returns:
        attention_maps: List of attention maps from different layers
    """
    # This is a placeholder - actual implementation requires registering hooks
    # For now, we'll return None and skip attention visualization
    # You can implement this later if needed

    return None

def visualize_batch(model, batch, device, save_dir, batch_idx=0, model_type='geometric'):
    """
    Visualize predictions for one batch.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    rgb = batch['rgb'].to(device)
    rgb_geometric = batch['rgb_geometric'].to(device)
    labels = batch['labels'].cpu().numpy()
    difficulties = batch['difficulties']
    positions = batch['positions']  # *** NEW ***

    # Get predictions
    with torch.no_grad():
        if model_type == 'multimodal':
            scores = model(rgb, rgb_geometric).squeeze()
        elif model_type == 'geometric':
            scores = model(rgb_geometric).squeeze()
        else:
            scores = model(rgb).squeeze()

    scores_np = scores.cpu().numpy()

    # Identify groups
    positive_indices = np.where(labels == 1.0)[0]

    for group_idx, pos_idx in enumerate(positive_indices):
        # Find group boundaries
        if group_idx < len(positive_indices) - 1:
            next_pos_idx = positive_indices[group_idx + 1]
        else:
            next_pos_idx = len(labels)

        # Extract this group
        group_indices = range(pos_idx, next_pos_idx)
        group_size = len(group_indices)

        # Extract data for this group
        group_rgb = [rgb[i] for i in group_indices]
        group_rgb_geom = [rgb_geometric[i] for i in group_indices]
        group_scores = scores_np[list(group_indices)]
        group_labels = labels[list(group_indices)]
        group_difficulties = [difficulties[i] for i in group_indices]
        group_positions = [positions[i] for i in group_indices]  # *** NEW ***

        # Rank by score (descending)
        rank_order = np.argsort(-group_scores)

        # Check if positive is ranked first
        is_correct = (group_labels[rank_order[0]] == 1.0)

        # *** CORRECT CHECK: Where is the positive in THIS group? ***
        positive_idx_in_group = np.where(group_labels == 1.0)[0][0]  # Index within this group
        positive_original_position = group_positions[positive_idx_in_group]  # Where it came from (always 0)

        # What we ACTUALLY want to check: is the positive at different indices?
        # The index itself tells us if shuffling worked
        shuffling_worked = True  # We can see from positive_idx_in_group varying

        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, group_size, hspace=0.3, wspace=0.2)

        for i, rank_idx in enumerate(rank_order):
            actual_idx = rank_idx

            # Row 0: RGB image
            ax_rgb = fig.add_subplot(gs[0, i])
            img_rgb = denormalize_image(group_rgb[actual_idx])
            ax_rgb.imshow(img_rgb)

            # Title with position info
            score = group_scores[actual_idx]
            label = group_labels[actual_idx]
            diff = group_difficulties[actual_idx]
            orig_pos = group_positions[actual_idx]  # *** NEW ***

            title_color = 'green' if label == 1.0 else 'red'
            ax_rgb.set_title(
                f"Rank {i+1} | Pos {orig_pos}\nScore: {score:.3f}\n{diff}",  # *** ADDED POSITION ***
                color=title_color,
                fontsize=10,
                fontweight='bold' if i == 0 else 'normal'
            )
            ax_rgb.axis('off')

            # Row 3: Contact region (channel 5)
            ax_contact = fig.add_subplot(gs[1, i])
            contact = group_rgb_geom[actual_idx][5].cpu().numpy()
            im_contact = ax_contact.imshow(contact, cmap='hot', vmin=0, vmax=1)
            ax_contact.set_title(f'Contact (max={contact.max():.2f})', fontsize=8)
            ax_contact.axis('off')

        # Add colorbars
        fig.colorbar(im_contact, ax=ax_contact,
                    orientation='horizontal', pad=0.05, fraction=0.05)

        # Overall title with shuffling info
        correct_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        correct_color = 'green' if is_correct else 'red'
        shuffle_text = "✓ Shuffled" if shuffling_worked else "⚠ NOT shuffled (pos always at 0)"  # *** NEW ***

        # Update title to show the INDEX, not the original position
        fig.suptitle(
            f"Batch {batch_idx}, Group {group_idx+1} | {correct_text}\n"
            f"Positive is at index {positive_idx_in_group}/{group_size-1} in this group "
            f"(came from pre-shuffle position {positive_original_position})",
            fontsize=14,
            fontweight='bold',
            color=correct_color
        )

        # ... [save] ...

        print(f"  Group {group_idx}: "
            f"difficulties: {batch['difficulties']} "
            f"Positive at index {positive_idx_in_group}/{group_size-1} "
            f"(original position: {positive_original_position})")

        # Save
        save_path = save_dir / f'batch{batch_idx:03d}_group{group_idx:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization to {save_path} | Positive originally at position {positive_original_position} and now at {positive_idx_in_group}")

def visualize_batch_v1(model, batch, device, save_dir, batch_idx=0, model_type='geometric'):
    """
    Visualize predictions for one batch.

    Args:
        model: Trained model
        batch: Batch from dataloader
        device: 'cuda' or 'cpu'
        save_dir: Directory to save visualizations
        batch_idx: Batch index for naming files
        model_type: Type of model for inference
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    rgb = batch['rgb'].to(device)
    rgb_geometric = batch['rgb_geometric'].to(device)
    labels = batch['labels'].cpu().numpy()
    difficulties = batch['difficulties']

    # Get predictions
    with torch.no_grad():
        if model_type == 'multimodal':
            scores = model(rgb, rgb_geometric).squeeze()
        elif model_type == 'geometric':
            scores = model(rgb_geometric).squeeze()
        else:
            scores = model(rgb).squeeze()

    scores_np = scores.cpu().numpy()

    # Identify groups (each group starts with a positive, potentially)
    # We'll visualize each group separately

    # Simple grouping: find positives
    positive_indices = np.where(labels == 1.0)[0]

    for group_idx, pos_idx in enumerate(positive_indices):
        # Find group boundaries
        if group_idx < len(positive_indices) - 1:
            next_pos_idx = positive_indices[group_idx + 1]
        else:
            next_pos_idx = len(labels)

        # Extract this group
        group_indices = range(pos_idx, next_pos_idx)
        group_size = len(group_indices)

        # Extract data for this group
        group_rgb = [rgb[i] for i in group_indices]
        group_rgb_geom = [rgb_geometric[i] for i in group_indices]
        group_scores = scores_np[list(group_indices)]
        group_labels = labels[list(group_indices)]
        group_difficulties = [difficulties[i] for i in group_indices]

        # Rank by score (descending)
        rank_order = np.argsort(-group_scores)  # Negative for descending

        # Check if positive is ranked first
        is_correct = (group_labels[rank_order[0]] == 1.0)

        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, group_size, hspace=0.3, wspace=0.2)

        for i, rank_idx in enumerate(rank_order):
            actual_idx = rank_idx

            # Row 0: RGB image
            ax_rgb = fig.add_subplot(gs[0, i])
            img_rgb = denormalize_image(group_rgb[actual_idx])
            ax_rgb.imshow(img_rgb)

            # Title: rank, score, ground truth
            score = group_scores[actual_idx]
            label = group_labels[actual_idx]
            diff = group_difficulties[actual_idx]

            title_color = 'green' if label == 1.0 else 'red'
            ax_rgb.set_title(
                f"Rank {i+1}\nScore: {score:.3f}\n{diff}",
                color=title_color,
                fontsize=10,
                fontweight='bold' if i == 0 else 'normal'
            )
            ax_rgb.axis('off')

            # Row 1: Proximity to piece A (channel 3)
            ax_prox_a = fig.add_subplot(gs[1, i])
            prox_a = group_rgb_geom[actual_idx][3].cpu().numpy()
            im_a = ax_prox_a.imshow(prox_a, cmap='Reds', vmin=0, vmax=1)
            ax_prox_a.set_title('Proximity A', fontsize=8)
            ax_prox_a.axis('off')

            # Row 2: Proximity to piece B (channel 4)
            ax_prox_b = fig.add_subplot(gs[2, i])
            prox_b = group_rgb_geom[actual_idx][4].cpu().numpy()
            im_b = ax_prox_b.imshow(prox_b, cmap='Blues', vmin=0, vmax=1)
            ax_prox_b.set_title('Proximity B', fontsize=8)
            ax_prox_b.axis('off')

            # Row 3: Contact region (channel 5)
            ax_contact = fig.add_subplot(gs[3, i])
            contact = group_rgb_geom[actual_idx][5].cpu().numpy()
            im_contact = ax_contact.imshow(contact, cmap='hot', vmin=0, vmax=1)
            ax_contact.set_title(f'Contact (max={contact.max():.2f})', fontsize=8)
            ax_contact.axis('off')

        # Add colorbars
        fig.colorbar(im_a, ax=fig.get_axes()[group_size:2*group_size],
                    orientation='horizontal', pad=0.05, fraction=0.05)
        fig.colorbar(im_b, ax=fig.get_axes()[2*group_size:3*group_size],
                    orientation='horizontal', pad=0.05, fraction=0.05)
        fig.colorbar(im_contact, ax=fig.get_axes()[3*group_size:4*group_size],
                    orientation='horizontal', pad=0.05, fraction=0.05)

        # Overall title
        correct_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        correct_color = 'green' if is_correct else 'red'

        fig.suptitle(
            f"Batch {batch_idx}, Group {group_idx+1} | {correct_text} | "
            f"Positive Score: {group_scores[group_labels == 1.0][0]:.3f}",
            fontsize=14,
            fontweight='bold',
            color=correct_color
        )

        # Save
        save_path = save_dir / f'batch{batch_idx:03d}_group{group_idx:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization to {save_path}")


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


def analyze_failures(model, dataloader, device, save_dir, model_type='geometric', max_failures=20):
    """
    Find and visualize failure cases (where positive is not ranked first).

    Args:
        model: Trained model
        dataloader: DataLoader
        device: 'cuda' or 'cpu'
        save_dir: Directory to save visualizations
        model_type: Type of model
        max_failures: Maximum number of failures to visualize
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
                        model_type=model_type
                    )
                    failures_found += 1

    print(f"Found and visualized {failures_found} failure cases in {failure_dir}")


def main(args):

    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir) / Path(args.model).stem
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir}")

    # Load model
    model = load_model(args.model, args.model_type, device)

    # Load dataset (use validation set for visualization)
    full_dataset = PrecomposedAlignmentDataset(
        data_root=args.data_root,
        negatives_per_positive=6
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
            model_type=args.model_type
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
            max_failures=20
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

    args = parser.parse_args()
    main(args)

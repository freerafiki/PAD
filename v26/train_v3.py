import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your dataset and models
from dataset_v3 import PrecomposedAlignmentDataset, collate_alignment_samples, ShuffledBatchSampler
from models import BaselineScorer, GeometricScorer, MultiModalScorer
from loss_v2 import RankingLoss

def evaluate_ranking(model, dataloader, device):
    """
    Evaluate model on ranking task.

    For each group of samples (1 positive + K negatives),
    check if the positive is ranked highest.

    Returns:
        accuracy: Fraction of groups where positive is ranked first
        avg_positive_score: Average score given to positive samples
        avg_negative_score: Average score given to negative samples
    """
    model.eval()

    correct = 0
    total_groups = 0
    all_pos_scores = []
    all_neg_scores = []

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device)
            rgb_geometric = batch['rgb_geometric'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (adapt based on model type)
            if hasattr(model, 'dino'):  # MultiModalScorer
                scores = model(rgb, rgb_geometric).squeeze()
            elif hasattr(model, 'projection'):
                scores = model(rgb_geometric).squeeze()
            else:
                scores = model(rgb).squeeze()

            # Group samples by their original puzzle
            # In our dataloader, each "batch" contains multiple puzzles
            # Each puzzle has 1 positive + N negatives

            # We need to identify groups
            # Assumption: positives come first in each group
            # Let's reconstruct groups based on label patterns

            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Find where positives are (start of each group)
            positive_indices = np.where(labels_np == 1.0)[0]

            for i, pos_idx in enumerate(positive_indices):
                # Find the next positive (or end of batch)
                if i < len(positive_indices) - 1:
                    next_pos_idx = positive_indices[i + 1]
                else:
                    next_pos_idx = len(scores_np)

                # This group: from pos_idx to next_pos_idx
                group_scores = scores_np[pos_idx:next_pos_idx]
                group_labels = labels_np[pos_idx:next_pos_idx]

                # Check if positive (first in group) has highest score
                if group_scores[0] == group_scores.max():
                    correct += 1

                total_groups += 1

                # Collect statistics
                all_pos_scores.append(group_scores[0])
                all_neg_scores.extend(group_scores[1:])

    accuracy = correct / total_groups if total_groups > 0 else 0.0
    avg_pos_score = np.mean(all_pos_scores) if all_pos_scores else 0.0
    avg_neg_score = np.mean(all_neg_scores) if all_neg_scores else 0.0

    return accuracy, avg_pos_score, avg_neg_score


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        rgb = batch['rgb'].to(device)
        rgb_geometric = batch['rgb_geometric'].to(device)
        labels = batch['labels'].to(device)
        difficulties = batch['difficulties']

        optimizer.zero_grad()

        # Forward pass (depends on model architecture)
        if hasattr(model, 'dino'):  # MultiModalScorer
            scores = model(rgb, rgb_geometric).squeeze()
        elif hasattr(model, 'projection'):
            scores = model(rgb_geometric).squeeze()
        else:
            scores = model(rgb).squeeze()

        # Compute loss
        loss = criterion(scores, labels, difficulties)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_model(
    model,
    train_dataset,
    val_dataset,
    num_epochs=10,
    batch_size=32,
    lr=1e-4,
    device='cuda',
    save_dir='checkpoints/wikiart',
    model_name='model'
):
    """
    Complete training loop.

    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Number of epochs
        batch_size: Batch size (number of puzzle groups per batch)
        lr: Learning rate
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
        model_name: Name for saving checkpoints
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=ShuffledBatchSampler(train_dataset, shuffle=True, seed=42),  # *** NEW ***
        collate_fn=collate_alignment_samples,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=ShuffledBatchSampler(val_dataset, shuffle=True, seed=42),  # *** NEW ***
        collate_fn=collate_alignment_samples,
        num_workers=4,
        pin_memory=True
    )

    # Loss and optimizer
    criterion = RankingLoss(margin=0.3, hard_negative_weight=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Move model to device
    model = model.to(device)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_pos_score': [],
        'val_neg_score': []
    }

    best_val_acc = 0.0

    print(f"Training {model_name} for {num_epochs} epochs")
    print(f"Train samples: {len(train_dataset)} groups")
    print(f"Val samples: {len(val_dataset)} groups")
    print(f"Device: {device}")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                rgb_geometric = batch['rgb_geometric'].to(device)
                labels = batch['labels'].to(device)
                difficulties = batch['difficulties']

                if hasattr(model, 'dino'):
                    scores = model(rgb, rgb_geometric).squeeze()
                elif hasattr(model, 'projection'):
                    scores = model(rgb_geometric).squeeze()
                else:
                    scores = model(rgb).squeeze()

                loss = criterion(scores, labels, difficulties)
                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0

        # Evaluate ranking accuracy
        val_acc, avg_pos, avg_neg = evaluate_ranking(model, val_loader, device)

        # Learning rate step
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_pos_score'].append(avg_pos)
        history['val_neg_score'].append(avg_neg)

        # Print epoch summary
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.3f} | "
              f"Pos/Neg: {avg_pos:.3f}/{avg_neg:.3f}")

        # Save best model (BETTER WAY)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save model weights only (safe for weights_only=True loading)
            torch.save(
                model.state_dict(),
                save_dir / f'{model_name}_best_weights.pth'
            )

            # Save full checkpoint with metadata separately (weights_only=False needed)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': float(val_acc),  # Convert to Python float, not numpy
                'train_loss': float(train_loss),
                'val_loss': float(avg_val_loss),
                # Don't save full history with numpy arrays - save separately
            }
            torch.save(checkpoint, save_dir / f'{model_name}_best.pth')

            # Save history as separate JSON file (cleaner)
            import json
            history_serializable = {
                k: [float(x) for x in v]  # Convert numpy to Python floats
                for k, v in history.items()
            }
            with open(save_dir / f'{model_name}_history.json', 'w') as f:
                json.dump(history_serializable, f, indent=2)

            print(f"  → Saved best model (acc: {val_acc:.3f})")

        # Similar for periodic checkpoints
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, save_dir / f'{model_name}_epoch{epoch}.pth')

    print("-" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.3f}")

    # Plot training curves
    plot_training_history(history, save_dir / f'{model_name}_history.png')

    return model, history


def plot_training_history(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(epochs, history['val_accuracy'], label='Val Accuracy', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy (Positive Ranked First)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Score distribution
    axes[1, 0].plot(epochs, history['val_pos_score'], label='Avg Positive Score')
    axes[1, 0].plot(epochs, history['val_neg_score'], label='Avg Negative Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Average Scores by Class')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Score separation (margin)
    margin = np.array(history['val_pos_score']) - np.array(history['val_neg_score'])
    axes[1, 1].plot(epochs, margin, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score Margin')
    axes[1, 1].set_title('Positive - Negative Score Margin')
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {save_path}")


def main():
    """Main training script."""

    # Configuration
    DATA_ROOT = '/media/lucap/big_data/datasets/wikiart_PAD/PAD_dataset__Wikiart'
    # DATA_ROOT = '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v4'
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_NEGATIVES_PER_POSITIVE = 12  # Adjusted for your data size

    # DATASET PARAMETERS
    RADIUS = 30
    THRESHOLD = 30

    # DINO PARAMETERS
    DINO_MODEL = 'facebook/dinov2-base'

    print(f"Using device: {DEVICE}")

    # Load full dataset
    full_dataset = PrecomposedAlignmentDataset(
        data_root=DATA_ROOT,
        max_negatives_per_positive=MAX_NEGATIVES_PER_POSITIVE,
        hard_negative_ratio=0.6,
        radius = RADIUS,
        threshold = THRESHOLD
    )

    # *** CHANGED: Split by puzzles, not random ***
    train_dataset, val_dataset = PrecomposedAlignmentDataset.create_puzzle_split(
        full_dataset,
        radius = RADIUS,
        threshold = THRESHOLD,
        train_ratio=0.8,
        seed=42
    )

    print(f"\n=== Dataset Ready ===")
    print(f"Train: {len(train_dataset)} pairs")
    print(f"Val: {len(val_dataset)} pairs")

    # Verify no overlap
    train_puzzles = set(k.split('|')[0] for k in train_dataset.pair_keys)
    val_puzzles = set(k.split('|')[0] for k in val_dataset.pair_keys)
    overlap = train_puzzles & val_puzzles

    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} puzzles appear in both train and val!")
    else:
        print("✓ No puzzle overlap between train and val")

    # Train Version 1: Baseline (RGB only)
    print("\n" + "="*60)
    print("TRAINING VERSION 1: Baseline (RGB only)")
    print("="*60)

    model_v1 = BaselineScorer()
    model_v1, history_v1 = train_model(
        model_v1,
        train_dataset,
        val_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        device=DEVICE,
        model_name='baseline'
    )

    # Train Version 2: + Geometry
    print("\n" + "="*60)
    print("TRAINING VERSION 2: RGB + Geometry")
    print("="*60)

    model_v2 = GeometricScorer()
    model_v2, history_v2 = train_model(
        model_v2,
        train_dataset,
        val_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        device=DEVICE,
        model_name='geometric'
    )

    # Train Version 3: + DINO
    print("\n" + "="*60)
    print("TRAINING VERSION 3: RGB + Geometry + DINO")
    print("="*60)

    model_v3 = MultiModalScorer(dino_model=DINO_MODEL)
    model_v3, history_v3 = train_model(
        model_v3,
        train_dataset,
        val_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        device=DEVICE,
        model_name='multimodal'
    )

    # Compare results
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"Baseline (RGB only):      {max(history_v1['val_accuracy']):.3f}")
    print(f"+ Geometry:               {max(history_v2['val_accuracy']):.3f}")
    print(f"+ DINO:                   {max(history_v3['val_accuracy']):.3f}")


if __name__ == '__main__':
    main()

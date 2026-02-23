from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import your dataset and models
from dataset_v3 import (
    PrecomposedAlignmentDataset,
    ShuffledBatchSampler,
    collate_alignment_samples,
)
from models import MultiModalScorerV2_Practical
from loss_v2 import AdaptiveTopNRankingLoss
from training_utils import plot_training_history
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm



def train_model(
    model,
    train_dataset,
    val_dataset,
    optimizer=None,
    num_epochs=50,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    save_dir="checkpoints",
    model_name="model",
    early_stopping_patience=10,  # NEW: Stop if no improvement
    bce_weight=0.2,  # Weight for BCE loss (lower for ranking-focused training)
    ranking_weight=0.8,  # Weight for Adaptive ranking loss (higher priority)
    ranking_margin=0.3,  # Margin for ranking loss
    hard_negative_weight=2.0,  # Weight for hard negatives in ranking loss
    top_n=3,  # Target top-N positions for AdaptiveTopNRankingLoss
    temperature=1.0,  # Temperature for smooth position penalty
):
    """
    Training with combined BCE + AdaptiveTopNRankingLoss and early stopping for small datasets.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=ShuffledBatchSampler(train_dataset, shuffle=True, seed=42),
        collate_fn=collate_alignment_samples,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=ShuffledBatchSampler(val_dataset, shuffle=False),
        collate_fn=collate_alignment_samples,
        num_workers=4,
        pin_memory=True,
    )

    # Loss functions
    bce_criterion = nn.BCEWithLogitsLoss()  # More stable than BCE + Sigmoid
    ranking_criterion = AdaptiveTopNRankingLoss(
        top_n=top_n,
        margin=ranking_margin,
        hard_negative_weight=hard_negative_weight,
        temperature=temperature,
    )
    
    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )

    model = model.to(device)

    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_bce_loss": [],
        "train_ranking_loss": [],
        "val_loss": [],
        "val_bce_loss": [],
        "val_ranking_loss": [],
        "val_accuracy": [],
        "val_top3_accuracy": [],
        "val_top5_accuracy": [],
        "val_pos_score": [],
        "val_neg_score": [],
        "val_hard_neg_score": [],
        "learning_rates": [],
    }

    print(f"Training {model_name} for up to {num_epochs} epochs")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Loss: BCE={bce_weight} + AdaptiveTopNRankingLoss={ranking_weight}")
    print(f"  AdaptiveTopNRankingLoss: top_n={top_n}, margin={ranking_margin}, temperature={temperature}")
    print(f"Train samples: {len(train_dataset)} pairs")
    print(f"Val samples: {len(val_dataset)} pairs")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_bce_loss = 0.0
        train_ranking_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch in pbar:
            rgb = batch["rgb"].to(device)
            rgb_geometric = batch["rgb_geometric"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)  # (B, 1)
            difficulties = batch["difficulties"]

            optimizer.zero_grad()

            # Forward
            logits = model(rgb, rgb_geometric)
            scores = torch.sigmoid(logits).squeeze()  # Convert to [0, 1] for ranking loss

            # BCE Loss
            bce_loss = bce_criterion(logits, labels)

            # Ranking Loss
            ranking_loss = ranking_criterion(scores, labels.squeeze(), difficulties)

            # Combined loss
            loss = bce_weight * bce_loss + ranking_weight * ranking_loss

            # Backward
            loss.backward()

            # Gradient clipping (helps with small data)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_bce_loss += bce_loss.item()
            train_ranking_loss += ranking_loss.item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "bce": f"{bce_loss.item():.4f}",
                    "rank": f"{ranking_loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                }
            )

        avg_train_loss = train_loss / num_batches
        avg_train_bce = train_bce_loss / num_batches
        avg_train_ranking = train_ranking_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_bce_loss = 0.0
        val_ranking_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device)
                rgb_geometric = batch["rgb_geometric"].to(device)
                labels = batch["labels"].to(device).unsqueeze(1)
                difficulties = batch["difficulties"]

                logits = model(rgb, rgb_geometric)
                scores = torch.sigmoid(logits).squeeze()

                bce_loss = bce_criterion(logits, labels)
                ranking_loss = ranking_criterion(scores, labels.squeeze(), difficulties)
                loss = bce_weight * bce_loss + ranking_weight * ranking_loss

                val_loss += loss.item()
                val_bce_loss += bce_loss.item()
                val_ranking_loss += ranking_loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches
        avg_val_bce = val_bce_loss / num_batches
        avg_val_ranking = val_ranking_loss / num_batches
        
        # Ranking accuracy (Top-1, Top-3, Top-5)
        val_acc, val_top3_acc, val_top5_acc, avg_pos, avg_neg, avg_hard_neg = evaluate_ranking(model, val_loader, device)

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["train_bce_loss"].append(avg_train_bce)
        history["train_ranking_loss"].append(avg_train_ranking)
        history["val_loss"].append(avg_val_loss)
        history["val_bce_loss"].append(avg_val_bce)
        history["val_ranking_loss"].append(avg_val_ranking)
        history["val_accuracy"].append(val_acc)
        history["val_top3_accuracy"].append(val_top3_acc)
        history["val_top5_accuracy"].append(val_top5_acc)
        history["val_pos_score"].append(avg_pos)
        history["val_neg_score"].append(avg_neg)
        history["val_hard_neg_score"].append(avg_hard_neg)
        history["learning_rates"].append(scheduler.get_last_lr()[0])


        # Ranking accuracy
        val_acc, avg_pos, avg_neg, avg_hard_neg = evaluate_ranking(model, val_loader, device)

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["train_bce_loss"].append(avg_train_bce)
        history["train_ranking_loss"].append(avg_train_ranking)
        history["val_loss"].append(avg_val_loss)
        history["val_bce_loss"].append(avg_val_bce)
        history["val_ranking_loss"].append(avg_val_ranking)
        history["val_accuracy"].append(val_acc)
        history["val_pos_score"].append(avg_pos)
        history["val_neg_score"].append(avg_neg)
        history["val_hard_neg_score"].append(avg_hard_neg)
        history["learning_rates"].append(scheduler.get_last_lr()[0])

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce:.4f}, Rank: {avg_train_ranking:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"Pos/Neg/Hard: {avg_pos:.3f}/{avg_neg:.3f}/{avg_hard_neg:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "history": history,
            }
            torch.save(checkpoint, save_dir / f"{model_name}_best.pth")
            print(f"  → Saved best model (acc: {val_acc:.3f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.3f}")
            break

    print("-" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.3f}")

    # Plot training curves
    plot_training_history(history, save_dir / f"{model_name}_history.png")

    return model, history


def evaluate_ranking(model, dataloader, device):
    """
    Evaluate ranking accuracy: is positive ranked first in each group?
    Also returns average scores for positives, negatives, and hard negatives.
    """
    model.eval()

    correct = 0
    total_groups = 0
    all_pos_scores = []
    all_neg_scores = []
    all_hard_neg_scores = []

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            rgb_geometric = batch["rgb_geometric"].to(device)
            labels = batch["labels"].to(device)
            difficulties = batch["difficulties"]

            # Get logits and convert to probabilities
            logits = model(rgb, rgb_geometric).squeeze()
            scores = torch.sigmoid(logits)  # Convert to [0, 1]

            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Find groups
            positive_indices = np.where(labels_np == 1.0)[0]

            for i, pos_idx in enumerate(positive_indices):
                if i < len(positive_indices) - 1:
                    next_pos_idx = positive_indices[i + 1]
                else:
                    next_pos_idx = len(scores_np)

                group_scores = scores_np[pos_idx:next_pos_idx]
                group_labels = labels_np[pos_idx:next_pos_idx]
                group_difficulties = difficulties[pos_idx:next_pos_idx]

                # Is positive ranked first?
                if group_scores[0] == group_scores.max():
                    correct += 1

                total_groups += 1

                # Collect scores by type
                for score, label, diff in zip(group_scores, group_labels, group_difficulties):
                    if label == 1.0:
                        all_pos_scores.append(score)
                    elif diff == 'hard_negative':
                        all_hard_neg_scores.append(score)
                    else:
                        all_neg_scores.append(score)

    accuracy = correct / total_groups if total_groups > 0 else 0.0
    avg_pos_score = np.mean(all_pos_scores) if all_pos_scores else 0.0
    avg_neg_score = np.mean(all_neg_scores) if all_neg_scores else 0.0
    avg_hard_neg_score = np.mean(all_hard_neg_scores) if all_hard_neg_scores else 0.0

    return accuracy, avg_pos_score, avg_neg_score, avg_hard_neg_score


def diagnose_data_sufficiency(history):
    """
    Analyze learning curves to diagnose data issues.
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    print("\n=== Data Sufficiency Diagnosis ===")

    # Check 1: Overfitting
    final_gap = val_loss[-1] - train_loss[-1]
    if final_gap > 0.2:
        print("❌ SEVERE OVERFITTING: Val loss >> Train loss")
        print("   → Need MORE DATA or MORE REGULARIZATION")
    elif final_gap > 0.1:
        print("⚠️  Moderate overfitting")
        print("   → Could benefit from more data")
    else:
        print("✓ No major overfitting")

    # Check 2: Convergence
    if val_loss[-1] < val_loss[2]:
        print("✓ Model is learning (val loss decreasing)")
    else:
        print("❌ Val loss not improving")
        print("   → Model may be too complex for data size")

    # Check 3: Early stopping
    best_epoch = np.argmin(val_loss) + 1
    total_epochs = len(val_loss)

    if best_epoch < total_epochs * 0.3:
        print(f"❌ Best epoch: {best_epoch}/{total_epochs} (very early)")
        print("   → DEFINITELY need more data")
    elif best_epoch < total_epochs * 0.6:
        print(f"⚠️  Best epoch: {best_epoch}/{total_epochs}")
        print("   → Data size is marginal")
    else:
        print(f"✓ Best epoch: {best_epoch}/{total_epochs}")
        print("   → Data size seems adequate")


def estimate_data_needs(model, train_dataset):
    """
    Estimate data needs accounting for trainable, frozen, and pre-trained parameters.
    
    Key insight: Pre-trained models need much less data than training from scratch.
    - Frozen parameters: No data needed (not updated)
    - Fine-tuned parameters: ~1-5 samples per parameter (already have good representations)
    - Trainable from scratch (heads/classifiers): ~10-20 samples per parameter
    """
    # Count different parameter types
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        if param.requires_grad:
            trainable_params += param_count
        else:
            frozen_params += param_count
    
    total_params = trainable_params + frozen_params
    
    # Identify which parameters are from pre-trained backbones vs new layers
    # Pre-trained backbones typically have names like 'vit', 'dino', 'geometric_vit'
    pretrain_finetune_params = 0
    new_layer_params = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_count = param.numel()
        # Check if this is part of a pre-trained backbone being fine-tuned
        if any(backbone in name for backbone in ['vit', 'dino', 'geometric_vit', 'backbone', 'encoder']):
            pretrain_finetune_params += param_count
        else:
            # New layers (projection, fusion, scorer heads, etc.)
            new_layer_params += param_count
    
    print(f"\n=== Data Needs Estimate ===")
    print(f"Total parameters: {total_params:,}")
    print(f"  ├── Trainable: {trainable_params:,}")
    print(f"  │   ├── Fine-tuned (pre-trained backbones): {pretrain_finetune_params:,}")
    print(f"  │   └── New layers (from scratch): {new_layer_params:,}")
    print(f"  └── Frozen: {frozen_params:,}")
    
    # Calculate data needs with different requirements for each parameter type
    # Fine-tuned parameters: 1-5 samples per parameter (already have good representations)
    # New layers: 10-20 samples per parameter (learning from scratch)
    
    # Conservative estimate
    min_finetune_samples = pretrain_finetune_params * 1  # Minimum for fine-tuning
    min_newlayer_samples = new_layer_params * 10  # Minimum for new layers
    min_samples = min_finetune_samples + min_newlayer_samples
    
    # Recommended estimate
    rec_finetune_samples = pretrain_finetune_params * 5  # Recommended for fine-tuning
    rec_newlayer_samples = new_layer_params * 20  # Recommended for new layers
    recommended_samples = rec_finetune_samples + rec_newlayer_samples
    
    print(f"\nData requirements:")
    print(f"  Minimum (conservative):")
    print(f"    ├── Fine-tuned params: {pretrain_finetune_params:,} × 1 = {min_finetune_samples:,} samples")
    print(f"    └── New layers: {new_layer_params:,} × 10 = {min_newlayer_samples:,} samples")
    print(f"    └── Total minimum: {min_samples:,} samples")
    print(f"  Recommended:")
    print(f"    ├── Fine-tuned params: {pretrain_finetune_params:,} × 5 = {rec_finetune_samples:,} samples")
    print(f"    └── New layers: {new_layer_params:,} × 20 = {rec_newlayer_samples:,} samples")
    print(f"    └── Total recommended: {recommended_samples:,} samples")
    
    # Current data estimate
    # Each pair has 1 positive + N negatives, so total samples = pairs * (1 + negatives_per_positive)
    # This is approximate since negatives_per_positive can vary
    current_samples = len(train_dataset)  # Each item in dataset is one sample
    
    print(f"\nCurrent training data: {current_samples:,} samples")
    
    # Assessment
    if current_samples < min_samples:
        deficit = min_samples - current_samples
        print(f"❌ INSUFFICIENT data (need at least {deficit:,} more samples)")
        print(f"   Consider: more data, stronger regularization, or freezing more layers")
    elif current_samples < recommended_samples:
        deficit = recommended_samples - current_samples
        print(f"⚠️  MARGINAL data size (could use {deficit:,} more samples)")
        print(f"   Should be OK with proper regularization (dropout, weight decay)")
    else:
        surplus = current_samples - recommended_samples
        print(f"✓ ADEQUATE data size ({surplus:,} samples above recommended)")
        print(f"   Can potentially unfreeze more layers or reduce regularization")


def main():
    """Main training script."""

    # Configuration
    DATA_ROOT = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v4"
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NEGATIVES_PER_POSITIVE = 4  # Adjusted for your data size

    # DATASET PARAMETERS
    RADIUS = 30
    THRESHOLD = 30

    # DINO PARAMETERS
    DINO_MODEL = "facebook/dinov2-base"

    print(f"Using device: {DEVICE}")

    # Load full dataset
    full_dataset = PrecomposedAlignmentDataset(
        data_root=DATA_ROOT,
        negatives_per_positive=NEGATIVES_PER_POSITIVE,
        hard_negative_ratio=0.6,
        radius=RADIUS,
        threshold=THRESHOLD,
    )

    # *** CHANGED: Split by puzzles, not random ***
    train_dataset, val_dataset = PrecomposedAlignmentDataset.create_puzzle_split(
        full_dataset, radius=RADIUS, threshold=THRESHOLD, train_ratio=0.8, seed=42
    )

    print(f"\n=== Dataset Ready ===")
    print(f"Train: {len(train_dataset)} pairs")
    print(f"Val: {len(val_dataset)} pairs")

    # Verify no overlap
    train_puzzles = set(k.split("|")[0] for k in train_dataset.pair_keys)
    val_puzzles = set(k.split("|")[0] for k in val_dataset.pair_keys)
    overlap = train_puzzles & val_puzzles

    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} puzzles appear in both train and val!")
    else:
        print("✓ No puzzle overlap between train and val")

    print("\n" + "=" * 60)
    print("TRAINING MODEL 5: RGB + Geometry + DINO (with more frozen layers)")
    print("=" * 60)

    model = MultiModalScorerV2_Practical(
        freeze_vit_layers=10,  # Only train last 2 layers
        dropout=0.5
    )

    # Train with strong regularization
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    estimate_data_needs(model, train_dataset)

    # Train with combined BCE + AdaptiveTopNRankingLoss
    model, history = train_model(
        model,
        train_dataset,
        val_dataset,
        optimizer=optimizer,
        num_epochs=5,
        batch_size=8,
        lr=1e-4,
        weight_decay=1e-4,
        early_stopping_patience=3,
        model_name="multimodal_v2_practical",
        # Combined loss weights (ranking-focused)
        bce_weight=0.2,  # Lower weight for BCE classification loss
        ranking_weight=0.8,  # Higher weight for Adaptive ranking loss
        # AdaptiveTopNRankingLoss parameters
        ranking_margin=0.3,  # Margin for ranking loss
        hard_negative_weight=2.0,  # Weight for hard negatives in ranking loss
        top_n=3,  # Target top-3 positions
        temperature=1.0,  # Smoothness of position penalty
    )

    # Use it
    diagnose_data_sufficiency(history)


if __name__ == "__main__":
    main()

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
from models import MultiModalScorerV2_Practical, GeometricScorer
from loss_v2 import AdaptiveTopNRankingLoss, PerceptualBoundaryLoss
from training_utils import plot_training_history, debug_group_structure
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from training_utils import plot_training_history, evaluate_ranking, estimate_data_needs, diagnose_data_sufficiency


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
    bce_weight=0.15,  # Weight for BCE loss (lower for ranking-focused training)
    ranking_weight=0.55,  # Weight for Adaptive ranking loss (higher priority)
    ranking_margin=0.3,  # Margin for ranking loss
    boundary_weight=0.3,
    hard_negative_weight=2.0,  # Weight for hard negatives in ranking loss
    top_n=3,  # Target top-N positions for AdaptiveTopNRankingLoss
    temperature=1.0,  # Temperature for smooth position penalty
    max_norm=1.0 # Gradient clipping (helps with small data)
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

    # for batch in train_loader:
    #     debug_group_structure(batch)
    #     breakpoint()

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
    boundary_criterion = PerceptualBoundaryLoss()  # Start simple
    
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
            group_sizes = batch['group_sizes']

            optimizer.zero_grad()

            # Forward
            if model_name == 'geometric':
                logits = model(rgb_geometric)
            else:
                logits = model(rgb, rgb_geometric)
            scores = torch.sigmoid(logits).squeeze()  # Convert to [0, 1] for ranking loss

            # Loss 1: BCE Loss
            bce_loss = bce_criterion(logits, labels)

            # Loss 2: Ranking Loss
            ranking_loss = ranking_criterion(scores, labels.squeeze(), difficulties, group_sizes=group_sizes)

            # Loss 3: Boundary consistency (auxiliary)
            boundary_loss = boundary_criterion(rgb, rgb_geometric, labels)

            # Combined loss
            loss = bce_weight * bce_loss + ranking_weight * ranking_loss + boundary_weight * boundary_loss

            # Backward
            loss.backward()

            # Gradient clipping (helps with small data)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

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

                if model_name == 'geometric':
                    logits = model(rgb_geometric)
                else:
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
        model_type = "geometric" if "geometric" == model_name else "multimodal"
        val_acc, val_top3_acc, val_top5_acc, avg_pos, avg_neg, avg_hard_neg = evaluate_ranking(model, val_loader, device, model_type=model_type)

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

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce:.4f}, Rank: {avg_train_ranking:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f} (Top3: {val_top3_acc:.3f}, Top5: {val_top5_acc:.3f}) | "
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

def main():
    """Main training script."""

    # Configuration
    DATA_ROOT = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v4"
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_NEGATIVES_PER_POSITIVE = 15 # use all available
      # Adjusted for your data size

    # DATASET PARAMETERS
    RADIUS = 80
    THRESHOLD = 100

    # DINO PARAMETERS
    DINO_MODEL = "facebook/dinov2-base"

    print(f"Using device: {DEVICE}")

    # Load full dataset
    full_dataset = PrecomposedAlignmentDataset(
        data_root=DATA_ROOT,
        max_negatives_per_positive=MAX_NEGATIVES_PER_POSITIVE,
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

    # model = GeometricScorer()
    # breakpoint()
    # model_v2, history_v2 = train_model(
    #     model_v2,
    #     train_dataset,
    #     val_dataset,
    #     num_epochs=NUM_EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     lr=LEARNING_RATE,
    #     device=DEVICE,
    #     model_name='geometric'
    # )

    # Train with strong regularization
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    estimate_data_needs(model, train_dataset)

    # Train with combined BCE + AdaptiveTopNRankingLoss
    model, history = train_model(
        model,
        train_dataset,
        val_dataset,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=1e-4,
        weight_decay=1e-4,
        early_stopping_patience=3,
        model_name="multimodal_boundary2",
        # Combined loss weights (ranking-focused)
        bce_weight=0.15,  # Lower weight for BCE classification loss
        ranking_weight=0.55,  # Higher weight for Adaptive ranking loss
        boundary_weight=0.3,  #  weight for boundary loss
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

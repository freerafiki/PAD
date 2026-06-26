import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(history, save_path):
    """
    Plot training curves in a 2x4 grid.

    Row 0: Loss curves (total + per-component)
    Row 1: Ranking accuracy, score trends, score margins, overfitting gap

    All 8 panels are used — no dead space.
    """
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs, history["train_loss"], label="Train", color="steelblue")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val", color="coral", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss (BCE + Ranking + Boundary)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.4)

    axes[0, 1].plot(epochs, history["train_bce_loss"], label="Train", color="steelblue")
    axes[0, 1].plot(epochs, history["val_bce_loss"], label="Val", color="coral", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("BCE Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.4)

    axes[0, 2].plot(epochs, history["train_ranking_loss"], label="Train", color="steelblue")
    axes[0, 2].plot(epochs, history["val_ranking_loss"], label="Val", color="coral", linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].set_title("Ranking Loss")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.4)

    axes[0, 3].plot(epochs, history["train_boundary_loss"], label="Train", color="steelblue")
    axes[0, 3].plot(epochs, history["val_boundary_loss"], label="Val", color="coral", linewidth=2)
    axes[0, 3].set_xlabel("Epoch")
    axes[0, 3].set_ylabel("Loss")
    axes[0, 3].set_title("Boundary Loss")
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.4)

    axes[1, 0].plot(epochs, history["val_accuracy"], label="Top-1", color="#2ca02c", linewidth=2)
    axes[1, 0].plot(epochs, history["val_top3_accuracy"], label="Top-3", color="#1f77b4", linewidth=2)
    axes[1, 0].plot(epochs, history["val_top5_accuracy"], label="Top-5", color="#9467bd", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Ranking Accuracy")
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].plot(epochs, history["val_pos_score"], label="Positive", color="#2ca02c")
    axes[1, 1].plot(epochs, history["val_neg_score"], label="Negative", color="#d62728")
    axes[1, 1].plot(epochs, history["val_hard_neg_score"], label="Hard Neg", color="#ff7f0e")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Avg Score")
    axes[1, 1].set_title("Average Scores by Class")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.4)

    pos_arr = np.array(history["val_pos_score"])
    neg_arr = np.array(history["val_neg_score"])
    hard_arr = np.array(history["val_hard_neg_score"])
    axes[1, 2].plot(epochs, pos_arr - neg_arr, label="Pos - Neg", color="#9467bd")
    axes[1, 2].plot(epochs, pos_arr - hard_arr, label="Pos - Hard Neg", color="#ff7f0e")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Score Margin")
    axes[1, 2].set_title("Score Margins")
    axes[1, 2].axhline(y=0, color="r", linestyle="--", alpha=0.3)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.4)

    train_arr = np.array(history["train_loss"])
    val_arr = np.array(history["val_loss"])
    gap = val_arr - train_arr
    axes[1, 3].plot(epochs, gap, color="darkorange", linewidth=2)
    axes[1, 3].fill_between(epochs, 0, gap, alpha=0.2, color="darkorange")
    axes[1, 3].set_xlabel("Epoch")
    axes[1, 3].set_ylabel("Val - Train Loss")
    axes[1, 3].set_title("Overfitting Gap")
    axes[1, 3].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[1, 3].grid(True, alpha=0.4)

    fig.suptitle("Training History", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_new_accuracy_metrics(history, save_path):
    """
    Plot the three new accuracy metrics introduced in v26.

    Panel 1: Alignment ranking accuracy (Top-1, Top-3, Top-5)
    Panel 2: Single-alignment classification accuracy (all / positive / negative / hard-negative)
    Panel 3: Alignment set classification score (with partial credit breakdown)
    Panel 4: Score calibration — average scores by class

    Compatible with checkpoints from before these metrics were added
    (uses .get() with defaults).
    """
    if len(history.get("train_loss", [])) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No training history available", ha="center", va="center", transform=ax.transAxes)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved new accuracy metrics plot to {save_path}")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    axes[0, 0].plot(epochs, history.get("val_accuracy", []), label="Top-1", color="#2ca02c", linewidth=2)
    axes[0, 0].plot(epochs, history.get("val_top3_accuracy", []), label="Top-3", color="#1f77b4", linewidth=2)
    axes[0, 0].plot(epochs, history.get("val_top5_accuracy", []), label="Top-5", color="#9467bd", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Alignment Ranking Accuracy")
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.4)

    axes[0, 1].plot(epochs, history.get("val_single_align_accuracy", []), label="All Samples", color="#1f77b4", linewidth=2)
    axes[0, 1].plot(epochs, history.get("val_single_pos_accuracy", []), label="Positive", color="#2ca02c", linewidth=2)
    axes[0, 1].plot(epochs, history.get("val_single_neg_accuracy", []), label="Negative", color="#d62728", linewidth=2)
    axes[0, 1].plot(epochs, history.get("val_single_hard_neg_accuracy", []), label="Hard Neg", color="#ff7f0e", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Single-Alignment Classification (score > 0.5)")
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.4)

    align_set_arr = np.array(history.get("val_align_set_score", []))
    axes[0, 2].plot(epochs, align_set_arr, color="#9467bd", linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Score")
    axes[0, 2].set_title("Alignment Set Score (partial credit)")
    axes[0, 2].set_ylim([0, 1.05])
    axes[0, 2].grid(True, alpha=0.4)

    pos_arr = np.array(history.get("val_pos_score", []))
    neg_arr = np.array(history.get("val_neg_score", []))
    hard_arr = np.array(history.get("val_hard_neg_score", []))
    axes[0, 3].plot(epochs, pos_arr, label="Positive", color="#2ca02c")
    axes[0, 3].plot(epochs, neg_arr, label="Negative", color="#d62728")
    axes[0, 3].plot(epochs, hard_arr, label="Hard Neg", color="#ff7f0e")
    axes[0, 3].axhline(y=0.5, color="k", linestyle="--", alpha=0.3, label="Threshold")
    axes[0, 3].set_xlabel("Epoch")
    axes[0, 3].set_ylabel("Avg Score")
    axes[0, 3].set_title("Average Scores vs 0.5 Threshold")
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.4)

    single_pos_arr = np.array(history.get("val_single_pos_accuracy", []))
    single_neg_arr = np.array(history.get("val_single_neg_accuracy", []))
    single_hard_arr = np.array(history.get("val_single_hard_neg_accuracy", []))
    axes[1, 0].plot(epochs, single_pos_arr - single_neg_arr, label="Pos - Neg", color="#9467bd")
    axes[1, 0].plot(epochs, single_pos_arr - single_hard_arr, label="Pos - Hard Neg", color="#ff7f0e")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy Gap")
    axes[1, 0].set_title("Single-Alignment Accuracy Separation")
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].plot(epochs, history.get("val_single_pos_accuracy", []), label="Pos > 0.5", color="#2ca02c")
    axes[1, 1].plot(epochs, history.get("val_single_neg_accuracy", []), label="Neg <= 0.5", color="#d62728")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Per-Class Threshold Recall")
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, alpha=0.4)

    train_loss_arr = np.array(history.get("train_loss", []))
    val_loss_arr = np.array(history.get("val_loss", []))
    if len(train_loss_arr) > 0:
        gap = val_loss_arr - train_loss_arr
        axes[1, 2].plot(epochs, gap, color="darkorange", linewidth=2)
        axes[1, 2].fill_between(epochs, 0, gap, alpha=0.2, color="darkorange")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Val - Train Loss")
        axes[1, 2].set_title("Overfitting Gap")
        axes[1, 2].axhline(y=0, color="k", linestyle="-", alpha=0.3)
        axes[1, 2].grid(True, alpha=0.4)

    val_bce_arr = np.array(history.get("val_bce_loss", []))
    val_rank_arr = np.array(history.get("val_ranking_loss", []))
    val_bound_arr = np.array(history.get("val_boundary_loss", []))
    val_total = val_bce_arr + val_rank_arr + val_bound_arr
    val_total = np.where(val_total == 0, 1e-8, val_total)
    axes[1, 3].plot(epochs, val_bce_arr / val_total, label="BCE", color="#1f77b4")
    axes[1, 3].plot(epochs, val_rank_arr / val_total, label="Ranking", color="#ff7f0e")
    axes[1, 3].plot(epochs, val_bound_arr / val_total, label="Boundary", color="#2ca02c")
    axes[1, 3].set_xlabel("Epoch")
    axes[1, 3].set_ylabel("Fraction")
    axes[1, 3].set_title("Relative Loss Contribution")
    axes[1, 3].set_ylim([0, 1])
    axes[1, 3].legend(loc="upper right")
    axes[1, 3].grid(True, alpha=0.4)

    fig.suptitle("New Accuracy Metrics", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved new accuracy metrics plot to {save_path}")


def evaluate_ranking(model, dataloader, device, model_type='multimodal'):
    """
    Evaluate ranking accuracy: is positive ranked first in each group?
    Also returns top-3 and top-5 accuracy, and average scores for positives, 
    negatives, and hard negatives.
    """
    model.eval()

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total_groups = 0
    total_groups_with_positive = 0
    all_pos_scores = []
    all_neg_scores = []
    all_hard_neg_scores = []
    alignment_set_score = 0.0
    set_total = 0
    set_positive_full = 0
    set_positive_partial = 0
    set_no_positive_correct = 0
    set_no_positive_wrong = 0

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            rgb_geometric = batch["rgb_geometric"].to(device)
            labels = batch["labels"].to(device)
            difficulties = batch["difficulties"]
            group_sizes = batch['group_sizes']  # *** NEW ***

            # Get logits and convert to probabilities
            if model_type == 'geometric':
                logits = model(rgb_geometric)
            elif model_type == 'baseline' or model_type == 'RGB':
                logits = model(rgb)
            else:
                logits = model(rgb, rgb_geometric)
            scores = torch.sigmoid(logits)  # Convert to [0, 1]

            # breakpoint()
            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
            labels_pt = labels.cpu()

            # Find groups
            start_idx = 0
            for group_size in group_sizes:
                end_idx = start_idx + group_size

                group_scores = scores_np[start_idx:end_idx]
                group_labels = labels_pt[start_idx:end_idx]
                group_labels_np = labels_np[start_idx:end_idx]
                group_difficulties = difficulties[start_idx:end_idx]

                # Validate: should have exactly 1 positive
                pos_mask = group_labels == 1.0
                if pos_mask.sum() != 1:
                    # we are in the "non neighbours" group
                    if pos_mask.sum() > 1: 
                        print(f"⚠️  Warning: we actaully have more than one positive: {pos_mask.sum().item()} positives in this group!")
                    # print(f"⚠️  Warning: Group {group_idx} has {pos_mask.sum().item()} positives")
                
                else:
                    # we are in the "neighbours" group
                    # Get positive and negative scores
                    pos_idx_in_group = torch.where(pos_mask)[0][0].item()
                    neg_mask = ~pos_mask

                    group_scores_arr = np.array(group_scores)
                    if pos_idx_in_group >= len(group_scores_arr):
                        start_idx = end_idx
                        continue

                    # Rank of positive: count how many samples have a higher score
                    pos_rank = int(np.sum(group_scores_arr > group_scores_arr[pos_idx_in_group]))

                    # Top-1: positive has the highest score
                    if pos_rank == 0:
                        correct_top1 += 1
                    
                    # Top-3: positive is in top 3 highest scores
                    if pos_rank < 3:
                        correct_top3 += 1
                    
                    # Top-5: positive is in top 5 highest scores
                    if pos_rank < 5:
                        correct_top5 += 1

                    total_groups_with_positive += 1

                total_groups += 1
                start_idx = end_idx

                # --- Collect all individual sample scores for reporting ---
                for score, label, diff in zip(group_scores, group_labels_np, group_difficulties):
                    if label == 1.0:
                        all_pos_scores.append(score)
                    elif diff == 'hard_negative':
                        all_hard_neg_scores.append(score)
                    else:
                        all_neg_scores.append(score)

                # --- Alignment set classification per group (partial credit) ---
                max_score = float(np.max(group_scores))
                has_positive = pos_mask.sum() == 1

                if has_positive:
                    set_total += 1
                    if max_score > 0.5:
                        alignment_set_score += 1.0
                        set_positive_full += 1
                    else:
                        alignment_set_score += 0.5
                        set_positive_partial += 1
                else:
                    set_total += 1
                    if max_score <= 0.5:
                        alignment_set_score += 1.0
                        set_no_positive_correct += 1
                    else:
                        alignment_set_score += 0.0
                        set_no_positive_wrong += 1

    accuracy_top1 = correct_top1 / total_groups_with_positive if total_groups_with_positive > 0 else 0.0
    accuracy_top3 = correct_top3 / total_groups_with_positive if total_groups_with_positive > 0 else 0.0
    accuracy_top5 = correct_top5 / total_groups_with_positive if total_groups_with_positive > 0 else 0.0
    avg_pos_score = np.mean(all_pos_scores) if all_pos_scores else 0.0
    avg_neg_score = np.mean(all_neg_scores) if all_neg_scores else 0.0
    avg_hard_neg_score = np.mean(all_hard_neg_scores) if all_hard_neg_scores else 0.0

    # --- Single-alignment classification: per-sample accuracy with fixed threshold 0.5 ---
    # Iterate over full batch arrays after the group loop (covers all samples)
    all_single_total = len(scores_np)
    all_single_correct = 0
    pos_correct = 0
    neg_correct = 0
    hard_neg_correct = 0

    for score, label, diff in zip(scores_np, labels_np, difficulties):
        if label == 1.0:
            if score > 0.5:
                pos_correct += 1
            all_single_correct += 1
        else:
            if score <= 0.5:
                neg_correct += 1
                if diff == "hard_negative":
                    hard_neg_correct += 1
            all_single_correct += 1

    single_align_accuracy = all_single_correct / all_single_total if all_single_total > 0 else 0.0
    single_align_pos_accuracy = pos_correct / len(all_pos_scores) if all_pos_scores else 0.0
    single_align_neg_accuracy = neg_correct / len(all_neg_scores) if all_neg_scores else 0.0
    single_align_hard_neg_accuracy = hard_neg_correct / len(all_hard_neg_scores) if all_hard_neg_scores else 0.0

    # Alignment set classification score (partial credit per alignment set)
    align_set_score = alignment_set_score / set_total if set_total > 0 else 0.0

    return (
        accuracy_top1, accuracy_top3, accuracy_top5,
        avg_pos_score, avg_neg_score, avg_hard_neg_score,
        single_align_accuracy, single_align_pos_accuracy,
        single_align_neg_accuracy, single_align_hard_neg_accuracy,
        align_set_score,
        {
            "set_total": set_total,
            "positive_full": set_positive_full,
            "positive_partial": set_positive_partial,
            "no_positive_correct": set_no_positive_correct,
            "no_positive_wrong": set_no_positive_wrong,
        },
    )

def debug_group_structure(batch):
    """
    Verify that group_sizes correctly partition the batch.
    """
    labels = batch['labels']
    group_sizes = batch['group_sizes']
    
    print("\n=== Group Structure Debug ===")
    print(f"Total samples: {len(labels)}")
    print(f"Group sizes: {group_sizes}")
    print(f"Sum of group sizes: {sum(group_sizes)}")
    
    assert len(labels) == sum(group_sizes), "Group sizes don't match batch size!"
    
    start_idx = 0
    for group_idx, group_size in enumerate(group_sizes):
        end_idx = start_idx + group_size
        group_labels = labels[start_idx:end_idx]
        
        num_pos = (group_labels == 1.0).sum().item()
        num_neg = (group_labels == 0.0).sum().item()
        
        print(f"Group {group_idx}: size={group_size}, pos={num_pos}, neg={num_neg}")
        
        if num_pos != 1:
            print(f"  ⚠️  WARNING: Expected 1 positive, got {num_pos}")
        
        start_idx = end_idx
    
    print("✓ Group structure is valid")

# # Use it
# for batch in train_loader:
#     debug_group_structure(batch)
#     break
# ```

# Expected output:
# ```
# === Group Structure Debug ===
# Total samples: 28
# Group sizes: [7, 5, 8, 6, 2]
# Sum of group sizes: 28
# Group 0: size=7, pos=1, neg=6
# Group 1: size=5, pos=1, neg=4
# Group 2: size=8, pos=1, neg=7
# Group 3: size=6, pos=1, neg=5
# Group 4: size=2, pos=1, neg=1
# ✓ Group structure is valid

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

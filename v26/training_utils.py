import matplotlib.pyplot as plt 
import numpy as np 
import torch 

def plot_training_history(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Total Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss (BCE + Ranking)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # BCE Loss
    axes[0, 1].plot(epochs, history["train_bce_loss"], label="Train BCE", color="blue")
    axes[0, 1].plot(epochs, history["val_bce_loss"], label="Val BCE", color="orange")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("BCE Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Ranking Loss
    axes[0, 2].plot(epochs, history["train_ranking_loss"], label="Train Ranking", color="green")
    axes[0, 2].plot(epochs, history["val_ranking_loss"], label="Val Ranking", color="red")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].set_title("Ranking Loss")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Top-1, Top-3, Top-5 Accuracy
    axes[1, 0].plot(epochs, history["val_accuracy"], label="Top-1", color="green", linewidth=2)
    axes[1, 0].plot(epochs, history["val_top3_accuracy"], label="Top-3", color="blue", linewidth=2)
    axes[1, 0].plot(epochs, history["val_top5_accuracy"], label="Top-5", color="purple", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Ranking Accuracy (Top-1, Top-3, Top-5)")
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1.05])

    # Score distribution
    axes[1, 1].plot(epochs, history["val_pos_score"], label="Avg Positive Score", color="green")
    axes[1, 1].plot(epochs, history["val_neg_score"], label="Avg Negative Score", color="red")
    axes[1, 1].plot(epochs, history["val_hard_neg_score"], label="Avg Hard Neg Score", color="orange")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Average Scores by Class")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Score separation (margin)
    margin = np.array(history["val_pos_score"]) - np.array(history["val_neg_score"])
    hard_margin = np.array(history["val_pos_score"]) - np.array(history["val_hard_neg_score"])
    axes[1, 2].plot(epochs, margin, label="Pos - Neg Margin", color="purple")
    axes[1, 2].plot(epochs, hard_margin, label="Pos - Hard Neg Margin", color="orange")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Score Margin")
    axes[1, 2].set_title("Score Margins")
    axes[1, 2].grid(True)
    axes[1, 2].axhline(y=0, color="r", linestyle="--", alpha=0.3)
    axes[1, 2].legend()

    # Top-3 Accuracy detail
    axes[2, 0].plot(epochs, history["val_top3_accuracy"], label="Top-3 Accuracy", color="blue", linewidth=2)
    axes[2, 0].fill_between(epochs, history["val_accuracy"], history["val_top3_accuracy"], 
                            alpha=0.3, color="blue", label="Top-1 to Top-3 gain")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Accuracy")
    axes[2, 0].set_title("Top-3 Accuracy (with Top-1 baseline)")
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    axes[2, 0].set_ylim([0, 1.05])

    # Top-5 Accuracy detail
    axes[2, 1].plot(epochs, history["val_top5_accuracy"], label="Top-5 Accuracy", color="purple", linewidth=2)
    axes[2, 1].fill_between(epochs, history["val_top3_accuracy"], history["val_top5_accuracy"], 
                            alpha=0.3, color="purple", label="Top-3 to Top-5 gain")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Accuracy")
    axes[2, 1].set_title("Top-5 Accuracy (with Top-3 baseline)")
    axes[2, 1].grid(True)
    axes[2, 1].legend()
    axes[2, 1].set_ylim([0, 1.05])

    # Learning rate
    axes[2, 2].plot(epochs, history["learning_rates"], label="Learning Rate", color="orange")
    axes[2, 2].set_xlabel("Epoch")
    axes[2, 2].set_ylabel("Learning Rate")
    axes[2, 2].set_title("Learning Rate Schedule")
    axes[2, 2].grid(True)
    axes[2, 2].legend()
    axes[2, 2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_training_history_old(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Total Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss (BCE + Ranking)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # BCE Loss
    axes[0, 1].plot(epochs, history["train_bce_loss"], label="Train BCE", color="blue")
    axes[0, 1].plot(epochs, history["val_bce_loss"], label="Val BCE", color="orange")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("BCE Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Ranking Loss
    axes[0, 2].plot(epochs, history["train_ranking_loss"], label="Train Ranking", color="green")
    axes[0, 2].plot(epochs, history["val_ranking_loss"], label="Val Ranking", color="red")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].set_title("Ranking Loss")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Accuracy
    axes[1, 0].plot(
        epochs, history["val_accuracy"], label="Val Accuracy", color="green"
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Validation Accuracy (Positive Ranked First)")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Score distribution
    axes[1, 1].plot(epochs, history["val_pos_score"], label="Avg Positive Score", color="green")
    axes[1, 1].plot(epochs, history["val_neg_score"], label="Avg Negative Score", color="red")
    axes[1, 1].plot(epochs, history["val_hard_neg_score"], label="Avg Hard Neg Score", color="orange")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Average Scores by Class")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Score separation (margin)
    margin = np.array(history["val_pos_score"]) - np.array(history["val_neg_score"])
    hard_margin = np.array(history["val_pos_score"]) - np.array(history["val_hard_neg_score"])
    axes[1, 2].plot(epochs, margin, label="Pos - Neg Margin", color="purple")
    axes[1, 2].plot(epochs, hard_margin, label="Pos - Hard Neg Margin", color="orange")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Score Margin")
    axes[1, 2].set_title("Score Margins")
    axes[1, 2].grid(True)
    axes[1, 2].axhline(y=0, color="r", linestyle="--", alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {save_path}")


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
    all_pos_scores = []
    all_neg_scores = []
    all_hard_neg_scores = []

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
            else:
                logits = model(rgb, rgb_geometric)
            scores = torch.sigmoid(logits)  # Convert to [0, 1]

            # breakpoint()
            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Find groups
            start_idx = 0
            for group_size in group_sizes:
                end_idx = start_idx + group_size

                group_scores = scores_np[start_idx:end_idx]
                group_labels = labels_np[start_idx:end_idx]
                group_difficulties = difficulties[start_idx:end_idx]

                # Find positive (should be exactly one)
                pos_mask = group_labels == 1.0
                if pos_mask.sum() != 1:
                    print(f"⚠️  Warning: Group has {pos_mask.sum()} positives (expected 1)")
                    start_idx = end_idx
                    continue

                pos_idx_in_group = np.where(pos_mask)[0][0]
                pos_score = group_scores[pos_idx_in_group]
                
                # Check if positive has highest score
                if pos_score == group_scores.max():
                    correct += 1
                
                total_groups += 1
                
                # Collect statistics
                all_pos_scores.append(pos_score)
                all_neg_scores.extend(group_scores[~pos_mask])
                
                start_idx = end_idx
                
                # Top-1: positive is ranked first
                if pos_idx_in_group == 0:
                    correct_top1 += 1
                
                # Top-3: positive is in top 3
                if pos_idx_in_group < 3:
                    correct_top3 += 1
                
                # Top-5: positive is in top 5
                if pos_idx_in_group < 5:
                    correct_top5 += 1

                total_groups += 1

                # Collect scores by type
                for score, label, diff in zip(group_scores, group_labels, group_difficulties):
                    if label == 1.0:
                        all_pos_scores.append(score)
                    elif diff == 'hard_negative':
                        all_hard_neg_scores.append(score)
                    else:
                        all_neg_scores.append(score)

    accuracy_top1 = correct_top1 / total_groups if total_groups > 0 else 0.0
    accuracy_top3 = correct_top3 / total_groups if total_groups > 0 else 0.0
    accuracy_top5 = correct_top5 / total_groups if total_groups > 0 else 0.0
    avg_pos_score = np.mean(all_pos_scores) if all_pos_scores else 0.0
    avg_neg_score = np.mean(all_neg_scores) if all_neg_scores else 0.0
    avg_hard_neg_score = np.mean(all_hard_neg_scores) if all_hard_neg_scores else 0.0

    return accuracy_top1, accuracy_top3, accuracy_top5, avg_pos_score, avg_neg_score, avg_hard_neg_score

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

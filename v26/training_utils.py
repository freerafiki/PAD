import matplotlib.pyplot as plt 
import numpy as np 

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


def evaluate_ranking(model, dataloader, device):
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

                # Find the rank of the positive sample
                # Sort indices by score (descending)
                sorted_indices = np.argsort(-group_scores)
                # Find position of positive (index 0 in group_labels since positive is first)
                positive_rank = np.where(sorted_indices == 0)[0][0]  # 0-indexed rank
                
                # Top-1: positive is ranked first
                if positive_rank == 0:
                    correct_top1 += 1
                
                # Top-3: positive is in top 3
                if positive_rank < 3:
                    correct_top3 += 1
                
                # Top-5: positive is in top 5
                if positive_rank < 5:
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
    avg_hard_neg_score = np.mean(all_hard_neg_scores) if all_hard_neg_scores else 0.0

    return accuracy, avg_pos_score, avg_neg_score, avg_hard_neg_score


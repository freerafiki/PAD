"""
Checkpoint Analysis Script

Deep-dive analysis of a trained ranking model checkpoint.
Produces detailed diagnostic plots that go beyond the basic training history.

Usage:
    python analyze_checkpoint.py --checkpoint checkpoints/multimodal_bceW_wikiart_frozen8_best.pth

Produces a multi-page PDF with the following analyses:
    Page 1: Score histograms + score calibration
    Page 2: Per-difficulty accuracy breakdown
    Page 3: Failure case analysis (worst-ranking groups)
    Page 4: Loss decomposition over training

All outputs are saved to the same directory as the checkpoint.

IMPORTANT: The checkpoint must contain 'history' and 'model_state_dict'.
The model architecture must match the checkpoint (same class, same layer names).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from dataset_ranking import (
    PrecomposedAlignmentDataset,
    ShuffledBatchSampler,
    collate_alignment_samples,
)
from models_ranking import (
    BaselineScorer,
    GeometricScorer,
    MultiModalScorerV2_Practical,
    MultiModalScorerWeightedVit,
)
from training_utils import evaluate_ranking


# ============================================================
# MODEL DETECTION
# ============================================================

def get_model_type_from_ckpt(ckpt):
    """
    Determine which model class to instantiate based on checkpoint keys.

    We inspect the state_dict keys to figure out which architecture was used.
    This avoids hardcoding the model name and makes the script robust to
    different training runs.

    Returns:
        model_class: The nn.Module class to instantiate
        model_name: Human-readable name for labels
        model_type: 'baseline', 'geometric', or 'multimodal' (for data loading)
    """
    keys = ckpt["model_state_dict"].keys()
    key_str = " ".join(keys)

    if "dino" in key_str:
        if "contact_proj" in key_str or "weighted" in key_str:
            return MultiModalScorerWeightedVit, "MultiModalScorerWeightedVit", "multimodal"
        return MultiModalScorerV2_Practical, "MultiModalScorerV2_Practical", "multimodal"
    elif "projection" in key_str and "vit" in key_str:
        return GeometricScorer, "GeometricScorer", "geometric"
    else:
        return BaselineScorer, "BaselineScorer", "baseline"


# ============================================================
# DATA LOADING
# ============================================================

def load_data_and_model(ckpt_path, cfg):
    """
    Load dataset, model, and checkpoint state.

    This function:
    1. Loads the checkpoint file
    2. Creates the full dataset and train/val split (same as training)
    3. Instantiates the correct model class
    4. Loads the saved weights
    5. Sets the model to eval mode

    Returns:
        model, val_dataset, val_loader, model_type, model_name, ckpt
    """
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=cfg.training.DEVICE, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=cfg.training.DEVICE, weights_only=False)

    model_class, model_name, model_type = get_model_type_from_ckpt(ckpt)
    print(f"Detected model: {model_name} (type: {model_type})")

    print(f"\nLoading dataset from: {cfg.data.DATA_ROOT}")
    full_dataset = PrecomposedAlignmentDataset(
        data_root=cfg.data.DATA_ROOT,
        max_negatives_per_positive=cfg.data.MAX_NEGATIVES_PER_POSITIVE,
        min_negatives_per_positive=cfg.data.MIN_NEGATIVES_PER_POSITIVE,
        radius=cfg.data.RADIUS,
        threshold=cfg.data.THRESHOLD,
        debug_mode=cfg.data.DEBUG,
    )

    _, val_dataset = PrecomposedAlignmentDataset.create_puzzle_split(
        full_dataset,
        radius=cfg.data.RADIUS,
        threshold=cfg.data.THRESHOLD,
        train_ratio=cfg.data.TRAIN_RATIO,
        seed=cfg.data.SEED,
    )
    print(f"  Val pairs: {len(val_dataset)}")

    if model_class == MultiModalScorerV2_Practical:
        model = MultiModalScorerV2_Practical(
            dino_model=cfg.model.DINO_MODEL,
            geometric_vit=cfg.model.VIT_MODEL,
            freeze_vit_layers=cfg.model.FROZEN_LAYERS,
            dropout=cfg.model.DROPOUT,
            geometric_channel_scale=cfg.model.GEOMETRIC_CHANNEL_SCALE,
        )
    elif model_class == MultiModalScorerWeightedVit:
        model = MultiModalScorerWeightedVit(
            dino_model=cfg.model.DINO_MODEL,
            geometric_vit=cfg.model.VIT_MODEL,
            freeze_vit_layers=cfg.model.FROZEN_LAYERS,
            dropout=cfg.model.DROPOUT,
            geometric_channel_scale=cfg.model.GEOMETRIC_CHANNEL_SCALE,
        )
    elif model_class == GeometricScorer:
        model = GeometricScorer(
            pretrained_name=cfg.model.VIT_MODEL,
            geometric_channel_scale=cfg.model.GEOMETRIC_CHANNEL_SCALE,
        )
    else:
        model = BaselineScorer(pretrained_name=cfg.model.VIT_MODEL)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(cfg.training.DEVICE)
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.BATCH_SIZE,
        shuffle=False,
        sampler=ShuffledBatchSampler(val_dataset, shuffle=False, seed=42),
        collate_fn=collate_alignment_samples,
        num_workers=4,
        pin_memory=True,
    )

    return model, val_dataset, val_loader, model_type, model_name, ckpt


# ============================================================
# COLLECT ALL PREDICTIONS
# ============================================================

def collect_predictions(model, loader, device, model_type):
    """
    Run a full pass over the dataset and collect all per-sample predictions.

    Returns a dict with:
        - scores: array of all predicted scores (after sigmoid)
        - labels: array of all ground-truth labels (1.0 = positive, 0.0 = negative)
        - difficulties: list of difficulty strings per sample
        - group_sizes: list of group sizes in order
        - pair_keys: list of pair keys (one per group, aligned with group_sizes)

    This data is used by all subsequent analysis functions.
    """
    all_scores = []
    all_labels = []
    all_difficulties = []
    group_sizes = []
    pair_keys = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            rgb_geometric = batch["rgb_geometric"].to(device)
            labels = batch["labels"]
            difficulties = batch["difficulties"]
            sizes = batch["group_sizes"]
            pks = batch.get("pair_key", [""] * len(sizes))

            if model_type == "geometric":
                logits = model(rgb_geometric)
            elif model_type == "baseline":
                logits = model(rgb)
            else:
                logits = model(rgb, rgb_geometric)

            scores = torch.sigmoid(logits).cpu().numpy().flatten()
            labels_np = labels.numpy().flatten()

            all_scores.extend(scores.tolist())
            all_labels.extend(labels_np.tolist())
            all_difficulties.extend(difficulties)
            group_sizes.extend(sizes)
            pair_keys.extend(pks)

    return {
        "scores": np.array(all_scores),
        "labels": np.array(all_labels),
        "difficulties": all_difficulties,
        "group_sizes": group_sizes,
        "pair_keys": pair_keys,
    }


def compute_new_metrics(preds):
    """
    Compute single-alignment classification and alignment-set metrics from collected predictions.

    Single-alignment classification:
    - Correct if: positive label AND score > 0.5, OR negative label AND score <= 0.5
    - Returns overall accuracy plus breakdowns by positive/negative/hard-negative

    Alignment-set score:
    - Groups with positive: 1.0 pts if max > 0.5, 0.5 pts if max <= 0.5
    - Groups without positive: 1.0 pts if max <= 0.5, 0.0 pts if max > 0.5 (false positive)
    - Score = total_points / total_groups

    Returns dict with all computed values for display.
    """
    scores = preds["scores"]
    labels = preds["labels"]
    diffs = preds["difficulties"]
    group_sizes = preds["group_sizes"]

    scores_np = scores
    labels_np = labels

    total_samples = len(scores_np)
    single_align_correct = 0
    pos_correct = 0
    neg_correct = 0
    hard_neg_correct = 0
    num_pos = 0
    num_neg = 0
    num_hard_neg = 0

    for score, label, diff in zip(scores_np, labels_np, diffs):
        is_positive = label == 1.0
        if is_positive:
            num_pos += 1
            if score > 0.5:
                pos_correct += 1
            single_align_correct += 1
        else:
            num_neg += 1
            if score <= 0.5:
                neg_correct += 1
                if diff == "hard_negative":
                    hard_neg_correct += 1
            single_align_correct += 1

    single_align_accuracy = single_align_correct / total_samples if total_samples > 0 else 0.0
    single_align_pos_accuracy = pos_correct / num_pos if num_pos > 0 else 0.0
    single_align_neg_accuracy = neg_correct / num_neg if num_neg > 0 else 0.0
    single_align_hard_neg_accuracy = hard_neg_correct / num_hard_neg if num_hard_neg > 0 else 0.0

    alignment_set_score = 0.0
    set_total = 0
    set_positive_full = 0
    set_positive_partial = 0
    set_no_positive_correct = 0
    set_no_positive_wrong = 0

    start = 0
    for size in group_sizes:
        end = start + size
        group_scores = scores_np[start:end]
        group_labels = labels_np[start:end]

        max_score = float(np.max(group_scores))
        has_positive = np.sum(group_labels == 1.0) == 1

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

        start = end

    align_set_score = alignment_set_score / set_total if set_total > 0 else 0.0

    return {
        "single_align_accuracy": single_align_accuracy,
        "single_align_pos_accuracy": single_align_pos_accuracy,
        "single_align_neg_accuracy": single_align_neg_accuracy,
        "single_align_hard_neg_accuracy": single_align_hard_neg_accuracy,
        "align_set_score": align_set_score,
        "set_total": set_total,
        "set_positive_full": set_positive_full,
        "set_positive_partial": set_positive_partial,
        "set_no_positive_correct": set_no_positive_correct,
        "set_no_positive_wrong": set_no_positive_wrong,
    }


# ============================================================
# PAGE 1: Score Histograms + Calibration
# ============================================================

def plot_score_histograms(preds, save_path):
    """
    Plot 4-panel figure showing score distributions.

    Panel A: Full score histogram — Are scores bimodal (good separation) or
             unimodal centered at 0.5 (poor discrimination)?

    Panel B: Positive scores only — Where does the model place true positives?
             Ideally concentrated near 1.0.

    Panel C: Negative scores only — Where does the model place negatives?
             Ideally concentrated near 0.0.

    Panel D: Hard negative scores only — The most challenging samples.
             If these are near 1.0, the model is confusing them with positives.

    Interpretation guide:
    - Well-separated model: bimodal full histogram, positive peak near 1.0,
      negative peak near 0.0, hard-neg peak near 0.3-0.5.
    - Poor model: single peak near 0.5 for all categories.
    - Overconfident model: all scores at extremes (0 or 1), even for negatives.
    """
    scores = preds["scores"]
    labels = preds["labels"]
    diffs = preds["difficulties"]

    pos_scores = scores[labels == 1.0]
    neg_scores = scores[labels == 0.0]
    hard_neg_scores = np.array([
        s for s, d in zip(scores, diffs) if d == "hard_negative"
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    bins = np.linspace(0, 1, 50)

    axes[0, 0].hist(scores, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0, 0].set_xlabel("Predicted Score")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("All Scores — Shape Indicates Discrimination")
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(pos_scores, bins=bins, color="#2ca02c", edgecolor="white", alpha=0.8)
    axes[0, 1].axvline(x=pos_scores.mean(), color="#2ca02c", linestyle="--", label=f"Mean: {pos_scores.mean():.3f}")
    axes[0, 1].set_xlabel("Score")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Positive Samples — Target: Near 1.0")
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(neg_scores, bins=bins, color="#d62728", edgecolor="white", alpha=0.8)
    axes[1, 0].axvline(x=neg_scores.mean(), color="#d62728", linestyle="--", label=f"Mean: {neg_scores.mean():.3f}")
    axes[1, 0].set_xlabel("Score")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Negative Samples — Target: Near 0.0")
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if len(hard_neg_scores) > 0:
        axes[1, 1].hist(hard_neg_scores, bins=bins, color="#ff7f0e", edgecolor="white", alpha=0.8)
        axes[1, 1].axvline(x=hard_neg_scores.mean(), color="#ff7f0e", linestyle="--",
                          label=f"Mean: {hard_neg_scores.mean():.3f}")
    else:
        axes[1, 1].text(0.5, 0.5, "No hard negatives", ha="center", va="center",
                        transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlabel("Score")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Hard Negatives — Target: < 0.5")
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Score Distribution Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# PAGE 2: Per-Difficulty Accuracy Breakdown
# ============================================================

def plot_difficulty_breakdown(preds, save_path):
    """
    Plot 3-panel figure showing ranking accuracy by difficulty category.

    Panel A: Stacked bar chart — For each difficulty (easy, hard), shows how many
             groups had the positive ranked at position 1, 2-3, 4-5, or 6+.
             This reveals which difficulty levels are most challenging.

    Panel B: Accuracy by difficulty — Top-1 accuracy for each difficulty category.
             A clear drop from easy to hard is expected; a flat line means the
             model doesn't distinguish difficulty well.

    Panel C: Score separation by difficulty — For each difficulty, shows the
             average score margin (positive - mean negative). Positive margins
             mean correct ranking; negative means the model got it wrong.

    This breakdown is critical for understanding where the model struggles
    and guides data augmentation or loss function adjustments.
    """
    scores = preds["scores"]
    labels = preds["labels"]
    diffs = preds["difficulties"]
    group_sizes = preds["group_sizes"]

    start = 0
    easy_groups = []
    hard_groups = []
    all_groups = []

    for size in group_sizes:
        end = start + size
        group_scores = scores[start:end]
        group_labels = labels[start:end]
        group_diffs = diffs[start:end]

        pos_mask = group_labels == 1.0
        if pos_mask.sum() == 1:
            pos_idx = np.where(pos_mask)[0][0]
            rank = int(np.sum(group_scores > group_scores[pos_idx]))

            avg_neg = group_scores[~pos_mask].mean() if (~pos_mask).sum() > 0 else 0
            margin = group_scores[pos_idx] - avg_neg

            has_hard = any(d == "hard_negative" for d in group_diffs)

            entry = {"rank": rank, "pos_score": group_scores[pos_idx], "margin": margin}
            all_groups.append(entry)

            if has_hard:
                hard_groups.append(entry)
            else:
                easy_groups.append(entry)

        start = end

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    categories = ["All Groups", "Easy (No Hard Neg)", "Hard Neg Present"]
    group_lists = [all_groups, easy_groups, hard_groups]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    rank_buckets = {"Top-1": 0, "Top-2/3": 0, "Top-4/5": 0, "Rank 6+": 0}
    for idx, (cat, groups) in enumerate(zip(categories, group_lists)):
        if len(groups) == 0:
            axes[0].bar(idx, 0, color=colors[idx])
            continue
        buckets = {"Top-1": 0, "Top-2/3": 0, "Top-4/5": 0, "Rank 6+": 0}
        for g in groups:
            r = g["rank"]
            if r == 0:
                buckets["Top-1"] += 1
            elif r <= 2:
                buckets["Top-2/3"] += 1
            elif r <= 4:
                buckets["Top-4/5"] += 1
            else:
                buckets["Rank 6+"] += 1

        bottom = np.zeros(4)
        bucket_names = list(buckets.keys())
        for bi, bn in enumerate(bucket_names):
            axes[0].bar(idx, buckets[bn], bottom=bottom[bi], color=colors[idx],
                       alpha=0.7, edgecolor="white", label=bn if idx == 0 else "")
            bottom[bi] += buckets[bn]

    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(categories, rotation=15, ha="right")
    axes[0].set_ylabel("Number of Groups")
    axes[0].set_title("Rank Distribution by Difficulty")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, axis="y", alpha=0.3)

    accuracies = []
    for groups in group_lists:
        if len(groups) == 0:
            accuracies.append(0)
        else:
            top1 = sum(1 for g in groups if g["rank"] == 0)
            accuracies.append(top1 / len(groups))

    axes[1].bar(categories, accuracies, color=colors, edgecolor="white", alpha=0.8)
    axes[1].set_ylim([0, 1.05])
    axes[1].set_ylabel("Top-1 Accuracy")
    axes[1].set_title("Top-1 Accuracy by Difficulty")
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    margins = []
    for groups in group_lists:
        if len(groups) == 0:
            margins.append(0)
        else:
            margins.append(np.mean([g["margin"] for g in groups]))

    axes[2].bar(categories, margins, color=colors, edgecolor="white", alpha=0.8)
    axes[2].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[2].set_ylabel("Avg Score Margin (Pos - Neg)")
    axes[2].set_title("Score Separation by Difficulty")
    axes[2].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(margins):
        axes[2].text(i, v + (0.01 if v > 0 else -0.03), f"{v:.3f}", ha="center", fontweight="bold")

    fig.suptitle("Difficulty Breakdown Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# PAGE 3: Failure Case Analysis
# ============================================================

def plot_failure_cases(preds, save_path):
    """
    Plot 3-panel figure analyzing the worst failure cases.

    Panel A: Histogram of positive ranks — How often does the positive end up
             at rank 1, 2, 3, ...? A right-skewed distribution (peak at 0) is ideal.
             This is the single most important diagnostic for ranking quality.

    Panel B: Failure severity — For groups where the positive is NOT ranked first,
             what is the score of the sample that beat it? If this is close to the
             positive's score, the failure is "near-miss" (model almost got it right).
             If it's much higher, the model is fundamentally confused.

    Panel C: Score gap analysis — For failure cases, shows the gap between the
             positive score and the score of the sample ranked above it.
             Positive gaps mean the model should have ranked correctly but didn't
             (ties or numerical issues). Negative gaps mean the model was wrong.

    This analysis tells you whether failures are:
    - Ambiguous cases (small gaps): May need more data or better features
    - Clear errors (large gaps): May need loss function adjustment or more training
    """
    scores = preds["scores"]
    labels = preds["labels"]
    group_sizes = preds["group_sizes"]

    start = 0
    ranks = []
    failure_gaps = []
    failure_pos_scores = []
    failure_beater_scores = []

    for size in group_sizes:
        end = start + size
        group_scores = scores[start:end]
        group_labels = labels[start:end]

        pos_mask = group_labels == 1.0
        if pos_mask.sum() == 1:
            pos_idx = np.where(pos_mask)[0][0]
            rank = int(np.sum(group_scores > group_scores[pos_idx]))
            ranks.append(rank)

            if rank > 0:
                sorted_scores = np.sort(group_scores)[::-1]
                beater_score = sorted_scores[0]
                pos_score = group_scores[pos_idx]
                failure_gaps.append(pos_score - beater_score)
                failure_pos_scores.append(pos_score)
                failure_beater_scores.append(beater_score)

        start = end

    ranks = np.array(ranks)
    failure_gaps = np.array(failure_gaps)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    max_rank = min(int(ranks.max()) + 1, 20)
    rank_counts = np.zeros(max_rank)
    for r in ranks:
        if r < max_rank:
            rank_counts[r] += 1

    bar_colors = ["#2ca02c"] + ["#ff7f0e"] * (max_rank - 1)
    axes[0].bar(range(max_rank), rank_counts, color=bar_colors, edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Rank of Positive Sample")
    axes[0].set_ylabel("Number of Groups")
    axes[0].set_title("Positive Rank Distribution")
    axes[0].set_xticks(range(0, max_rank, max(1, max_rank // 10)))
    axes[0].grid(True, axis="y", alpha=0.3)

    top1_count = rank_counts[0] if len(rank_counts) > 0 else 0
    total = len(ranks)
    axes[0].text(0.02, 0.95, f"Top-1: {top1_count}/{total} ({top1_count/total:.1%})" if total > 0 else "No data",
                 transform=axes[0].transAxes, va="top", fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if len(failure_gaps) > 0:
        axes[1].hist(failure_gaps, bins=40, color="#d62728", edgecolor="white", alpha=0.8)
        axes[1].axvline(x=0, color="k", linestyle="--", alpha=0.5, label="Zero gap")
        axes[1].set_xlabel("Score Gap (Positive - Best Negative)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Failure Severity — Score Gaps")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        near_miss = np.sum(failure_gaps > -0.1)
        clear_error = np.sum(failure_gaps <= -0.1)
        axes[1].text(0.02, 0.95,
                     f"Near-miss (gap > -0.1): {near_miss} ({near_miss/len(failure_gaps):.1%})\n"
                     f"Clear error (gap <= -0.1): {clear_error} ({clear_error/len(failure_gaps):.1%})"
                     if len(failure_gaps) > 0 else "",
                     transform=axes[1].transAxes, va="top", fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        axes[2].scatter(failure_pos_scores, failure_beater_scores, alpha=0.5, s=20, c="#1f77b4")
        axes[2].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Equal scores")
        axes[2].set_xlabel("Positive Score")
        axes[2].set_ylabel("Best Negative Score")
        axes[2].set_title("Positive vs Best Negative (Failures)")
        axes[2].set_xlim([0, 1])
        axes[2].set_ylim([0, 1])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No failures!", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=16, fontweight="bold", color="green")
        axes[2].text(0.5, 0.5, "Perfect ranking!", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=16, fontweight="bold", color="green")

    fig.suptitle("Failure Case Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# PAGE 4: Loss Decomposition Over Training
# ============================================================

def plot_loss_decomposition(history, save_path):
    """
    Plot 3-panel figure showing how loss terms evolved during training.

    Panel A: Stacked area chart — Shows the contribution of each loss term
             (BCE, Ranking, Boundary) to the total validation loss over time.
             Reveals which loss dominates at different training stages.

    Panel B: Relative contribution — Each loss term as a percentage of the total.
             Helps identify if one loss is being ignored by the optimizer.

    Panel C: Train vs Val loss ratio — For each loss term, shows val/train ratio.
             A ratio > 1 means the loss is higher on validation (overfitting for
             that specific objective). A ratio < 1 means validation is easier.

    This analysis is particularly useful when adjusting loss weights. If one
    term dominates, its weight can be reduced. If a term is ignored, its weight
    can be increased.
    """
    if history is None or len(history.get("train_loss", [])) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No training history available in checkpoint",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path} (no history)")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    train_bce = np.array(history["train_bce_loss"])
    train_rank = np.array(history["train_ranking_loss"])
    train_bound = np.array(history["train_boundary_loss"])
    val_bce = np.array(history["val_bce_loss"])
    val_rank = np.array(history["val_ranking_loss"])
    val_bound = np.array(history["val_boundary_loss"])

    axes[0].stackplot(epochs, val_bce, val_rank, val_bound,
                     labels=["BCE", "Ranking", "Boundary"],
                     colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
                     alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Loss Decomposition (Val)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    total_val = val_bce + val_rank + val_bound
    total_val = np.where(total_val == 0, 1e-8, total_val)
    axes[1].stackplot(epochs,
                     val_bce / total_val,
                     val_rank / total_val,
                     val_bound / total_val,
                     labels=["BCE %", "Ranking %", "Boundary %"],
                     colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
                     alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Fraction of Total Loss")
    axes[1].set_title("Relative Loss Contribution (Val)")
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    train_total = train_bce + train_rank + train_bound
    val_total = val_bce + val_rank + val_bound
    train_total = np.where(train_total == 0, 1e-8, train_total)
    val_total = np.where(val_total == 0, 1e-8, val_total)

    total_ratio = val_total / train_total
    bce_ratio = np.where(train_bce == 0, 1, val_bce / train_bce)
    rank_ratio = np.where(train_rank == 0, 1, val_rank / train_rank)
    bound_ratio = np.where(train_bound == 0, 1, val_bound / train_bound)

    axes[2].plot(epochs, total_ratio, label="Total", color="black", linewidth=2)
    axes[2].plot(epochs, bce_ratio, label="BCE", color="#1f77b4")
    axes[2].plot(epochs, rank_ratio, label="Ranking", color="#ff7f0e")
    axes[2].plot(epochs, bound_ratio, label="Boundary", color="#2ca02c")
    axes[2].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="No overfitting")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Val / Train Ratio")
    axes[2].set_title("Train-Val Gap by Loss Term")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Loss Decomposition Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# SUMMARY REPORT
# ============================================================

def print_summary(preds, history, model_name, ckpt):
    """
    Print a text summary of the model's performance and key metrics.

    This goes to stdout for quick review without opening plots.
    """
    scores = preds["scores"]
    labels = preds["labels"]
    diffs = preds["difficulties"]
    group_sizes = preds["group_sizes"]

    print("\n" + "=" * 60)
    print(f"  CHECKPOINT ANALYSIS SUMMARY")
    print(f"  Model: {model_name}")
    print("=" * 60)

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_val_acc = ckpt.get("val_accuracy", "?")
    if isinstance(ckpt_val_acc, float):
        ckpt_val_acc = f"{ckpt_val_acc:.3f}"
    print(f"\n  Checkpoint epoch: {ckpt_epoch}")
    print(f"  Checkpoint val accuracy (ranking): {ckpt_val_acc}")

    if history and len(history.get("val_accuracy", [])) > 0:
        best_epoch = np.argmax(history["val_accuracy"]) + 1
        final_epoch = len(history["val_accuracy"])
        print(f"  Best epoch (from history): {best_epoch}/{final_epoch} "
              f"(acc: {max(history['val_accuracy']):.3f})")

    start = 0
    total_groups = 0
    top1_correct = 0
    failures_by_rank = {}

    for size in group_sizes:
        end = start + size
        group_scores = scores[start:end]
        group_labels = labels[start:end]

        pos_mask = group_labels == 1.0
        if pos_mask.sum() == 1:
            pos_idx = np.where(pos_mask)[0][0]
            rank = int(np.sum(group_scores > group_scores[pos_idx]))
            if rank == 0:
                top1_correct += 1
            else:
                failures_by_rank[rank] = failures_by_rank.get(rank, 0) + 1
            total_groups += 1

        start = end

    if total_groups > 0:
        print(f"\n  Alignment Ranking (groups with positive only):")
        print(f"    Top-1 accuracy: {top1_correct/total_groups:.3f} ({top1_correct}/{total_groups})")

        if failures_by_rank:
            print(f"    Failure rank distribution:")
            for rank in sorted(failures_by_rank.keys()):
                count = failures_by_rank[rank]
                print(f"      Rank {rank+1}: {count} groups ({count/total_groups:.1%})")

    # Compute the new metrics
    m = compute_new_metrics(preds)
    single_align_acc = m["single_align_accuracy"]
    single_pos_acc = m["single_align_pos_accuracy"]
    single_neg_acc = m["single_align_neg_accuracy"]
    single_hard_neg_acc = m["single_align_hard_neg_accuracy"]
    align_set = m["align_set_score"]
    set_total = m["set_total"]
    set_positive_full = m["set_positive_full"]
    set_positive_partial = m["set_positive_partial"]
    set_no_positive_correct = m["set_no_positive_correct"]
    set_no_positive_wrong = m["set_no_positive_wrong"]

    # Count samples by type for the breakdown
    num_pos = int(np.sum(labels == 1.0))
    num_neg = int(np.sum(labels == 0.0))
    num_hard_neg = int(sum(1 for d in diffs if d == "hard_negative"))

    pos_correct = int(single_pos_acc * num_pos) if num_pos > 0 else 0
    neg_correct = int(single_neg_acc * num_neg) if num_neg > 0 else 0
    hard_neg_correct = int(single_hard_neg_acc * num_hard_neg) if num_hard_neg > 0 else 0

    pos_s = scores[labels == 1.0]
    neg_s = scores[labels == 0.0]
    hard_s = np.array([s for s, d in zip(scores, diffs) if d == "hard_negative"])

    print(f"\n  Score statistics:")
    print(f"    Positive:     mean={pos_s.mean():.3f}, std={pos_s.std():.3f}")
    print(f"    Negative:     mean={neg_s.mean():.3f}, std={neg_s.std():.3f}")
    if len(hard_s) > 0:
        print(f"    Hard Neg:     mean={hard_s.mean():.3f}, std={hard_s.std():.3f}")
    print(f"    Margin (pos-neg): {pos_s.mean() - neg_s.mean():.3f}")
    if len(hard_s) > 0:
        print(f"    Margin (pos-hard): {pos_s.mean() - hard_s.mean():.3f}")

    print(f"\n  Single-Alignment Classification (threshold = 0.5):")
    print(f"    Overall accuracy: {single_align_acc:.3f}")
    print(f"    Positive samples:  {single_pos_acc:.3f} correct ({pos_correct}/{num_pos})")
    print(f"    Negative samples: {single_neg_acc:.3f} correct ({neg_correct}/{num_neg})")
    print(f"    Hard neg samples:  {single_hard_neg_acc:.3f} correct ({hard_neg_correct}/{num_hard_neg})")

    print(f"\n  Alignment Set Score (partial credit = 0.5 per partial group):")
    print(f"    Overall score: {align_set:.3f}")
    print(f"    Total sets: {set_total}")
    print(f"    With positive, max > 0.5 (full): {set_positive_full}")
    print(f"    With positive, max <= 0.5 (partial): {set_positive_partial}")
    print(f"    No positive, max <= 0.5 (correct): {set_no_positive_correct}")
    print(f"    No positive, max > 0.5 (FALSE POSITIVE): {set_no_positive_wrong}  <- most critical metric")
    if set_no_positive_wrong > 0 and set_total > 0:
        print(f"    False positive rate: {set_no_positive_wrong/set_total:.1%}")

    print("\n" + "=" * 60)
    print(f"  CHECKPOINT ANALYSIS SUMMARY")
    print(f"  Model: {model_name}")
    print("=" * 60)

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_val_acc = ckpt.get("val_accuracy", "?")
    if isinstance(ckpt_val_acc, float):
        ckpt_val_acc = f"{ckpt_val_acc:.3f}"
    print(f"\n  Checkpoint epoch: {ckpt_epoch}")
    print(f"  Checkpoint val accuracy: {ckpt_val_acc}")

    if history and len(history.get("val_accuracy", [])) > 0:
        best_epoch = np.argmax(history["val_accuracy"]) + 1
        final_epoch = len(history["val_accuracy"])
        print(f"  Best epoch (from history): {best_epoch}/{final_epoch} "
              f"(acc: {max(history['val_accuracy']):.3f})")

    print("\n" + "=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main entry point for checkpoint analysis.

    Modes:
        --quick  : Load checkpoint metadata only, plot loss decomposition. Instant.
        (default): Load model + dataset, collect predictions, run full analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze a trained ranking model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the checkpoint .pth file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save analysis plots (default: <checkpoint_parent>/<checkpoint_name>/)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: skip model+dataset loading, only plot loss decomposition from history")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / ckpt_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    if args.quick:
        # Quick mode: only analyze history from checkpoint, no model or dataset
        print("Quick mode — loading checkpoint metadata only...")
        try:
            ckpt = torch.load(ckpt_path, map_location=cfg.training.DEVICE, weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=cfg.training.DEVICE, weights_only=False)
        history = ckpt.get("history", None)
        model_name = "unknown"
        for key in ["model_state_dict"]:
            if key in ckpt:
                _, model_name, _ = get_model_type_from_ckpt(ckpt)
                break

        print("\nGenerating analysis plots...")
        print("  Page 4: Loss decomposition")
        plot_loss_decomposition(history, output_dir / "analysis_04_loss_decomposition.png")

        print("\n" + "=" * 60)
        print(f"  CHECKPOINT ANALYSIS SUMMARY (QUICK MODE)")
        print(f"  Model: {model_name}")
        print("=" * 60)

        ckpt_epoch = ckpt.get("epoch", "?")
        ckpt_val_acc = ckpt.get("val_accuracy", "?")
        if isinstance(ckpt_val_acc, float):
            ckpt_val_acc = f"{ckpt_val_acc:.3f}"
        print(f"\n  Checkpoint epoch: {ckpt_epoch}")
        print(f"  Checkpoint val accuracy: {ckpt_val_acc}")

        if history and len(history.get("val_accuracy", [])) > 0:
            best_epoch = np.argmax(history["val_accuracy"]) + 1
            final_epoch = len(history["val_accuracy"])
            print(f"  Best epoch (from history): {best_epoch}/{final_epoch} "
                  f"(acc: {max(history['val_accuracy']):.3f})")

            print(f"\n  Score trends over training:")
            print(f"    Final pos score: {history['val_pos_score'][-1]:.3f}")
            print(f"    Final neg score: {history['val_neg_score'][-1]:.3f}")
            print(f"    Final hard neg score: {history['val_hard_neg_score'][-1]:.3f}")
            print(f"    Final margin (pos-neg): {history['val_pos_score'][-1] - history['val_neg_score'][-1]:.3f}")

        print("\n" + "=" * 60)
        print(f"\nAll analysis plots saved to: {output_dir}")
        print("Files: analysis_04_loss_decomposition.png")
        print("\nFor full analysis (score histograms, difficulty breakdown, failure cases),")
        print("run without --quick: python analyze_checkpoint.py --checkpoint <path>")
        return

    # Full mode: load model, dataset, collect predictions, run all analyses
    model, val_dataset, val_loader, model_type, model_name, ckpt = \
        load_data_and_model(ckpt_path, cfg)
    history = ckpt.get("history", None)

    print("Collecting predictions on validation set...")
    val_preds = collect_predictions(model, val_loader, cfg.training.DEVICE, model_type)

    print("\nGenerating analysis plots...")

    print("  Page 1: Score histograms")
    plot_score_histograms(val_preds, output_dir / "analysis_01_score_histograms.png")

    print("  Page 2: Difficulty breakdown")
    plot_difficulty_breakdown(val_preds, output_dir / "analysis_02_difficulty_breakdown.png")

    print("  Page 3: Failure cases")
    plot_failure_cases(val_preds, output_dir / "analysis_03_failure_cases.png")

    print("  Page 4: Loss decomposition")
    plot_loss_decomposition(history, output_dir / "analysis_04_loss_decomposition.png")

    print_summary(val_preds, history, model_name, ckpt)

    print(f"\nAll analysis plots saved to: {output_dir}")
    print("Files: analysis_01_score_histograms.png")
    print("       analysis_02_difficulty_breakdown.png")
    print("       analysis_03_failure_cases.png")
    print("       analysis_04_loss_decomposition.png")


if __name__ == "__main__":
    main()

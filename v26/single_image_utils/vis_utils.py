import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from collections import defaultdict


def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


def _sample_title(label, score, threshold=0.5):
    gt_text = "correct" if label == 1.0 else "wrong"
    correct = (score > threshold) == (label == 1.0)
    pred_text = "correct" if correct else "wrong"
    return f"gt={gt_text}\npred={pred_text}\nscore={score:.3f}"


def visualize_score_distribution(all_samples, save_path):
    pos_scores = [s['score'] for s in all_samples if s['label'] == 1.0]
    neg_scores = [s['score'] for s in all_samples if s['label'] == 0.0 and s['category'] == 'negative']
    hard_scores = [s['score'] for s in all_samples if s['label'] == 0.0 and s['category'] == 'hard_negative']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bins = np.linspace(0, 1, 50)
    titles = ['Positives', 'Negatives', 'Hard Negatives']
    data_list = [pos_scores, neg_scores, hard_scores]
    colors = ['#2ca02c', '#d62728', '#ff7f0e']

    for ax, data, title, color in zip(axes, data_list, titles, colors):
        if data:
            ax.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='white')
            ax.axvline(x=np.mean(data), color='black', linestyle='--', linewidth=1)
            ax.text(0.95, 0.95, f"n={len(data)}\nmean={np.mean(data):.3f}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved score distribution to {save_path}")


def visualize_predictions(groups, save_dir, max_groups=20, threshold=0.5,
                          attn_maps_dict=None, geom_maps_dict=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    def group_rank_key(item):
        pair_key, samples = item
        s_arr = np.array([s['score'] for s in samples])
        l_arr = np.array([s['label'] for s in samples])
        pm = l_arr == 1.0
        if pm.sum() >= 1:
            return 0 if l_arr[np.argmax(s_arr)] == 1.0 else 1
        return 2

    sorted_groups = sorted(groups.items(), key=group_rank_key)

    vis_count = 0
    for pair_key, samples in sorted_groups:
        if vis_count >= max_groups:
            break

        scores_arr = np.array([s['score'] for s in samples])
        labels_arr = np.array([s['label'] for s in samples])
        n = len(samples)

        pos_mask = labels_arr == 1.0
        ranking_ok = pos_mask.sum() >= 1 and labels_arr[np.argmax(scores_arr)] == 1.0
        fp_count = ((scores_arr > threshold) & (labels_arr == 0.0)).sum()

        attn_maps = attn_maps_dict.get(pair_key) if attn_maps_dict else None
        geom_maps = geom_maps_dict.get(pair_key) if geom_maps_dict else None

        ncols = 5
        fig, axes = plt.subplots(n, ncols, figsize=(ncols * 4, n * 4.5))
        if n == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(
            f"Group: {pair_key}  |  RankOK={ranking_ok}  FP={int(fp_count)}/{n}",
            fontsize=13, fontweight='bold', y=1.02,
        )

        for i in range(n):
            correct = (scores_arr[i] > threshold) == (labels_arr[i] == 1.0)
            color = 'green' if correct else 'red'
            lw = 3

            img = denormalize_image(samples[i]['rgb'])
            title_rgb = _sample_title(labels_arr[i], scores_arr[i], threshold)

            axes[i, 0].imshow(img)
            axes[i, 0].set_title(title_rgb, color=color, fontweight='bold', fontsize=10)
            axes[i, 0].axis('off')

            img_attn = img.copy()
            if attn_maps is not None:
                from skimage.transform import resize as sk_resize
                import matplotlib.cm as cm
                H, W = img.shape[:2]
                attn_resized = sk_resize(attn_maps[i], (H, W), order=3, preserve_range=True)
                attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
                heatmap = cm.jet(attn_norm)[:, :, :3]
                img_attn = np.clip(0.6 * heatmap + 0.4 * img, 0, 1)
            axes[i, 1].imshow(img_attn)
            axes[i, 1].set_title("Overlay", color=color, fontweight='bold', fontsize=10)
            axes[i, 1].axis('off')

            if geom_maps:
                axes[i, 2].imshow(geom_maps[i][0], cmap='viridis', vmin=0, vmax=1)
            axes[i, 2].set_title("Prox A", color=color, fontweight='bold', fontsize=10)
            axes[i, 2].axis('off')

            if geom_maps:
                axes[i, 3].imshow(geom_maps[i][1], cmap='viridis', vmin=0, vmax=1)
            axes[i, 3].set_title("Prox B", color=color, fontweight='bold', fontsize=10)
            axes[i, 3].axis('off')

            if geom_maps:
                axes[i, 4].imshow(geom_maps[i][2], cmap='hot', vmin=0, vmax=1)
            axes[i, 4].set_title("Contact", color=color, fontweight='bold', fontsize=10)
            axes[i, 4].axis('off')

            for col in range(ncols):
                for spine in axes[i, col].spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(lw)

        plt.tight_layout()
        plt.savefig(save_dir / f"group_{vis_count:04d}_{pair_key.replace('|', '_')}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        vis_count += 1

    print(f"Visualized {vis_count} groups in {save_dir}")


def analyze_failures(groups, save_dir, max_failures=20, threshold=0.5,
                     attn_maps_dict=None, geom_maps_dict=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    failures = []
    for pair_key, samples in groups.items():
        scores_arr = np.array([s['score'] for s in samples])
        labels_arr = np.array([s['label'] for s in samples])
        pos_mask = labels_arr == 1.0

        if pos_mask.sum() >= 1:
            best_idx = np.argmax(scores_arr)
            ranking_fail = labels_arr[best_idx] != 1.0
        else:
            ranking_fail = True

        fp_count = ((scores_arr > threshold) & (labels_arr == 0.0)).sum()

        if ranking_fail or fp_count > 0:
            failures.append((pair_key, samples, ranking_fail, int(fp_count)))

    failures.sort(key=lambda x: (0 if x[2] else 1, -x[3]))

    print(f"Found {len(failures)} groups with errors (showing up to {max_failures})")

    vis_count = 0
    for pair_key, samples, ranking_fail, fp_count in failures[:max_failures]:
        scores_arr = np.array([s['score'] for s in samples])
        labels_arr = np.array([s['label'] for s in samples])
        n = len(samples)

        attn_maps = attn_maps_dict.get(pair_key) if attn_maps_dict else None
        geom_maps = geom_maps_dict.get(pair_key) if geom_maps_dict else None

        ncols = 5
        fig, axes = plt.subplots(n, ncols, figsize=(ncols * 4, n * 4.5))
        if n == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(
            f"[{'RANK FAIL' if ranking_fail else 'FP ONLY'}] {pair_key}  "
            f"|  FP={fp_count}/{n}",
            fontsize=13, fontweight='bold', color='red' if ranking_fail else 'orange',
            y=1.02,
        )

        for i in range(n):
            correct = (scores_arr[i] > threshold) == (labels_arr[i] == 1.0)
            color = 'green' if correct else 'red'
            lw = 3

            img = denormalize_image(samples[i]['rgb'])
            title_rgb = _sample_title(labels_arr[i], scores_arr[i], threshold)

            axes[i, 0].imshow(img)
            axes[i, 0].set_title(title_rgb, color=color, fontweight='bold', fontsize=10)
            axes[i, 0].axis('off')

            img_attn = img.copy()
            if attn_maps is not None:
                from skimage.transform import resize as sk_resize
                import matplotlib.cm as cm
                H, W = img.shape[:2]
                attn_resized = sk_resize(attn_maps[i], (H, W), order=3, preserve_range=True)
                attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
                heatmap = cm.jet(attn_norm)[:, :, :3]
                img_attn = np.clip(0.6 * heatmap + 0.4 * img, 0, 1)
            axes[i, 1].imshow(img_attn)
            axes[i, 1].set_title("Overlay", color=color, fontweight='bold', fontsize=10)
            axes[i, 1].axis('off')

            if geom_maps:
                axes[i, 2].imshow(geom_maps[i][0], cmap='viridis', vmin=0, vmax=1)
            axes[i, 2].set_title("Prox A", color=color, fontweight='bold', fontsize=10)
            axes[i, 2].axis('off')

            if geom_maps:
                axes[i, 3].imshow(geom_maps[i][1], cmap='viridis', vmin=0, vmax=1)
            axes[i, 3].set_title("Prox B", color=color, fontweight='bold', fontsize=10)
            axes[i, 3].axis('off')

            if geom_maps:
                axes[i, 4].imshow(geom_maps[i][2], cmap='hot', vmin=0, vmax=1)
            axes[i, 4].set_title("Contact", color=color, fontweight='bold', fontsize=10)
            axes[i, 4].axis('off')

            for col in range(ncols):
                for spine in axes[i, col].spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(lw)

        plt.tight_layout()
        plt.savefig(save_dir / f"failure_{vis_count:04d}_{pair_key.replace('|', '_')}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        vis_count += 1

    print(f"Saved {vis_count} failure visualizations to {save_dir}")

    print(f"\n{'=' * 60}")
    print(f"{'Failure Analysis Summary':^60}")
    print(f"{'=' * 60}")
    print(f"{'Type':<12} {'Pair Key':<20} {'Samples':<8} {'FP':<6}")
    print(f"{'-' * 60}")
    for pair_key, samples, ranking_fail, fp_count in failures[:40]:
        ftype = "RANK_FAIL" if ranking_fail else "FP_ONLY"
        print(f"{ftype:<12} {pair_key[:20]:<20} {len(samples):<8} {fp_count:<6}")


def inspect_batch_channels(batch, save_path=None, show=False, model=None,
                           device=None, use_geom=False, max_samples=8):
    """
    Visualize channels of the first N samples in a batch.

    Each row: RGB | Proximity A | Proximity B | Contact
    """
    rgb_geom = batch['rgb_geometric']
    labels = batch['labels']
    categories = batch['category']

    scores = None
    if model is not None and device is not None:
        with torch.no_grad():
            rgb_in = batch['rgb'].to(device)
            rg_in = rgb_geom.to(device)
            if use_geom:
                logits = model(rg_in).squeeze()
            else:
                logits = model(rgb_in).squeeze()
            scores = torch.sigmoid(logits).cpu().numpy()

    n = min(max_samples, len(labels))
    channel_names = ['Proximity A', 'Proximity B', 'Contact']
    cmap_options = ['viridis', 'viridis', 'hot']

    fig, axes = plt.subplots(n, 4, figsize=(16, 3 * n))
    fig.suptitle("Batch Channel Inspection", fontsize=14, fontweight='bold', y=1.01)

    if n == 1:
        axes = axes[np.newaxis, :]

    for row in range(n):
        label = labels[row].item()
        title_color = 'green' if label == 1.0 else '#d62728'
        is_pos = label == 1.0

        img = denormalize_image(rgb_geom[row, :3])
        axes[row, 0].imshow(img)
        score_str = f"  score={scores[row]:.3f}" if scores is not None else ""
        axes[row, 0].set_title(
            f"label={int(label)}  [{categories[row]}]{score_str}",
            color=title_color, fontweight='bold' if is_pos else 'normal', fontsize=9,
        )
        axes[row, 0].axis('off')

        for col in range(1, 4):
            ch = col + 2
            data = rgb_geom[row, ch].cpu().numpy()
            axes[row, col].imshow(data, cmap=cmap_options[col - 1], vmin=0, vmax=1)
            axes[row, col].set_title(channel_names[col - 1], fontsize=9)
            axes[row, col].axis('off')

    plt.tight_layout()

    if show:
        plt.show()
    elif save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved batch inspection to {save_path}")
    plt.close()

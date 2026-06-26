"""
Group Size Distribution Analysis

Analyzes the dataset to understand patterns in group sizes (number of images per piece pair).

Group classification by type:
  Type A: has positive + negatives + hard_negatives (standard neighbour pair)
  Type B: no positive, negatives only
  Type C: no positive, hard_negatives only
  Type D: no positive, both negatives AND hard_negatives

Questions we want to answer:
  - Is group size constant across pairs? Or does it vary?
  - Do non-neighbour groups (Type B/C/D) have systematically different sizes?
  - Does group size correlate with puzzle size, art style?
  - Can the model exploit group size as a shortcut?

Usage:
    python analyze_group_size.py
    python analyze_group_size.py --data-root /path/to/dataset
    python analyze_group_size.py --data-root /path/to/dataset --output group_size_analysis
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import Config

SIZE_PATTERN = re.compile(r"__(XS|S|M|L|XL)_")


def extract_style(name):
    parts = name.split("__")
    return parts[4] if len(parts) > 4 else None


def extract_size(name):
    m = SIZE_PATTERN.search(name)
    return m.group(1) if m else None


def scan_groups(data_root, debug=False):
    """
    Scan dataset using PrecomposedAlignmentDataset and build group statistics.

    Uses the actual dataset class so statistics reflect the same filtering
    (min_negatives, max_negatives, hard_negative_ratio, etc.) used during training.

    Returns:
        groups: dict {pair_key: {positive, negative, hard_negative, style, size}}
    """
    from dataset_ranking import PrecomposedAlignmentDataset

    dataset = PrecomposedAlignmentDataset(
        data_root=data_root,
        max_negatives_per_positive=4,
        min_negatives_per_positive=1,
        radius=50,
        threshold=50,
        debug_mode=debug,
    )

    groups = defaultdict(lambda: {
        'positive': 0,
        'negative': 0,
        'hard_negative': 0,
        'style': None,
        'size': None,
    })

    for pair_key, pair_data in dataset.pairs.items():
        n_pos = len(pair_data['positive'])
        n_neg = len(pair_data['negative'])
        n_hn = len(pair_data['hard_negative'])

        if n_pos == 0 and n_neg == 0 and n_hn == 0:
            continue

        d = groups[pair_key]
        d['positive'] = n_pos
        d['negative'] = n_neg
        d['hard_negative'] = n_hn

        # Extract style/size from the first sample's filename
        for cat in ['positive', 'negative', 'hard_negative']:
            samples = pair_data[cat]
            if samples:
                fname = Path(samples[0]['image_path']).name
                style = extract_style(fname)
                size = extract_size(fname)
                if style:
                    d['style'] = style
                if size:
                    d['size'] = size
                break

    return groups


def classify_group(data):
    """Classify group type based on contents."""
    n_pos = data['positive']
    n_neg = data['negative']
    n_hn = data['hard_negative']

    if n_pos > 0:
        return 'A'
    elif n_neg > 0 and n_hn > 0:
        return 'D'
    elif n_neg > 0:
        return 'B'
    elif n_hn > 0:
        return 'C'
    else:
        return '?'


def compute_stats(groups):
    """Compute per-group and aggregate statistics."""
    type_stats = {t: {'count': 0, 'positive': [], 'negative': [], 'hard_negative': [],
                      'total': [], 'styles': Counter(), 'sizes': Counter(),
                      'puzzles': set()} for t in ['A', 'B', 'C', 'D']}

    for key, data in groups.items():
        gtype = classify_group(data)
        stats = type_stats[gtype]

        stats['count'] += 1
        stats['puzzles'].add(key.split('|')[0])
        stats['positive'].append(data['positive'])
        stats['negative'].append(data['negative'])
        stats['hard_negative'].append(data['hard_negative'])
        stats['total'].append(data['positive'] + data['negative'] + data['hard_negative'])

        if data['style']:
            stats['styles'][data['style']] += 1
        if data['size']:
            stats['sizes'][data['size']] += 1

    return type_stats


def print_report(type_stats, groups):
    """Print detailed text report."""
    total_groups = len(groups)
    total_images = sum(g['positive'] + g['negative'] + g['hard_negative'] for g in groups.values())

    print("\n" + "=" * 70)
    print("  GROUP SIZE ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nTotal groups: {total_groups:,}")
    print(f"Total images: {total_images:,}")
    print(f"Avg images/group: {total_images / max(total_groups, 1):.2f}")

    type_names = {
        'A': 'Type A (has positive, std neighbour pair)',
        'B': 'Type B (negatives only, no positive)',
        'C': 'Type C (hard_negatives only, no positive)',
        'D': 'Type D (negatives + hard_negatives, no positive)',
    }

    print(f"\n{'Group type distribution:'}")
    print(f"  {'Type':<5}  {'Groups':>8}  {'%':>7}  {'Puzzles':>8}")
    print(f"  {'-' * 5}  {'-' * 8}  {'-' * 7}  {'-' * 8}")
    for t in ['A', 'B', 'C', 'D']:
        s = type_stats[t]
        pct = 100 * s['count'] / total_groups if total_groups > 0 else 0
        print(f"  {t:<5}  {s['count']:>8,}  {pct:>6.1f}%  {len(s['puzzles']):>8,}")

    print(f"\n{'Per-type image counts:'}")
    print(f"  {'Type':<5}  {'Positive':>10}  {'Negative':>10}  {'Hard_Neg':>10}  {'Total':>10}")
    print(f"  {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for t in ['A', 'B', 'C', 'D']:
        s = type_stats[t]
        tot = sum(s['positive']) + sum(s['negative']) + sum(s['hard_negative'])
        print(f"  {t:<5}  {sum(s['positive']):>10,}  {sum(s['negative']):>10,}  "
              f"{sum(s['hard_negative']):>10,}  {tot:>10,}")

    print(f"\n{'Per-type group size distributions:'}")
    for t in ['A', 'B', 'C', 'D']:
        s = type_stats[t]
        if s['count'] == 0:
            print(f"\n  Type {t}: no groups")
            continue

        totals = np.array(s['total'])
        pos_arr = np.array(s['positive'])
        neg_arr = np.array(s['negative'])
        hn_arr = np.array(s['hard_negative'])

        print(f"\n  Type {t}: {s['count']:,} groups")
        print(f"    Total per group:  mean={totals.mean():.2f}, std={totals.std():.2f}, "
              f"min={totals.min()}, max={totals.max()}, median={np.median(totals):.1f}")
        print(f"    Positive: mean={pos_arr.mean():.2f}, min={pos_arr.min()}, max={pos_arr.max()}")
        print(f"    Negative: mean={neg_arr.mean():.2f}, std={neg_arr.std():.2f}, "
              f"min={neg_arr.min()}, max={neg_arr.max()}, median={np.median(neg_arr):.1f}")
        print(f"    Hard_neg:  mean={hn_arr.mean():.2f}, std={hn_arr.std():.2f}, "
              f"min={hn_arr.min()}, max={hn_arr.max()}, median={np.median(hn_arr):.1f}")

        if t == 'A':
            one_pos = sum(1 for p in pos_arr if p == 1)
            multi_pos = sum(1 for p in pos_arr if p > 1)
            print(f"    Positive count: exactly 1: {one_pos:,}, >1: {multi_pos:,}")
        if t == 'B':
            five_neg = sum(1 for n in neg_arr if n == 5)
            print(f"    Negative count: exactly 5: {five_neg:,} ({100*five_neg/s['count']:.1f}%)")
        if t in ['B', 'C', 'D']:
            size_counter = Counter(totals)
            size_dist = sorted(size_counter.items(), key=lambda x: x[0])
            size_str = ", ".join(f"size={k}: {v:,}" for k, v in size_dist[:8])
            print(f"    Total size distribution: {size_str}")

    print(f"\n{'Style distribution per type:'}")
    for t in ['A', 'B', 'C', 'D']:
        s = type_stats[t]
        if not s['styles']:
            continue
        top = s['styles'].most_common(3)
        total_style = sum(s['styles'].values())
        print(f"  Type {t} ({total_style:,} images): " +
              ", ".join(f"{st}: {ct:,}" for st, ct in top))

    print(f"\n{'Size distribution per type:'}")
    SIZE_ORDER = ['XS', 'S', 'M', 'L', 'XL']
    for t in ['A', 'B', 'C', 'D']:
        s = type_stats[t]
        if not s['sizes']:
            continue
        parts = [f"{sz}: {s['sizes'].get(sz, 0):,}" for sz in SIZE_ORDER if s['sizes'].get(sz, 0) > 0]
        print(f"  Type {t}: " + ", ".join(parts))

    print("\n" + "=" * 70)


def plot_group_size_distributions(type_stats, output_dir):
    """Generate comprehensive plot with 4 figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)

    total_groups = sum(s['count'] for s in type_stats.values())
    type_colors = {'A': '#2ecc71', 'B': '#e74c3c', 'C': '#3498db', 'D': '#f39c12'}
    type_labels = {'A': 'A: has positive', 'B': 'B: negatives only', 
                   'C': 'C: hard_neg only', 'D': 'D: neg + hard_neg'}

    ax1 = fig.add_subplot(gs[0, 0])
    sizes_all = []
    labels_all = []
    colors_all = []
    for t in ['A', 'B', 'C', 'D']:
        totals = type_stats[t]['total']
        if not totals:
            continue
        sizes_all.extend(totals)
        labels_all.extend([t] * len(totals))
        colors_all.extend([type_colors[t]] * len(totals))

    if sizes_all:
        bins = range(0, max(sizes_all) + 5, 5)
        ax1.hist(sizes_all, bins=bins, color='steelblue', edgecolor='white', alpha=0.7)
        ax1.axvline(np.mean(sizes_all), color='red', linestyle='--', linewidth=2,
                    label=f'mean={np.mean(sizes_all):.1f}')
        ax1.axvline(np.median(sizes_all), color='orange', linestyle='--', linewidth=2,
                    label=f'median={np.median(sizes_all):.1f}')
        ax1.set_xlabel("Total Images per Group")
        ax1.set_ylabel("Number of Groups")
        ax1.set_title(f"Group Size Distribution (All {total_groups:,} groups)")
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    type_counts = [type_stats[t]['count'] for t in ['A', 'B', 'C', 'D']]
    type_pcts = [100 * c / total_groups for c in type_counts]
    bars = ax2.bar(['A', 'B', 'C', 'D'], type_counts,
                   color=[type_colors[t] for t in ['A', 'B', 'C', 'D']], edgecolor='white')
    for bar, count, pct in zip(bars, type_counts, type_pcts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_groups * 0.005,
                 f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel("Group Type")
    ax2.set_ylabel("Number of Groups")
    ax2.set_title("Group Count by Type")
    ax2.grid(True, axis='y', alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :])
    type_labels = {'A': 'A: has positive', 'B': 'B: negatives only',
                   'C': 'C: hard_neg only', 'D': 'D: neg + hard_neg'}
    type_names_all = ['A', 'B', 'C', 'D']
    width = 0.25
    x = np.arange(len(type_names_all))

    for i, t in enumerate(type_names_all):
        s = type_stats[t]
        for j, arr_name in enumerate(['positive', 'negative', 'hard_negative']):
            arr = np.array(s[arr_name]) if s[arr_name] else np.array([0])
            mean_val = arr.mean()
            std_val = arr.std() if len(arr) > 1 else 0
            n = s['count']
            offset = (j - 1) * width
            ax3.bar(x[i] + offset, mean_val, width * 0.8,
                    color=type_colors[t], alpha=0.6 + 0.4 * (j == 0),
                    edgecolor='white')

    handles = [plt.Rectangle((0, 0), 1, 1, color=type_colors[t], label=f'Type {t}') for t in type_names_all]
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Type {t}\n(n={type_stats[t]["count"]:,})' for t in type_names_all])
    ax3.set_ylabel("Mean Count per Group")
    ax3.set_title("Mean Image Counts per Group Type")
    ax3.legend(handles=handles, fontsize=8)
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_ylim(bottom=0)

    ax4 = fig.add_subplot(gs[2, 0])
    SIZE_ORDER = ['XS', 'S', 'M', 'L', 'XL']
    size_total_groups = Counter()
    size_type_breakdown = {sz: {'A': 0, 'B': 0, 'C': 0, 'D': 0} for sz in SIZE_ORDER}
    for t in ['A', 'B', 'C', 'D']:
        for sz in SIZE_ORDER:
            size_type_breakdown[sz][t] = type_stats[t]['sizes'].get(sz, 0)
            size_total_groups[sz] += type_stats[t]['sizes'].get(sz, 0)

    present_sizes = [sz for sz in SIZE_ORDER if size_total_groups[sz] > 0]
    n = len(present_sizes)
    width = 0.2
    x = np.arange(n)

    for i, (t, label) in enumerate(type_labels.items()):
        counts = [size_type_breakdown[sz][t] for sz in present_sizes]
        ax4.bar(x + i * width, counts, width, label=label,
                color=type_colors[t], alpha=0.8, edgecolor='white')

    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(present_sizes)
    ax4.set_xlabel("Puzzle Size")
    ax4.set_ylabel("Number of Groups")
    ax4.set_title("Group Type Distribution by Puzzle Size")
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, axis='y', alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    type_group_sizes = {}
    for t in ['A', 'B', 'C', 'D']:
        if type_stats[t]['total']:
            type_group_sizes[t] = np.array(type_stats[t]['total'])

    parts = []
    for t in ['A', 'B', 'C', 'D']:
        if t in type_group_sizes:
            parts.append((t, type_group_sizes[t]))
    
    if parts:
        pos_data = [v for _, v in parts]
        labels = [f"Type {t}\n(n={len(v):,})" for t, v in parts]
        bp = ax5.boxplot(pos_data, labels=labels, patch_artist=True,
                         medianprops={'color': 'black', 'linewidth': 2})
        for patch, t in zip(bp['boxes'], ['A', 'B', 'C', 'D']):
            if t in type_group_sizes:
                patch.set_facecolor(type_colors[t])
                patch.set_alpha(0.7)

    ax5.set_ylabel("Total Images per Group")
    ax5.set_title("Group Size Distribution by Type (box plot)")
    ax5.grid(True, axis='y', alpha=0.3)

    fig.suptitle("Group Size Distribution Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'group_size_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'group_size_overview.png'}")


def plot_detailed_by_type(type_stats, output_dir):
    """Generate one figure per type with detailed breakdowns."""
    type_names = {
        'A': 'Type A - Has Positive (Standard Neighbour Pairs)',
        'B': 'Type B - Negatives Only, No Positive',
        'C': 'Type C - Hard Negatives Only, No Positive',
        'D': 'Type D - Both Negative Types, No Positive',
    }
    type_colors = {'A': '#2ecc71', 'B': '#e74c3c', 'C': '#3498db', 'D': '#f39c12'}

    for t in ['A', 'B', 'C', 'D']:
        s = type_stats[t]
        if s['count'] == 0:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        totals = np.array(s['total'])
        pos_arr = np.array(s['positive'])
        neg_arr = np.array(s['negative'])
        hn_arr = np.array(s['hard_negative'])

        ax = axes[0, 0]
        ax.hist(totals, bins=30, color=type_colors[t], edgecolor='white', alpha=0.8)
        ax.axvline(totals.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'mean={totals.mean():.2f}')
        ax.axvline(np.median(totals), color='orange', linestyle='--', linewidth=2,
                   label=f'median={np.median(totals):.0f}')
        ax.set_xlabel("Total Images per Group")
        ax.set_ylabel("Count")
        ax.set_title(f"Total Group Size Distribution (n={s['count']:,})")
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        if t == 'A':
            ax = axes[0, 1]
            ax.hist(pos_arr, bins=range(0, max(pos_arr.max() + 2, 5)), 
                    color='#2ecc71', edgecolor='white', alpha=0.8)
            ax.set_xlabel("Number of Positives")
            ax.set_ylabel("Count")
            ax.set_title(f"Positive Count per Group (mean={pos_arr.mean():.2f})")
            ax.grid(True, axis='y', alpha=0.3)

            cnt_exact_1 = sum(1 for v in pos_arr if v == 1)
            cnt_multi = sum(1 for v in pos_arr if v > 1)
            ax.text(0.05, 0.95, f"Exactly 1: {cnt_exact_1:,} ({100*cnt_exact_1/s['count']:.1f}%)\n"
                                f"Multiple: {cnt_multi:,} ({100*cnt_multi/s['count']:.1f}%)",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[0, 1].axis('off')

        ax = axes[1, 0]
        neg_unique = sorted(Counter(neg_arr).items(), key=lambda x: x[0])
        if neg_unique:
            sizes_n, counts_n = zip(*neg_unique)
            ax.bar([str(k) for k in sizes_n], counts_n, 
                   color='#e74c3c', edgecolor='white', alpha=0.8)
            ax.set_xlabel("Number of Negatives per Group")
            ax.set_ylabel("Count")
            ax.set_title(f"Negative Count Distribution (mean={neg_arr.mean():.2f})")
            ax.grid(True, axis='y', alpha=0.3)

            cnt_5 = sum(1 for v in neg_arr if v == 5)
            if cnt_5 > 0:
                ax.text(0.05, 0.95, f"Exactly 5: {cnt_5:,} ({100*cnt_5/s['count']:.1f}%)",
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax = axes[1, 1]
        if len(hn_arr) > 0 and hn_arr.max() > 0:
            ax.hist(hn_arr, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
            ax.axvline(hn_arr.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'mean={hn_arr.mean():.2f}')
            ax.axvline(np.median(hn_arr), color='orange', linestyle='--', linewidth=2,
                       label=f'median={np.median(hn_arr):.0f}')
            ax.set_xlabel("Number of Hard Negatives per Group")
            ax.set_ylabel("Count")
            ax.set_title(f"Hard Negative Count Distribution")
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No hard negatives in this type",
                    ha='center', va='center', transform=ax.transAxes)

        fig.suptitle(type_names[t], fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'group_size_type_{t}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / f'group_size_type_{t}.png'}")


def plot_style_size_heatmap(type_stats, output_dir):
    """Plot heatmap of group type vs puzzle size."""
    SIZE_ORDER = ['XS', 'S', 'M', 'L', 'XL']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    type_colors = {'A': '#2ecc71', 'B': '#e74c3c', 'C': '#3498db', 'D': '#f39c12'}

    for idx, t in enumerate(['A', 'B', 'C', 'D']):
        ax = axes[idx]
        s = type_stats[t]
        if s['count'] == 0:
            ax.text(0.5, 0.5, "No groups", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Type {t}")
            continue

        matrix = []
        for sz in SIZE_ORDER:
            row = []
            for cat in ['positive', 'negative', 'hard_negative']:
                if cat == 'positive':
                    arr = np.array(s['positive'])
                    mask = np.array([1 if s['sizes'].get(sz, 0) > 0 else 0] * len(arr))
                elif cat == 'negative':
                    arr = np.array(s['negative'])
                else:
                    arr = np.array(s['hard_negative'])
                mean_val = arr.mean() if len(arr) > 0 else 0
                row.append(mean_val)
            matrix.append(row)

        matrix = np.array(matrix)
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Positive', 'Negative', 'Hard Neg'])
        ax.set_yticks(range(len(SIZE_ORDER)))
        ax.set_yticklabels(SIZE_ORDER)
        ax.set_title(f"Type {t} (n={s['count']:,})")
        plt.colorbar(im, ax=ax, label='Mean count')

        for i in range(len(SIZE_ORDER)):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha='center', va='center',
                        color='black' if matrix[i, j] < matrix.max() * 0.7 else 'white', fontsize=8)

    fig.suptitle("Mean Image Counts by Size and Category per Type", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'group_size_style_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'group_size_style_heatmap.png'}")


def detect_exploitable_patterns(type_stats, groups):
    """Check for patterns the model could exploit."""
    print("\n" + "=" * 70)
    print("  EXPLOITABLE PATTERN CHECK")
    print("=" * 70)

    findings = []

    for t in ['B', 'C', 'D']:
        s = type_stats[t]
        if s['count'] == 0:
            continue
        totals = np.array(s['total'])
        mean_sz = totals.mean()
        std_sz = totals.std()
        median_sz = np.median(totals)

        unique_sizes = sorted(Counter(totals).keys())
        is_constant = len(unique_sizes) <= 2
        is_narrow = std_sz / max(mean_sz, 1) < 0.05

        finding = {
            'type': t,
            'count': s['count'],
            'mean': mean_sz,
            'std': std_sz,
            'median': median_sz,
            'min': totals.min(),
            'max': totals.max(),
            'unique_sizes': len(unique_sizes),
            'is_constant': is_constant or is_narrow,
        }
        findings.append(finding)

        print(f"\n  Type {t}: {s['count']:,} groups, mean size={mean_sz:.2f}, std={std_sz:.2f}")
        if is_constant:
            print(f"    WARNING: Group size is nearly constant ({unique_sizes})!")
            print(f"    The model could learn to detect Type {t} by group size alone!")
        else:
            print(f"    Group size varies (min={totals.min()}, max={totals.max()}, "
                  f"{len(unique_sizes)} unique values)")
            if std_sz < mean_sz * 0.1:
                print(f"    NOTE: Low variance — size is somewhat predictable")

    all_totals_A = np.array(type_stats['A']['total'])
    for f in findings:
        t = f['type']
        other_totals = np.array(type_stats[t]['total'])

        overlap_start = max(all_totals_A.min(), other_totals.min())
        overlap_end = min(all_totals_A.max(), other_totals.max())

        if overlap_end <= overlap_start:
            print(f"\n  Type A vs Type {t}: group sizes have NO overlap!")
            print(f"    Type A: [{all_totals_A.min()}, {all_totals_A.max()}]")
            print(f"    Type {t}: [{other_totals.min()}, {other_totals.max()}]")
            print(f"    CRITICAL: Model can perfectly distinguish by group size!")
        else:
            pct_overlap = 100 * (overlap_end - overlap_start) / max(
                all_totals_A.max() - all_totals_A.min(), other_totals.max() - other_totals.min(), 1)
            if pct_overlap < 20:
                print(f"\n  Type A vs Type {t}: group sizes have minimal overlap ({pct_overlap:.0f}%)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze group size distributions")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to dataset root (default: from config.py)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <data_root>/group_size_analysis)")
    parser.add_argument("--debug", action="store_true",
                        help="Limit to first 1000 images per category")
    args = parser.parse_args()

    cfg = Config()
    data_root = Path(args.data_root) if args.data_root else Path(cfg.data.DATA_ROOT)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = data_root / "group_size_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        print(f"Error: Dataset root not found: {data_root}")
        sys.exit(1)

    print(f"Scanning dataset: {data_root}")
    if args.debug:
        print("Debug mode: limiting to 1000 images per category")

    print("Grouping images by pair...")
    groups = scan_groups(data_root, debug=args.debug)
    print(f"Found {len(groups):,} unique groups")

    print("Computing statistics...")
    type_stats = compute_stats(groups)

    print_report(type_stats, groups)
    detect_exploitable_patterns(type_stats, groups)

    print("\nGenerating plots...")
    plot_group_size_distributions(type_stats, output_dir)
    plot_detailed_by_type(type_stats, output_dir)
    plot_style_size_heatmap(type_stats, output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
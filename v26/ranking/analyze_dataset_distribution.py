"""
Dataset Distribution Analysis

Scans the dataset directory and computes the distribution of:
  - Puzzle sizes: XS, S, M, L, XL (extracted from filename)
  - Art styles: extracted from the 5th __-separated token in filenames

The dataset has the structure:
  data_root/
    positive/images/*.png
    negative/images/*.png
    hard_negative/images/*.png

Usage:
    python analyze_dataset_distribution.py
    python analyze_dataset_distribution.py --data-root /path/to/dataset
    python analyze_dataset_distribution.py --data-root /path/to/dataset --output /path/to/prefix

Output (one per category + total):
    <prefix>_positive.png
    <prefix>_negative.png
    <prefix>_hard_negative.png
    <prefix>_total.png
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from config import Config

SIZE_PATTERN = re.compile(r"__(XS|S|M|L|XL)_(?:vis_piece|piece_)")

CATEGORIES = ["positive", "negative", "hard_negative"]
SIZE_ORDER = ["XS", "S", "M", "L", "XL"]


def extract_style(name):
    parts = name.split("__")
    return parts[4] if len(parts) > 4 else None


def parse_filename(filename):
    name = filename.name

    size_match = SIZE_PATTERN.search(name)
    size = size_match.group(1) if size_match else None

    style = extract_style(name)

    return size, style


def scan_category(data_root, category, debug=False):
    images_dir = Path(data_root) / category / "images"
    if not images_dir.exists():
        return Counter(), Counter(), [], 0

    png_files = list(images_dir.glob("*.png"))
    if debug:
        png_files = png_files[:1000]

    sizes = Counter()
    styles = Counter()
    failures = []
    total = len(png_files)

    for img_path in png_files:
        size, style = parse_filename(img_path)

        if size:
            sizes[size] += 1
        else:
            failures.append(img_path.name)

        if style:
            styles[style] += 1
        else:
            failures.append(img_path.name)

    return sizes, styles, failures, total


def plot_category(name, sizes, styles, total, failures, save_path):
    present_sizes = [s for s in SIZE_ORDER if s in sizes]
    size_counts = [sizes[s] for s in present_sizes]

    sorted_styles = sorted(styles.items(), key=lambda x: x[1], reverse=True)
    style_names = [s[0] for s in sorted_styles]
    style_counts = [s[1] for s in sorted_styles]

    n_styles = len(styles)
    fig_height = max(8, n_styles * 0.35 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_height))

    ax = axes[0]
    if present_sizes:
        cmap = plt.cm.Blues
        colors = [cmap(0.4 + 0.5 * i / max(len(present_sizes) - 1, 1))
                  for i in range(len(present_sizes))]
        bars = ax.barh(present_sizes, size_counts, color=colors, edgecolor="white")
        for bar, count in zip(bars, size_counts):
            pct = 100 * count / total if total > 0 else 0
            ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{count:,} ({pct:.1f}%)", va="center", fontsize=9)
        ax.set_xlim(0, max(size_counts) * 1.18 if size_counts else 1)
    else:
        ax.text(0.5, 0.5, "No size data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Puzzle Size")
    ax.set_title("Size Distribution")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    if style_names:
        cmap = plt.cm.viridis
        colors = [cmap(i / max(n_styles - 1, 1)) for i in range(n_styles)]
        bars = ax.barh(style_names, style_counts, color=colors, edgecolor="white")
        for bar, count in zip(bars, style_counts):
            pct = 100 * count / total if total > 0 else 0
            ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{count:,} ({pct:.1f}%)", va="center", fontsize=8)
        ax.set_xlim(0, max(style_counts) * 1.18)
    else:
        ax.text(0.5, 0.5, "No style data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Art Style")
    ax.set_title(f"Style Distribution ({n_styles} styles)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    title = f"[{name.upper()}] — {total:,} images"
    if failures:
        title += f"  |  {len(failures)} parse failures"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_report(category_data, grand_sizes, grand_styles, grand_failures, grand_total):
    print("\n" + "=" * 60)
    print("  DATASET DISTRIBUTION REPORT")
    print("=" * 60)

    print(f"\nTotal images: {grand_total:,}")
    if grand_failures:
        print(f"Parse failures: {len(grand_failures)} ({100 * len(grand_failures) / grand_total:.1f}%)")
        for f in grand_failures[:5]:
            print(f"  - {f}")
        if len(grand_failures) > 5:
            print(f"  ... and {len(grand_failures) - 5} more")
    else:
        print("Parse failures: 0")

    print(f"\n{'Size distribution (TOTAL):'}")
    print(f"  {'Size':<6}  {'Count':>8}  {'%':>7}")
    print(f"  {'-' * 6}  {'-' * 8}  {'-' * 7}")
    for size in SIZE_ORDER:
        if size in grand_sizes:
            count = grand_sizes[size]
            pct = 100 * count / grand_total if grand_total > 0 else 0
            print(f"  {size:<6}  {count:>8,}  {pct:>6.1f}%")

    print(f"\n{'Style distribution (' + str(len(grand_styles)) + ' unique, TOTAL):'}")
    print(f"  {'Style':<35}  {'Count':>8}  {'%':>7}")
    print(f"  {'-' * 35}  {'-' * 8}  {'-' * 7}")
    sorted_styles = sorted(grand_styles.items(), key=lambda x: x[1], reverse=True)
    for style, count in sorted_styles:
        pct = 100 * count / grand_total if grand_total > 0 else 0
        print(f"  {style:<35}  {count:>8,}  {pct:>6.1f}%")

    print(f"\n{'Per-category breakdown:'}")
    for cat in CATEGORIES:
        data = category_data.get(cat, {})
        sizes = data.get("sizes", Counter())
        styles = data.get("styles", Counter())
        failures = data.get("failures", [])
        total = data.get("total", 0)

        print(f"\n  [{cat}] {total:,} images")
        if failures:
            print(f"    Parse failures: {len(failures)}")

        if sizes:
            top_sizes = [(s, sizes[s]) for s in SIZE_ORDER if s in sizes]
            print(f"    Sizes: " + ", ".join(f"{s}: {c:,}" for s, c in top_sizes))
        if styles:
            top_styles = sorted(styles.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"    Styles (top 5): " + ", ".join(f"{s}: {c:,}" for s, c in top_styles))

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset size and style distributions per category"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Path to dataset root (default: from config.py)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output prefix for plots. "
            "Creates <prefix>_positive.png, _negative.png, _hard_negative.png, _total.png. "
            "Default: <data_root>/dataset_distribution"
        )
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Limit to first 1000 images per category (mirrors DEBUG mode)"
    )
    args = parser.parse_args()

    cfg = Config()
    data_root = Path(args.data_root) if args.data_root else Path(cfg.data.DATA_ROOT)

    if args.output:
        output_prefix = Path(args.output)
    else:
        output_prefix = data_root / "dataset_distribution"

    if not data_root.exists():
        print(f"Error: Dataset root not found: {data_root}")
        sys.exit(1)

    print(f"Scanning dataset: {data_root}")
    if args.debug:
        print("Debug mode: limiting to 1000 images per category")

    category_data = {}
    grand_sizes = Counter()
    grand_styles = Counter()
    grand_failures = []
    grand_total = 0

    for cat in CATEGORIES:
        print(f"\nScanning [{cat}]...", end=" ", flush=True)
        sizes, styles, failures, total = scan_category(data_root, cat, debug=args.debug)
        category_data[cat] = {
            "sizes": sizes,
            "styles": styles,
            "failures": failures,
            "total": total,
        }
        grand_sizes.update(sizes)
        grand_styles.update(styles)
        grand_failures.extend(failures)
        grand_total += total
        print(f"{total:,} images, {len(styles)} styles, {len(failures)} failures")

    if grand_total == 0:
        print(f"Error: No .png files found")
        sys.exit(1)

    print_report(category_data, grand_sizes, grand_styles, grand_failures, grand_total)

    print("\nGenerating plots...")
    for cat in CATEGORIES:
        data = category_data[cat]
        plot_category(
            cat,
            data["sizes"],
            data["styles"],
            data["total"],
            data["failures"],
            f"{output_prefix}_{cat}.png",
        )

    plot_category(
        "TOTAL",
        grand_sizes,
        grand_styles,
        grand_total,
        grand_failures,
        f"{output_prefix}_total.png",
    )


if __name__ == "__main__":
    main()

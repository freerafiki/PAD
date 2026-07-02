"""
precompute_cache.py

Precomputes what SingleImageDataset.__getitem__ currently does on-the-fly
(PIL decode + resize + scipy geometric feature computation) and writes the
result as one small .npz per sample onto fast local storage (e.g. the
internal SSD). SingleImageDataset then reads these directly via its
`ssd_cache_dir` argument, removing slow-disk reads and CPU-bound scipy work
from the training hot loop entirely.

The sample set (which files, stratification, puzzle split, etc.) is
determined by instantiating SingleImageDataset itself with the same
scan-relevant kwargs you use for training — this guarantees the precomputed
cache exactly matches what training will look up, with zero duplicated
scanning/stratification logic.

Usage:
    python precompute_cache.py \
        --data-root /osaka/puzzle_data \
        --ssd-cache-dir /mnt/ssd/puzzle_cache \
        --radius 25 --threshold 25 \
        --num-images 50000 --positive-ratio 0.1 \
        --workers 16

Re-running is safe/resumable: existing .npz files are skipped.
"""

import argparse
import hashlib
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset_single import SingleImageDataset, create_geometric_features


def cache_path_for(ssd_cache_dir, image_path, radius, threshold):
    key = hashlib.md5(image_path.encode()).hexdigest()
    return ssd_cache_dir / f"{key}_r{radius}_t{threshold}.npz"


def process_one(args):
    sample, ssd_cache_dir, radius, threshold = args
    out_path = cache_path_for(ssd_cache_dir, sample['image_path'], radius, threshold)
    if out_path.exists():
        return 'skipped'

    try:
        rgb_image = Image.open(sample['image_path']).convert('RGB')
        rgb_resized = rgb_image.resize((224, 224), Image.BILINEAR)
        rgb_arr = np.array(rgb_resized, dtype=np.uint8)  # (224, 224, 3)

        mask_image = Image.open(sample['mask_path']).convert('L')
        orig_w, orig_h = mask_image.size
        scale = 224.0 / max(orig_w, orig_h)
        mask_resized = mask_image.resize((224, 224), Image.NEAREST)
        mask_array = np.array(mask_resized)
        scaled_radius = max(1, int(round(radius * scale)))
        scaled_threshold = max(1, int(round(threshold * scale)))

        geometric = create_geometric_features(mask_array, scaled_radius, scaled_threshold)
        # Quantize float32 [0,1] -> uint8 [0,255]: 4x smaller on disk, negligible
        # precision loss for a soft proximity/contact map used as a guidance signal.
        geometric_u8 = (np.clip(geometric, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        # Write to a temp file then rename: atomic, so a killed/interrupted run
        # never leaves a corrupt .npz that _load_from_ssd_cache would choke on.
        tmp_path = out_path.with_suffix('.npz.tmp')
        np.savez(tmp_path, rgb=rgb_arr, geometric=geometric_u8)
        tmp_path.rename(out_path)
        return 'ok'
    except Exception as e:
        return f'error: {sample["image_path"]}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='Original (slow) dataset root')
    ap.add_argument('--ssd-cache-dir', required=True, help='Destination on fast local storage')
    ap.add_argument('--radius', type=int, default=50)
    ap.add_argument('--threshold', type=int, default=50)
    ap.add_argument('--num-images', type=int, default=0)
    ap.add_argument('--positive-ratio', type=float, default=0.1)
    ap.add_argument('--puzzle-ids', type=str, default=None,
                     help='Optional comma-separated puzzle id list, matches train/val split')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--scan-cache-dir', type=str, default=None,
                     help='cache_dir for the sample-list scan/stratification pickle (separate from ssd-cache-dir)')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    ssd_cache_dir = Path(args.ssd_cache_dir)
    ssd_cache_dir.mkdir(parents=True, exist_ok=True)

    puzzle_ids = set(args.puzzle_ids.split(',')) if args.puzzle_ids else None

    print("Scanning sample list (reuses SingleImageDataset's own scan/stratification logic)...")
    ds = SingleImageDataset(
        args.data_root,
        use_geometric=True,
        radius=args.radius,
        threshold=args.threshold,
        debug=args.debug,
        limit=args.limit,
        puzzle_ids=puzzle_ids,
        num_images=args.num_images,
        positive_ratio=args.positive_ratio,
        cache_dir=args.scan_cache_dir,
    )
    samples = ds.samples
    print(f"{len(samples)} samples to precompute -> {ssd_cache_dir}")

    tasks = [(s, ssd_cache_dir, args.radius, args.threshold) for s in samples]

    ok = skipped = errors = 0
    with Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)):
            if result == 'ok':
                ok += 1
            elif result == 'skipped':
                skipped += 1
            else:
                errors += 1
                tqdm.write(result)

    print(f"\nDone. ok={ok} skipped(already cached)={skipped} errors={errors}")
    if errors:
        print("Some samples failed - re-run the same command to retry just those "
              "(existing .npz files are skipped, so this is cheap).")


if __name__ == '__main__':
    main()

"""
dataset_single.py - Single-image dataset for alignment scoring.

Each image is an independent sample with a binary label (0 or 1).
No pair-level grouping — evaluation groups by pair_key after scoring.
Supports optional geometric features (proximity + contact channels).
"""

import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_erosion
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import re
import random
import hashlib
import json
import time
import pickle


def split_pieces(rgb, geom):
    raise NotImplementedError("TODO")

class SamePairBatchSampler(Sampler):
    """Yields one batch per pair_key. For small groups (< batch_size), cycles
    indices with repetition to fill the batch. For large groups, selects a
    random subset of batch_size indices each epoch."""
    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(seed)
        groups = {}
        for idx, s in enumerate(dataset.samples):
            pk = s.get('pair_key')
            if pk is None:
                continue
            groups.setdefault(pk, []).append(idx)
        self.groups = list(groups.values())
        self.groups.sort(key=lambda g: min(g))

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.groups)
        batches = []
        for group in self.groups:
            n = len(group)
            if n >= self.batch_size:
                batch = self.rng.sample(group, self.batch_size)
            else:
                k = -(-self.batch_size // n)
                batch = (group * k)[:self.batch_size]
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        return len(self.groups)


def parse_filename(filename, dataset_name='wikiart'):
    basename = Path(filename).stem
    if dataset_name == 'wikiart':
        PREFIX = (
            r'^(\d+)'
            r'__\w+__'
            r'img__(\d+)'
            r'__'
            r'(.+?)'
            r'__pmap__(\d+)'
            r'__'
            r'(\w+)'
        )
        PATTERN_VIS = re.compile(
            PREFIX
            + r'_vis_piece_(\d+)_piece_(\d+)'
            + r'_([\d.]+)_([\d.]+)_([-\d.]+)'
            + r'_(grid|wrong_\d+|gt)$'
        )
        PATTERN_VS = re.compile(
            PREFIX
            + r'_piece_(\d+)_vs_piece_(\d+)'
            + r'_score(\d+)$'
        )

        m = PATTERN_VIS.match(basename)
        if m:
            suffix = m.group(11)
            if suffix == 'gt':
                return None
            label = 'positive' if suffix == 'grid' else 'negative'
            return {
                'puzzle_id': m.group(1), 'img_num': m.group(2),
                'puzzle_name': m.group(3), 'pmap_num': m.group(4),
                'size': m.group(5), 'piece1_id': m.group(6),
                'piece2_id': m.group(7),
                'transform': (float(m.group(8)), float(m.group(9)), float(m.group(10))),
                'suffix': suffix, 'label': label, 'basename': basename,
            }

        m = PATTERN_VS.match(basename)
        if m:
            return {
                'puzzle_id': m.group(1), 'img_num': m.group(2),
                'puzzle_name': m.group(3), 'pmap_num': m.group(4),
                'size': m.group(5), 'piece1_id': m.group(6),
                'piece2_id': m.group(7), 'transform': None,
                'suffix': f'score{m.group(8)}', 'label': 'hard_negative',
                'basename': basename,
            }
        return None
    return None


def get_pair_key(filename):
    parsed = parse_filename(filename)
    if not parsed:
        return None
    pieces = sorted([parsed['piece1_id'], parsed['piece2_id']])
    return f"{parsed['puzzle_id']}|{pieces[0]}|{pieces[1]}"


def classify_file(filename):
    parsed = parse_filename(filename)
    if not parsed:
        return 'ignore'
    suffix = parsed['suffix']
    if suffix == 'grid':
        return 'positive'
    if re.match(r'score\d+', suffix):
        return 'hard_negative'
    if re.match(r'wrong_\d+', suffix):
        return 'negative'
    return 'ignore'


def get_difficulty_score(filename):
    parsed = parse_filename(filename)
    if not parsed:
        return None
    match = re.match(r'score(\d+)', parsed['suffix'])
    return int(match.group(1)) if match else None


def compute_proximity_inclusive(mask, other_mask, radius):
    from scipy.ndimage import distance_transform_edt

    dist_outside = distance_transform_edt(~mask)
    dist_inside = distance_transform_edt(mask)
    signed_distance = np.where(mask, -dist_inside, dist_outside)

    proximity = np.zeros_like(signed_distance, dtype=np.float32)
    proximity[mask] = 1.0
    outside = ~mask
    proximity[outside] = np.clip(1.0 - (signed_distance[outside] / radius), 0, 1)
    overlap = mask & other_mask
    proximity[overlap] = 1.0
    return proximity


def compute_contact_region_edge_based(mask_A, mask_B, threshold):
    from scipy.ndimage import distance_transform_edt, binary_erosion

    edge_A = mask_A & ~binary_erosion(mask_A)
    edge_B = mask_B & ~binary_erosion(mask_B)

    dist_to_edge_A = distance_transform_edt(~edge_A)
    dist_to_edge_B = distance_transform_edt(~edge_B)

    close_to_A = dist_to_edge_A < threshold
    close_to_B = dist_to_edge_B < threshold
    not_inside = (~mask_A) & (~mask_B)
    inside = mask_A & mask_B

    contact_region_outside_pieces = close_to_A & close_to_B & not_inside
    contact_region_anywhere = close_to_A & close_to_B | inside
    contact_region_inside_pieces = contact_region_anywhere ^ contact_region_outside_pieces

    contact_strength = contact_region_inside_pieces.astype(np.float32)
    combined_dist = dist_to_edge_A + dist_to_edge_B
    smooth_strength = np.maximum(0, threshold * 2 - combined_dist) / (threshold * 2)
    contact_strength = smooth_strength * contact_region_inside_pieces

    return np.clip(contact_strength, 0, 1)


def create_geometric_features(mask_array, radius, threshold):
    unique_values = np.unique(mask_array)
    unique_values = unique_values[unique_values > 0]
    if len(unique_values) < 2:
        return np.zeros((3, mask_array.shape[0], mask_array.shape[1]), dtype=np.float32)

    val_A, val_B = unique_values[0], unique_values[1]
    mask_A = mask_array == val_A
    mask_B = mask_array == val_B

    proximity_A = compute_proximity_inclusive(mask_A, mask_B, radius)
    proximity_B = compute_proximity_inclusive(mask_B, mask_A, radius)
    contact_strength = compute_contact_region_edge_based(mask_A, mask_B, threshold)

    return np.stack([proximity_A, proximity_B, contact_strength], axis=0).astype(np.float32)


class SingleImageDataset(Dataset):
    """
    Each image is one sample with a binary label.

    Returns:
        rgb: (3, H, W) normalized RGB tensor
        rgb_geometric: (6, H, W) RGB + geometric features (zeros if use_geometric=False)
        labels: scalar tensor (1.0 for positive, 0.0 otherwise)
        category: str ('positive', 'negative', 'hard_negative')
        pair_key: str for evaluation grouping
    """

    def __init__(self, data_root, use_geometric=False, radius=25, threshold=25,
                 transform=None, debug=False, limit=0, puzzle_ids=None,
                 augment=False, augment_cfg=None,
                 num_images=0, positive_ratio=0.1, cache_dir=None):
        self.data_root = Path(data_root)
        self.use_geometric = use_geometric
        self.radius = radius
        self.threshold = threshold
        self.debug = debug
        self.limit = limit
        self.puzzle_ids = puzzle_ids
        self.augment = augment
        self.augment_cfg = augment_cfg
        self.num_images = num_images
        self.positive_ratio = positive_ratio
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.color_augment = self._build_color_augment()

        self.samples = self._scan_files()
        self._print_statistics()

    def _build_color_augment(self):
        if not self.augment or self.augment_cfg is None:
            return None
        aug_list = []
        cfg = self.augment_cfg
        if cfg.COLOR_JITTER:
            aug_list.append(transforms.ColorJitter(
                cfg.COLOR_JITTER_BRIGHTNESS, cfg.COLOR_JITTER_CONTRAST,
                cfg.COLOR_JITTER_SATURATION, cfg.COLOR_JITTER_HUE,
            ))
        if cfg.GAUSSIAN_BLUR:
            aug_list.append(transforms.RandomApply(
                [transforms.GaussianBlur(cfg.GAUSSIAN_BLUR_KERNEL_SIZE)],
                p=cfg.GAUSSIAN_BLUR_PROB,
            ))
        if cfg.RANDOM_GRAYSCALE:
            aug_list.append(transforms.RandomGrayscale(p=cfg.RANDOM_GRAYSCALE_PROB))
        return transforms.Compose(aug_list) if aug_list else None

    def _cache_params(self):
        return {
            'num_images': self.num_images,
            'positive_ratio': self.positive_ratio,
            'debug': self.debug,
            'limit': self.limit,
            'puzzle_ids': tuple(sorted(self.puzzle_ids)) if self.puzzle_ids else None,
        }

    def _cache_tag(self):
        n_str = f"n{self.num_images}" if self.num_images > 0 else "nAll"
        r_str = f"r{int(self.positive_ratio * 100):03d}"
        p_str = "full" if self.puzzle_ids is None else f"p{len(self.puzzle_ids)}"
        return f"{n_str}_{r_str}_{p_str}"

    def _reload_paths(self, sample_dict):
        s = dict(sample_dict)
        s['image_path'] = str(self.data_root / sample_dict['image_path'])
        s['mask_path'] = str(self.data_root / sample_dict['mask_path'])
        return s

    def _scan_files(self):
        t0 = time.perf_counter()
        cache_hit = False
        cache_file = None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_params = self._cache_params()
            cache_hash = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
            cache_tag = self._cache_tag()
            cache_file = self.cache_dir / f"samples_{cache_tag}_{cache_hash[:8]}.pkl"
            if cache_file.exists():
                try:
                    t_load = time.perf_counter()
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)
                    if cached.get('hash') == cache_hash:
                        reloaded = [self._reload_paths(s) for s in cached['samples']]
                        elapsed = time.perf_counter() - t_load
                        cache_hit = True
                        print(f"  [CACHE HIT] Loaded {len(reloaded)} samples from {cache_file.name} "
                              f"(pickle load: {elapsed:.2f}s)")
                        return reloaded
                    print(f"  [CACHE] Hash mismatch in {cache_file.name}, re-scanning...")
                except Exception as e:
                    print(f"  [CACHE] Read failed ({e}), re-scanning...")

        t_scan = time.perf_counter()
        samples = []
        for category in ['positive', 'negative', 'hard_negative']:
            images_dir = self.data_root / category / 'images'
            masks_dir = self.data_root / category / 'masks'
            if not images_dir.exists():
                continue

            png_files = list(images_dir.glob('*.png'))
            if self.debug:
                random.Random(42).shuffle(png_files)
                png_files = png_files[:200]

            for img_path in png_files:
                file_type = classify_file(img_path.name)
                if file_type == 'ignore':
                    continue

                pair_key = get_pair_key(img_path.name)
                if not pair_key:
                    continue

                mask_path = masks_dir / img_path.name
                if not mask_path.exists():
                    continue

                parsed = parse_filename(img_path.name)
                samples.append({
                    'image_path': str(img_path),
                    'mask_path': str(mask_path),
                    'label': 1.0 if file_type == 'positive' else 0.0,
                    'category': file_type,
                    'pair_key': pair_key,
                    'puzzle_id': parsed['puzzle_id'] if parsed else None,
                    'piece1_id': parsed['piece1_id'] if parsed else None,
                    'piece2_id': parsed['piece2_id'] if parsed else None,
                    'difficulty_score': get_difficulty_score(img_path.name),
                })

        t_filter = time.perf_counter()
        t_scan_dur = t_filter - t_scan
        print(f"  [SCAN] Directory walk + filename parsing: {len(samples)} files in {t_scan_dur:.2f}s")

        if self.puzzle_ids is not None:
            before = len(samples)
            samples = [s for s in samples if s['puzzle_id'] in self.puzzle_ids]
            print(f"  [SCAN] Puzzle filter: {before} -> {len(samples)} samples ({time.perf_counter() - t_filter:.2f}s)")

        t_strat = time.perf_counter()
        if self.num_images > 0 and len(samples) > self.num_images:
            pos = [s for s in samples if s['label'] == 1.0]
            neg = [s for s in samples if s['label'] == 0.0]
            target_pos = min(int(self.num_images * self.positive_ratio), len(pos))
            target_neg = min(self.num_images - target_pos, len(neg))
            if len(pos) < target_pos:
                target_neg = min(self.num_images - len(pos), len(neg))
            if len(neg) < target_neg:
                target_pos = min(self.num_images - len(neg), len(pos))
            random.Random(42).shuffle(pos)
            random.Random(42).shuffle(neg)
            samples = pos[:target_pos] + neg[:target_neg]
        elif self.limit > 0 and len(samples) > self.limit:
            random.Random(42).shuffle(samples)
            samples = samples[:self.limit]
        print(f"  [SCAN] Stratification ({len(samples)} final): {time.perf_counter() - t_strat:.2f}s")

        if self.cache_dir and cache_file is not None:
            t_save = time.perf_counter()
            samples_cache = []
            for s in samples:
                sc = dict(s)
                sc['image_path'] = str(Path(s['image_path']).relative_to(self.data_root))
                sc['mask_path'] = str(Path(s['mask_path']).relative_to(self.data_root))
                samples_cache.append(sc)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'hash': cache_hash, 'samples': samples_cache}, f)
                print(f"  [CACHE MISS] Saved {len(samples)} samples to {cache_file.name} "
                      f"(pickle save: {time.perf_counter() - t_save:.2f}s)")
            except Exception as e:
                print(f"  [CACHE] Write failed ({e})")

        print(f"  [SCAN] Total _scan_files: {time.perf_counter() - t0:.2f}s")
        return samples

    def _print_statistics(self):
        n_pos = sum(1 for s in self.samples if s['label'] == 1.0)
        n_neg = sum(1 for s in self.samples if s['label'] == 0.0)
        unique_pairs = len(set(s['pair_key'] for s in self.samples if s['pair_key']))
        total_str = f" (stratified to {self.num_images})" if self.num_images > 0 else ""
        print(f"\nSingleImageDataset: {len(self.samples)} images{total_str}")
        print(f"  Positives: {n_pos}  Negatives: {n_neg}  (ratio 1:{n_neg / max(n_pos, 1):.1f})")
        print(f"  Unique pairs: {unique_pairs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rgb_image = Image.open(sample['image_path']).convert('RGB')
        rgb_resized = rgb_image.resize((224, 224), Image.BILINEAR)

        if self.use_geometric:
            mask_image = Image.open(sample['mask_path']).convert('L')
            orig_w, orig_h = mask_image.size
            scale = 224.0 / max(orig_w, orig_h)
            mask_resized = mask_image.resize((224, 224), Image.NEAREST)
            mask_array = np.array(mask_resized)
            scaled_radius = max(1, int(round(self.radius * scale)))
            scaled_threshold = max(1, int(round(self.threshold * scale)))

            if self.cache_dir:
                geom_cache_dir = self.cache_dir / 'geom'
                geom_cache_dir.mkdir(parents=True, exist_ok=True)
                mask_key = hashlib.md5(sample['mask_path'].encode()).hexdigest()
                geom_file = geom_cache_dir / f"{mask_key}_r{self.radius}_t{self.threshold}.npy"
                if geom_file.exists():
                    geometric = np.load(geom_file)
                else:
                    geometric = create_geometric_features(mask_array, scaled_radius, scaled_threshold)
                    np.save(geom_file, geometric)
            else:
                geometric = create_geometric_features(mask_array, scaled_radius, scaled_threshold)
        else:
            geometric = np.zeros((3, 224, 224), dtype=np.float32)

        if self.augment and self.augment_cfg is not None and self.augment_cfg.ENABLED:
            rgb_resized, geometric = self._apply_augmentation(rgb_resized, geometric)

        rgb_tensor = self.transform(rgb_resized)
        geometric_tensor = torch.from_numpy(geometric).float()
        rgb_geometric = torch.cat([rgb_tensor, geometric_tensor], dim=0)

        return {
            'rgb': rgb_tensor,
            'rgb_geometric': rgb_geometric,
            'labels': torch.tensor(sample['label'], dtype=torch.float32),
            'category': sample['category'],
            'pair_key': sample['pair_key'],
        }

    def _apply_augmentation(self, rgb_pil, geometric_np):
        cfg = self.augment_cfg

        if cfg.HORIZONTAL_FLIP and np.random.rand() < cfg.HORIZONTAL_FLIP_PROB:
            rgb_pil = TF.hflip(rgb_pil)
            geometric_np = geometric_np[:, :, ::-1].copy()

        if cfg.VERTICAL_FLIP and np.random.rand() < cfg.VERTICAL_FLIP_PROB:
            rgb_pil = TF.vflip(rgb_pil)
            geometric_np = geometric_np[:, ::-1, :].copy()

        if cfg.ROTATION_90 and np.random.rand() < cfg.ROTATION_90_PROB:
            k = np.random.randint(1, 4)
            rgb_pil = TF.rotate(rgb_pil, k * 90, expand=False, fill=0)
            geometric_np = np.rot90(geometric_np, k=k, axes=(1, 2)).copy()

        if self.color_augment is not None:
            rgb_pil = self.color_augment(rgb_pil)

        return rgb_pil, geometric_np

    @classmethod
    def create_puzzle_split(cls, data_root, train_ratio=0.8, seed=42,
                            num_images_val=20000, **kwargs):
        t0 = time.perf_counter()
        puzzles = set()
        for category in ['positive', 'negative', 'hard_negative']:
            img_dir = Path(data_root) / category / 'images'
            for fname in sorted(img_dir.glob('*.png')):
                parsed = parse_filename(fname.name)
                if parsed and parsed.get('puzzle_id'):
                    puzzles.add(parsed['puzzle_id'])
        puzzles = sorted(puzzles)
        print(f"  [PUZZLE GLOB] Found {len(puzzles)} puzzles in {time.perf_counter() - t0:.2f}s")
        random.Random(seed).shuffle(puzzles)
        n_train = max(1, int(len(puzzles) * train_ratio))
        train_puzzles = set(puzzles[:n_train])
        val_puzzles = set(puzzles[n_train:])
        print(f"Split: {len(train_puzzles)} train / {len(val_puzzles)} val puzzles")
        train_ds = cls(data_root, puzzle_ids=train_puzzles, **kwargs)
        val_kwargs = dict(kwargs)
        val_kwargs['num_images'] = num_images_val
        val_ds = cls(data_root, puzzle_ids=val_puzzles, **val_kwargs)
        return train_ds, val_ds

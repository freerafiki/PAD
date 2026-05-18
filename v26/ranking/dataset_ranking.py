"""
dataset_v4.py - Non-Neighbour Support for No-Alignment Training

Extends dataset_v3.py with support for training on fragment pairs that are NOT neighbors.
This helps the model learn to give LOW scores to ALL samples when no valid alignment exists.

Key changes from v3:
- New 'non_neighbour' category for fragments from different walls/rooms/puzzle groups
- Batches can have NO positive sample (has_positive=False)
- All labels are 0.0 for non-neighbour batches
- Model learns to reject invalid fragment pairs
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import re
import json
from torch.utils.data import Sampler

# ============================================================================
# Filename Parsing Utilities
# ============================================================================

def parse_filename(filename, dataset_name:str='wikiart'):
    """
    Parse alignment filename into components.

    Example: 
    positive and negatives have:
    "puzzle_0000029_RP_group_28_vis_8_RPf_00202_7_RPf_00201_493_411_0_gt.png"
    hard_negatives have:
    'puzzle_0000001_RP_group_1_1_RPf_00002_vs_8_RPf_00009_score61.png'

    Returns:
        dict with puzzle_id, piece1_id, piece2_id, transform, suffix
        or None if parsing fails
    """
    basename = Path(filename).stem
    if dataset_name == 'repair':
        # Pattern: puzzle_ID_vis_PIECE1_PIECE2_TRANSFORM_SUFFIX
        pattern = r'(puzzle_\d+_RP_group_\d+)_vis_(\d+_RPf_\d+)_(\d+_RPf_\d+)_(\d+_\d+_\d+)_(.+)$'

        match = re.match(pattern, basename)

        pattern2 = r'(puzzle_\d+_RP_group_\d+)_(\d+_RPf_\d+)_vs_(\d+_RPf_\d+)_(.+)$'

        parsed = None
        if not match:
            match2 = re.match(pattern2, basename)
            if not match2:
                return None
            else:
                parsed = {
                    'puzzle_id': match2.group(1),
                    'piece1_id': match2.group(2),
                    'piece2_id': match2.group(3),
                    'transform': "",
                    'suffix': match2.group(4),
                    'basename': basename
                }
        else:
            parsed = {
                'puzzle_id': match.group(1),
                'piece1_id': match.group(2),
                'piece2_id': match.group(3),
                'transform': match.group(4),
                'suffix': match.group(5),
                'basename': basename
            }
    elif dataset_name  == 'wikiart':
        # ─── Regex breakdown ────────────────────────────────────────────────────
        #
        # Shared prefix (always the same structure):
        #   (\d+)          → ID           e.g. 00085
        #   __P2__         → puzzle type  (literal, or generalize with \w+)
        #   img__(\d+)     → img number   e.g. 36792
        #   __             → separator
        #   (.+?)          → puzzle name  (non-greedy! stops at the next __)
        #   __pmap__(\d+)  → pmap number  e.g. 06139
        #   __             → separator
        #   (\w+)          → size tag     e.g. M, XL, L
        #
        # Then the "pieces info" string branches into two shapes:
        #
        #   POSITIVE / NEGATIVE  (has _vis_ marker):
        #     _vis_piece_(\d+)_piece_(\d+)      → piece1, piece2
        #     _([\d.]+)_([\d.]+)_([-\d.]+)     → tx, ty, rot  (floats + possibly negative)
        #     _(grid|wrong_\d+|gt)$             → class suffix
        #
        #   HARD NEGATIVE (no _vis_, uses _vs_):
        #     _piece_(\d+)_vs_piece_(\d+)       → piece1, piece2
        #     _score(\d+)$                      → score (class suffix)

        PREFIX = (
            r'^(\d+)'           # group 1: ID
            r'__\w+__'          # puzzle type (P2, etc.) — captured but often not needed
            r'img__(\d+)'       # group 2: img number
            r'__'
            r'(.+?)'            # group 3: puzzle name (non-greedy)
            r'__pmap__(\d+)'    # group 4: pmap number
            r'__'
            r'(\w+)'            # group 5: size tag
        )

        PATTERN_VIS = re.compile(
            PREFIX
            + r'_vis_piece_(\d+)_piece_(\d+)'      # groups 6, 7: piece ids
            + r'_([\d.]+)_([\d.]+)_([-\d.]+)'     # groups 8, 9, 10: tx, ty, rotation
            + r'_(grid|wrong_\d+|gt)$'            # group 11: class suffix
        )

        PATTERN_VS = re.compile(
            PREFIX
            + r'_piece_(\d+)_vs_piece_(\d+)'      # groups 6, 7: piece ids
            + r'_score(\d+)$'                     # group 8: score value
        )

        # Try positive / negative pattern first
        m = PATTERN_VIS.match(basename)
        if m:
            suffix = m.group(11)

            if suffix == 'gt':
                return None  # skip gt samples

            label = 'positive' if suffix == 'grid' else 'negative'

            return {
                'puzzle_id':    m.group(1),
                'img_num':      m.group(2),
                'puzzle_name':  m.group(3),
                'pmap_num':     m.group(4),
                'size':         m.group(5),
                'piece1_id':    m.group(6),
                'piece2_id':    m.group(7),
                'transform':    (float(m.group(8)), float(m.group(9)), float(m.group(10))),
                'suffix':       suffix,
                'label':        label,
                'basename':     basename,
            }

        # Try hard negative pattern
        m = PATTERN_VS.match(basename)
        if m:
            return {
                'puzzle_id':    m.group(1),
                'img_num':      m.group(2),
                'puzzle_name':  m.group(3),
                'pmap_num':     m.group(4),
                'size':         m.group(5),
                'piece1_id':    m.group(6),
                'piece2_id':    m.group(7),
                'transform':    None,          # hard negatives have no transform
                'suffix':       f'score{m.group(8)}',
                'label':        'hard_negative',
                'basename':     basename,
            }

        return None  # unrecognized format

    else:
        # TODO: parse only suffix?
        print(f"Did not find any `parse_filename` implementation for dataset `{dataset_name}`. Please check the name or add implementation!")
        print("TODO")
    return parsed


def get_pair_key(filename):
    """
    Get unique key identifying a piece pair.

    Returns: "puzzle_id|piece1|piece2" (pieces sorted for consistency)
    """
    parsed = parse_filename(filename)

    if not parsed:
        return None

    # Sort pieces so order doesn't matter
    pieces = sorted([parsed['piece1_id'], parsed['piece2_id']])

    return f"{parsed['puzzle_id']}|{pieces[0]}|{pieces[1]}"


def classify_file(filename, is_non_neighbour_folder=False):
    """
    Determine file category: 'positive', 'negative', 'hard_negative', 'non_neighbour', or 'ignore'
    
    Args:
        filename: The filename to classify
        is_non_neighbour_folder: If True, file is from non_neighbour folder
    
    Returns:
        Category string
    """
    # If from non_neighbour folder, always classify as non_neighbour
    # if is_non_neighbour_folder:
    #     return 'non_neighbour'
    
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
    
    # # Check for non_neighbour suffix pattern (e.g., "non_neighbour" or "non_neighbor")
    # if 'non_neighbour' in suffix or 'non_neighbor' in suffix:
    #     return 'non_neighbour'

    # Ignore grid visualizations and unknown types
    return 'ignore'


def get_difficulty_score(filename):
    """
    Extract difficulty score from hard negative filenames.
    Higher score = harder negative.

    Returns: int or None
    """
    parsed = parse_filename(filename)
    if not parsed:
        return None

    match = re.match(r'score(\d+)', parsed['suffix'])
    if match:
        return int(match.group(1))

    return None

# ============================================================================
# Custom Shuffler
# ============================================================================

class ShuffledBatchSampler(Sampler):
    """
    Sampler that shuffles pairs, but keeps samples within each pair together.

    Each 'batch' is one pair (multiple samples), and we shuffle the order of pairs.
    """

    def __init__(self, dataset, shuffle=True, seed=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed

        # Number of pairs
        self.num_pairs = len(dataset)

    def __iter__(self):
        if self.shuffle:
            # Shuffle pair indices
            if self.seed is not None:
                np.random.seed(self.seed)
            indices = np.random.permutation(self.num_pairs).tolist()
        else:
            indices = list(range(self.num_pairs))

        return iter(indices)

    def __len__(self):
        return self.num_pairs


# ============================================================================
# Dataset Class
# ============================================================================

class PrecomposedAlignmentDataset(Dataset):
    """
    Dataset for alignment scoring, grouped by piece pairs.

    Each item returns all alignments for one piece pair:
    - 1 positive (ground truth)
    - N hard negatives (high shape-matching score)
    - M easy negatives (random wrong alignments)
    
    v4 additions:
    - Support for non-neighbour pairs (fragments that don't belong together)
    - Batches can have NO positive sample (has_positive=False)
    - All labels are 0.0 for non-neighbour batches
    """

    def __init__(self,
                 data_root,
                 max_negatives_per_positive=4,
                 hard_negative_ratio=0.5,
                 radius=20,
                 threshold=15,
                 transform=None,
                 debug_mode=False,
                 augment=False,
                 augment_cfg=None):
        """
        Args:
            data_root: Path to data directory with positive/negative/hard_negative/non_neighbour folders
            negatives_per_positive: How many negatives to sample per positive
            hard_negative_ratio: Fraction of negatives that should be hard
            radius: Radius for proximity feature computation
            threshold: Threshold for contact region computation
            transform: Optional torchvision transforms for RGB images
            include_non_neighbours: If True, include non-neighbour pairs in training
            non_neighbour_ratio: Fraction of batches that should be non-neighbour (0.0-1.0)
            debug_mode: If True, limit dataset to 1000 images per category for fast testing
            augment: If True, apply data augmentation (requires augment_cfg)
            augment_cfg: AugmentationConfig object with transform parameters
        """
        self.data_root = Path(data_root)
        self.max_negatives_per_positive = max_negatives_per_positive
        self.hard_negative_ratio = hard_negative_ratio
        self.radius = radius
        self.threshold = threshold
        self.debug_mode = debug_mode
        self.augment = augment
        self.augment_cfg = augment_cfg

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Pre-build color augmentation pipeline (RGB-only transforms)
        self.color_augment = None
        if self.augment and self.augment_cfg is not None:
            aug_list = []
            cfg = self.augment_cfg
            if cfg.COLOR_JITTER:
                aug_list.append(transforms.ColorJitter(
                    cfg.COLOR_JITTER_BRIGHTNESS,
                    cfg.COLOR_JITTER_CONTRAST,
                    cfg.COLOR_JITTER_SATURATION,
                    cfg.COLOR_JITTER_HUE,
                ))
            if cfg.GAUSSIAN_BLUR:
                aug_list.append(transforms.RandomApply(
                    [transforms.GaussianBlur(cfg.GAUSSIAN_BLUR_KERNEL_SIZE)],
                    p=cfg.GAUSSIAN_BLUR_PROB,
                ))
            if cfg.RANDOM_GRAYSCALE:
                aug_list.append(transforms.RandomGrayscale(p=cfg.RANDOM_GRAYSCALE_PROB))
            self.color_augment = transforms.Compose(aug_list) if aug_list else None

        # Group files by piece pairs
        print("Loading and grouping files by piece pairs...")
        # Pre-compute which indices are non-neighbour
        self._non_neighbour_indices = set() # must be before `_group_by_pairs()`!
        self.pairs = self._group_by_pairs()
        
        # Store non-neighbour pairs separately
        # self.non_neighbour_pairs = self.pairs.pop('non_neighbour', {})

        # Statistics
        self._print_statistics()

        # Convert to list for indexing
        self.pair_keys = list(self.pairs.keys())
        # self.non_neighbour_keys = list(self.non_neighbour_pairs.keys())
        

        # if self.include_non_neighbours and self.non_neighbour_keys:
        #     # Calculate how many non-neighbour batches to include
        #     total_standard = len(self.pair_keys)
        #     n_non_neighbour = int(total_standard * self.non_neighbour_ratio)
        #     n_non_neighbour = min(n_non_neighbour, len(self.non_neighbour_keys))
            
        #     # Create mapping for non-neighbour indices
        #     # We'll use a probabilistic approach in __getitem__
        #     pass

    def _group_by_pairs(self):
        """
        Scan all files and group by piece pair.

        Returns:
            dict: {pair_key: {'positive': [...], 'negative': [...], 'hard_negative': [...], 'non_neighbour': [...]}}
        """
        from itertools import islice

        pairs = {}
        # non_neighbour_pairs = {}

        # Process standard categories
        for category in ['positive', 'negative', 'hard_negative']:
            images_dir = self.data_root / category / 'images'
            masks_dir = self.data_root / category / 'masks'

            if not images_dir.exists():
                print(f"Warning: {images_dir} does not exist")
                continue
            
            for img_path in (islice(images_dir.glob('*.png'), 1000) if self.debug_mode else images_dir.glob('*.png')):
            # for img_path in images_dir.glob('*.png'):
                # Classify file
                file_type = classify_file(img_path.name)
                if file_type == 'ignore':
                    continue

                # Get pair identifier
                pair_key = get_pair_key(img_path.name)
                if not pair_key:
                    print(f"Warning: Could not parse {img_path.name}")
                    continue

                # Check mask exists
                mask_path = masks_dir / img_path.name
                if not mask_path.exists():
                    print(f"Warning: No mask for {img_path.name}")
                    continue

                # Initialize pair entry
                if pair_key not in pairs:
                    pairs[pair_key] = {
                        'positive': [],
                        'negative': [],
                        'hard_negative': []
                    }

                # Parse metadata
                parsed = parse_filename(img_path.name)
                score = get_difficulty_score(img_path.name)

                # Add to appropriate category
                pairs[pair_key][file_type].append({
                    'image_path': str(img_path),
                    'mask_path': str(mask_path),
                    'pair_key': pair_key,
                    'puzzle_id': parsed['puzzle_id'],
                    'piece1_id': parsed['piece1_id'],
                    'piece2_id': parsed['piece2_id'],
                    'difficulty_score': score,
                    'category': file_type,
                    'label': 1.0 if file_type == 'positive' else 0.0
                })
        
        # # Process non-neighbour category
        # non_neighbour_dir = self.data_root / 'non_neighbour'
        # if non_neighbour_dir.exists():
        #     images_dir = non_neighbour_dir / 'images'
        #     masks_dir = non_neighbour_dir / 'masks'
            
        #     if images_dir.exists():
        #         for img_path in images_dir.glob('*.png'):
        #             # Classify as non_neighbour
        #             file_type = classify_file(img_path.name, is_non_neighbour_folder=True)
                    
        #             # Get pair identifier (may be from different puzzles)
        #             pair_key = get_pair_key(img_path.name)
        #             if not pair_key:
        #                 # For non-neighbours, create a unique key from filename
        #                 pair_key = f"non_neighbour_{img_path.stem}"
                    
        #             # Check mask exists
        #             mask_path = masks_dir / img_path.name
        #             if not mask_path.exists():
        #                 print(f"Warning: No mask for {img_path.name}")
        #                 continue
                    
        #             # Parse metadata (may fail for non-standard names)
        #             parsed = parse_filename(img_path.name)
                    
        #             # Initialize non-neighbour pair entry
        #             if pair_key not in non_neighbour_pairs:
        #                 non_neighbour_pairs[pair_key] = {
        #                     'samples': []
        #                 }
                    
        #             # Add sample
        #             non_neighbour_pairs[pair_key]['samples'].append({
        #                 'image_path': str(img_path),
        #                 'mask_path': str(mask_path),
        #                 'pair_key': pair_key,
        #                 'puzzle_id': parsed['puzzle_id'] if parsed else 'unknown',
        #                 'piece1_id': parsed['piece1_id'] if parsed else 'unknown',
        #                 'piece2_id': parsed['piece2_id'] if parsed else 'unknown',
        #                 'difficulty_score': None,
        #                 'category': 'non_neighbour',
        #                 'label': 0.0  # All non-neighbour samples have label 0
        #             })
        #     else:
        #         print(f"Warning: {images_dir} does not exist")
        # else:
        #     print(f"Info: No non_neighbour directory found at {non_neighbour_dir}")

        # Sort hard negatives by difficulty score (highest = hardest)
        self.neighbour_pairs = 0
        self.non_neighbour_pairs = 0
        for j, pair_data in enumerate(pairs.values()):
            if pair_data['hard_negative']:
                pair_data['hard_negative'].sort(
                    key=lambda x: x.get('difficulty_score', 0),
                    reverse=True
                )
            if len(pair_data['positive']) == 0:
                pair_data['non_neighbours'] = True
                self.non_neighbour_pairs += 1
                self._non_neighbour_indices.add(j)

            else:
                pair_data['non_neighbours'] = False
                self.neighbour_pairs += 1


        # Filter: only keep pairs with at least 1 positive
        # pairs = {k: v for k, v in pairs.items()} # if len(v['positive']) > 0}
        
        # Store non_neighbour pairs in main dict for unified access
        # pairs['non_neighbour'] = non_neighbour_pairs

        return pairs

    def _print_statistics(self):
        """Print dataset statistics."""
        # Standard pairs statistics
        total_pos = sum(len(p['positive']) for p in self.pairs.values() if 'positive' in p)
        total_neg = sum(len(p['negative']) for p in self.pairs.values() if 'negative' in p)
        total_hard = sum(len(p['hard_negative']) for p in self.pairs.values() if 'hard_negative' in p)

        print("\nDataset Statistics:")
        tot_samples = len(self.pairs)
        print(f"  Unique piece pairs: {tot_samples}")
        print(f"    Neighbour pairs: {(self.neighbour_pairs)}")
        print(f"    Non Neighbour pairs: {(self.non_neighbour_pairs)}")
        print(f"    Neigh / Non-Neigh split: {(self.neighbour_pairs / tot_samples):.2%} / {(self.non_neighbour_pairs / tot_samples):.2%}")
        print(f"    Non-Neighbour percentage: {(self.non_neighbour_pairs / tot_samples):.2%}")
        print(f"  Total positive samples: {total_pos}")
        print(f"  Total negative samples: {total_neg}")
        print(f"  Total hard negative samples: {total_hard}")
        print(f"    Avg negatives per pair: {(total_neg + total_hard) / max(len(self.pairs), 1):.2f}")


        # Check pairs with insufficient negatives
        insufficient = sum(1 for p in self.pairs.values()
                          if 'negative' in p and 'hard_negative' in p and
                          len(p['negative']) + len(p['hard_negative']) < 5)
        if insufficient > 0:
            print(f"  Warning: {insufficient} pairs have fewer than {5} negatives")
            print(f"    (will use sampling with replacement for these)")

    def __len__(self):
        """
        Return dataset length.
        
        When include_non_neighbours is True, the length is adjusted to account
        for non-neighbour batches.
        """
        base_len = len(self.pair_keys)
        return base_len

    def __getitem__(self, idx):
        """
        Taken from _v3
        Return samples for one piece pair.

        Returns:
            dict with:
                - rgb: (N, 3, H, W) RGB images
                - rgb_geometric: (N, 6, H, W) RGB + geometric features
                - labels: (N,) 1.0 for positive, 0.0 for negative
                - difficulties: List[str] category names
                - positions: List[int] original positions before shuffle
                - pair_key: str identifier for this pair
        Group size varies based on available negatives.
        """
        pair_key = self.pair_keys[idx]
        pair_data = self.pairs[pair_key]
        
        # Get positive (exactly 1)
        positive_pair = False
        if len(pair_data['positive']) > 0:
            # breakpoint()
            # raise ValueError(f"No positive for pair {pair_key}")
            pos_sample = pair_data['positive'][0]
            positive_pair = True
        else:
            pos_sample = []
            positive_pair = False
        
        # Get ALL available negatives (or up to max)
        available_hard = pair_data['hard_negative']
        available_easy = pair_data['negative']
        
        selected_negatives = []
        
        if self.max_negatives_per_positive is None:
            # Use ALL available negatives
            selected_negatives.extend(available_hard)
            selected_negatives.extend(available_easy)
        else:
            # Sample up to max, respecting ratio
            n_hard_target = int(self.max_negatives_per_positive * self.hard_negative_ratio)
            n_easy_target = self.max_negatives_per_positive - n_hard_target
            
            if positive_pair == False:
                n_easy_target += 1

            # Sample hard negatives
            if available_hard:
                n_hard = min(n_hard_target, len(available_hard))
                selected_negatives.extend(available_hard[:n_hard])  # Take top N hardest
            
            # Sample easy negatives
            if available_easy:
                n_easy_actual = min(n_easy_target, len(available_easy))
                if len(available_easy) <= n_easy_actual:
                    selected_negatives.extend(available_easy)
                else:
                    easy_indices = np.random.choice(
                        len(available_easy),
                        size=n_easy_actual,
                        replace=False
                    )
                    selected_negatives.extend([available_easy[i] for i in easy_indices])
        
        # Combine and shuffle
        if positive_pair == True:
            all_samples = [pos_sample] + selected_negatives
        else:
            all_samples = selected_negatives

        shuffle_indices = np.random.permutation(len(all_samples))
        all_samples = [all_samples[i] for i in shuffle_indices]
        original_positions = shuffle_indices.tolist()
        
        # Process samples (same as before)
        batch = {
            'rgb': [],
            'rgb_geometric': [],
            'labels': [],
            'difficulties': [],
            'positions': []
        }
        
        for sample, pos in zip(all_samples, original_positions):
            rgb, rgb_geom = self._process_sample(sample)
            
            batch['rgb'].append(rgb)
            batch['rgb_geometric'].append(rgb_geom)
            batch['labels'].append(sample['label'])
            batch['difficulties'].append(sample['category'])
            batch['positions'].append(pos)
        
        result = {
            'rgb': torch.stack(batch['rgb']),
            'rgb_geometric': torch.stack(batch['rgb_geometric']),
            'labels': torch.tensor(batch['labels'], dtype=torch.float32),
            'difficulties': batch['difficulties'],
            'positions': batch['positions'],
            'pair_key': pair_key
        }

        return result 

    ############################################
    ############################################
    ############################################
    # This was written to include the non-neighbour case
    # in a different way, but not needed
    # def __getitem__(self, idx):
    #     """
    #     Return samples for one piece pair.

    #     Returns:
    #         dict with:
    #             - rgb: (N, 3, H, W) RGB images
    #             - rgb_geometric: (N, 6, H, W) RGB + geometric features
    #             - labels: (N,) 1.0 for positive, 0.0 for negative
    #             - difficulties: List[str] category names
    #             - positions: List[int] original positions before shuffle
    #             - pair_key: str identifier for this pair
    #             - has_positive: True/False (False for non-neighbour batches)
    #             - batch_type: 'standard'/'non_neighbour'
    #     """
    #     # Determine if this is a non-neighbour batch
    #     is_non_neighbour_batch = False
    #     standard_len = len(self.pair_keys)
        
    #     if self.include_non_neighbours and self.non_neighbour_keys:
    #         # Check if idx is in the non-neighbour range
    #         if idx >= standard_len:
    #             is_non_neighbour_batch = True
    #             # Map to non-neighbour index
    #             non_neighbour_idx = idx - standard_len
    #         else:
    #             # Probabilistic approach: some standard batches become non-neighbour
    #             if np.random.random() < self.non_neighbour_ratio:
    #                 is_non_neighbour_batch = True
    #                 non_neighbour_idx = np.random.randint(0, len(self.non_neighbour_keys))
        
    #     if is_non_neighbour_batch:
    #         return self._get_non_neighbour_item(non_neighbour_idx)
    #     else:
    #         return self._get_standard_item(idx)

    # def _get_standard_item(self, idx):
    #     """
    #     Get a standard batch (with positive and negatives).
    #     """
    #     pair_key = self.pair_keys[idx]
    #     pair_data = self.pairs[pair_key]

    #     # Get positive sample (should be exactly 1, take first if multiple)
    #     if len(pair_data['positive']) == 0:
    #         raise ValueError(f"No positive for pair {pair_key}")

    #     pos_sample = pair_data['positive'][0]

    #     # Sample negatives
    #     n_hard_target = int(self.negatives_per_positive * self.hard_negative_ratio)
    #     n_easy_target = self.negatives_per_positive - n_hard_target

    #     selected_negatives = []

    #     # Sample hard negatives (prefer hardest ones)
    #     if pair_data['hard_negative']:
    #         n_hard = min(n_hard_target, len(pair_data['hard_negative']))
    #         # Take top N hardest (already sorted by score descending)
    #         selected_negatives.extend(pair_data['hard_negative'][:n_hard])

    #     # Sample easy negatives (random)
    #     if pair_data['negative']:
    #         # If we couldn't get enough hard negatives, compensate with easy ones
    #         n_easy_actual = self.negatives_per_positive - len(selected_negatives)
    #         n_easy_actual = min(n_easy_actual, len(pair_data['negative']))

    #         if n_easy_actual > 0:
    #             # Sample without replacement if possible
    #             replace = len(pair_data['negative']) < n_easy_actual

    #             if replace:
    #                 # Sample with replacement
    #                 easy_indices = np.random.choice(
    #                     len(pair_data['negative']),
    #                     size=n_easy_actual,
    #                     replace=True
    #                 )
    #             else:
    #                 # Sample without replacement
    #                 easy_indices = np.random.choice(
    #                     len(pair_data['negative']),
    #                     size=n_easy_actual,
    #                     replace=False
    #                 )

    #             selected_negatives.extend([pair_data['negative'][i] for i in easy_indices])

    #     # If still not enough negatives, warn and proceed with what we have
    #     if len(selected_negatives) < self.negatives_per_positive:
    #         # Could optionally duplicate some negatives here
    #         # For now, just proceed with what we have
    #         pass

    #     # Combine and shuffle
    #     all_samples = [pos_sample] + selected_negatives
    #     shuffle_indices = np.random.permutation(len(all_samples))
    #     all_samples = [all_samples[i] for i in shuffle_indices]
    #     original_positions = shuffle_indices.tolist()

    #     # Process samples
    #     batch = {
    #         'rgb': [],
    #         'rgb_geometric': [],
    #         'labels': [],
    #         'difficulties': [],
    #         'positions': []
    #     }

    #     for sample, pos in zip(all_samples, original_positions):
    #         rgb, rgb_geom = self._process_sample(sample)

    #         batch['rgb'].append(rgb)
    #         batch['rgb_geometric'].append(rgb_geom)
    #         batch['labels'].append(sample['label'])
    #         batch['difficulties'].append(sample['category'])
    #         batch['positions'].append(pos)

    #     result = {
    #         'rgb': torch.stack(batch['rgb']),
    #         'rgb_geometric': torch.stack(batch['rgb_geometric']),
    #         'labels': torch.tensor(batch['labels'], dtype=torch.float32),
    #         'difficulties': batch['difficulties'],
    #         'positions': batch['positions'],
    #         'pair_key': pair_key,
    #         'has_positive': True,
    #         'batch_type': 'standard'
    #     }
        
    #     return result

    # def _get_non_neighbour_item(self, idx):
    #     """
    #     Get a non-neighbour batch (all samples are negatives, no positive).
        
    #     The model should learn to give low scores to ALL samples in this batch.
    #     """
    #     pair_key = self.non_neighbour_keys[idx]
    #     pair_data = self.non_neighbour_pairs[pair_key]
        
    #     samples = pair_data['samples']
        
    #     # Sample if we have more than needed
    #     if len(samples) > self.negatives_per_positive + 1:
    #         indices = np.random.choice(len(samples), size=self.negatives_per_positive + 1, replace=False)
    #         samples = [samples[i] for i in indices]
    #     elif len(samples) < self.negatives_per_positive + 1:
    #         # If we have fewer samples, sample with replacement
    #         indices = np.random.choice(len(samples), size=self.negatives_per_positive + 1, replace=True)
    #         samples = [samples[i] for i in indices]
        
    #     # Shuffle
    #     shuffle_indices = np.random.permutation(len(samples))
    #     samples = [samples[i] for i in shuffle_indices]
    #     original_positions = shuffle_indices.tolist()
        
    #     # Process samples - ALL have label 0.0
    #     batch = {
    #         'rgb': [],
    #         'rgb_geometric': [],
    #         'labels': [],
    #         'difficulties': [],
    #         'positions': []
    #     }

    #     for sample, pos in zip(samples, original_positions):
    #         rgb, rgb_geom = self._process_sample(sample)

    #         batch['rgb'].append(rgb)
    #         batch['rgb_geometric'].append(rgb_geom)
    #         batch['labels'].append(0.0)  # All non-neighbour samples are negative
    #         batch['difficulties'].append('non_neighbour')
    #         batch['positions'].append(pos)

    #     result = {
    #         'rgb': torch.stack(batch['rgb']),
    #         'rgb_geometric': torch.stack(batch['rgb_geometric']),
    #         'labels': torch.tensor(batch['labels'], dtype=torch.float32),
    #         'difficulties': batch['difficulties'],
    #         'positions': batch['positions'],
    #         'pair_key': pair_key,
    #         'has_positive': False,  # No positive in non-neighbour batches
    #         'batch_type': 'non_neighbour'
    #     }
        
    #     return result
    ############################################
    ############################################
    ############################################

    def _process_sample(self, sample):
        """
        Load and process one alignment sample.

        Returns:
            rgb: (3, H, W) normalized RGB tensor
            rgb_geometric: (6, H, W) RGB + geometric features
        """
        # Load RGB
        rgb_image = Image.open(sample['image_path']).convert('RGB')

        # Load mask
        mask_image = Image.open(sample['mask_path']).convert('L')
        mask_array = np.array(mask_image)

        # Create geometric features BEFORE resizing
        geometric_features = self._create_geometric_features(mask_array)

        # Resize to 224x224
        rgb_resized = rgb_image.resize((224, 224), Image.BILINEAR)
        geometric_resized = self._resize_geometric_features(geometric_features, (224, 224))

        # Apply augmentation to BOTH rgb and geometric after resize
        if self.augment and self.augment_cfg is not None and self.augment_cfg.ENABLED:
            rgb_resized, geometric_resized = self._apply_augmentation(rgb_resized, geometric_resized)

        # Apply transforms to RGB
        rgb_tensor = self.transform(rgb_resized)

        # Convert geometric to tensor
        geometric_tensor = torch.from_numpy(geometric_resized).float()

        # Combine
        rgb_geometric = torch.cat([rgb_tensor, geometric_tensor], dim=0)

        return rgb_tensor, rgb_geometric

    def _apply_augmentation(self, rgb_pil, geometric_np):
        """
        Apply data augmentation consistently to RGB and geometric features.

        Spatial transforms (flip, rotate) are applied to BOTH.
        Color transforms are applied to RGB only.

        Args:
            rgb_pil: PIL Image (224, 224) RGB
            geometric_np: numpy array (3, 224, 224) geometric features

        Returns:
            rgb_pil: Augmented PIL Image
            geometric_np: Augmented numpy array
        """
        cfg = self.augment_cfg

        # === Spatial transforms (applied identically to BOTH) ===

        # Horizontal flip
        if cfg.HORIZONTAL_FLIP and np.random.rand() < cfg.HORIZONTAL_FLIP_PROB:
            rgb_pil = TF.hflip(rgb_pil)
            geometric_np = geometric_np[:, :, ::-1].copy()

        # Vertical flip
        if cfg.VERTICAL_FLIP and np.random.rand() < cfg.VERTICAL_FLIP_PROB:
            rgb_pil = TF.vflip(rgb_pil)
            geometric_np = geometric_np[:, ::-1, :].copy()

        # Rotation by multiples of 90°
        if cfg.ROTATION_90 and np.random.rand() < cfg.ROTATION_90_PROB:
            k = np.random.randint(1, 4)  # 1, 2, or 3 → 90°, 180°, 270°
            rgb_pil = TF.rotate(rgb_pil, k * 90, expand=False, fill=0)
            geometric_np = np.rot90(geometric_np, k=k, axes=(1, 2)).copy()

        # === Color transforms (RGB only) ===
        if self.color_augment is not None:
            rgb_pil = self.color_augment(rgb_pil)

        return rgb_pil, geometric_np

    def _create_geometric_features(self, mask_array):
        """
        Create 3 geometric feature channels.

        Returns:
            geometric: (3, H, W) numpy array
        """
        unique_values = np.unique(mask_array)
        unique_values = unique_values[unique_values > 0]

        if len(unique_values) < 2:
            return np.zeros((3, mask_array.shape[0], mask_array.shape[1]), dtype=np.float32)

        val_A = unique_values[0]
        val_B = unique_values[1]

        mask_A = mask_array == val_A
        mask_B = mask_array == val_B

        # Proximity channels (inclusive of piece interior)
        proximity_A = self._compute_proximity_inclusive(mask_A, mask_B)
        proximity_B = self._compute_proximity_inclusive(mask_B, mask_A)

        # Contact region
        contact_strength = self._compute_contact_region_edge_based(mask_A, mask_B)

        geometric = np.stack([proximity_A, proximity_B, contact_strength], axis=0)

        return geometric.astype(np.float32)

    def _compute_proximity_inclusive(self, mask, other_mask):
        """
        Proximity that includes pixels INSIDE the mask.

        Pixels inside the mask are considered "maximally close" to the piece.
        Pixels outside fade based on distance.

        Args:
            mask: Binary mask of the piece
            other_mask: Binary mask of the other piece (for overlap detection)

        Returns:
            proximity: (H, W) array in [0, 1]
        """
        # Compute signed distance:
        # - Negative inside the mask
        # - Positive outside the mask
        # - Zero at the boundary

        from scipy.ndimage import distance_transform_edt

        radius = self.radius

        # Distance from outside to mask
        dist_outside = distance_transform_edt(~mask)

        # Distance from inside to boundary
        dist_inside = distance_transform_edt(mask)

        # Combine: negative inside, positive outside
        signed_distance = np.where(mask, -dist_inside, dist_outside)

        # Convert to proximity:
        # - Inside mask (negative distance): proximity = 1.0
        # - At boundary (distance = 0): proximity = 1.0
        # - Outside mask: proximity fades over `radius` pixels

        proximity = np.zeros_like(signed_distance, dtype=np.float32)

        # Inside the mask: full proximity
        proximity[mask] = 1.0

        # Outside the mask: fade linearly over `radius` pixels
        outside = ~mask
        proximity[outside] = np.clip(1.0 - (signed_distance[outside] / radius), 0, 1)

        # Handle overlap: pixels in both masks get max proximity
        overlap = mask & other_mask
        proximity[overlap] = 1.0

        return proximity

    def _compute_proximity_inclusive_variant(self, mask, other_mask, radius=15):
        """
        Proximity that includes pixels inside the mask.
        """
        # Inside mask: full proximity
        proximity = np.zeros_like(mask, dtype=np.float32)
        proximity[mask] = 1.0

        # Outside mask: fade over radius
        dist_outside = distance_transform_edt(~mask)
        outside = ~mask
        proximity[outside] = np.clip(1.0 - (dist_outside[outside] / radius), 0, 1)

        # Overlap region: max proximity
        overlap = mask & other_mask
        proximity[overlap] = 1.0

        return proximity

    def _compute_contact_region_edge_based(self, mask_A, mask_B):
        """
        Contact region based on edge-to-edge distance.

        For each pixel:
        - Compute distance to edge of piece A
        - Compute distance to edge of piece B
        - If both are small, it's in the contact region
        """
        from scipy.ndimage import distance_transform_edt, binary_erosion

        threshold = self.threshold

        # Extract edges (boundaries) of each piece
        edge_A = mask_A & ~binary_erosion(mask_A)
        edge_B = mask_B & ~binary_erosion(mask_B)

        # Distance to nearest edge pixel
        dist_to_edge_A = distance_transform_edt(~edge_A)
        dist_to_edge_B = distance_transform_edt(~edge_B)

        # Contact region: close to both edges, but not inside either piece
        close_to_A = dist_to_edge_A < threshold
        close_to_B = dist_to_edge_B < threshold
        not_inside = (~mask_A) & (~mask_B)
        inside = mask_A & mask_B

        contact_region_outside_pieces = close_to_A & close_to_B & not_inside
        contact_region_anywhere = close_to_A & close_to_B | inside
        contact_region_inside_pieces = contact_region_anywhere ^ contact_region_outside_pieces

        # Convert to smooth strength
        contact_strength = contact_region_inside_pieces.astype(np.float32)

        # Optional: add smooth falloff
        combined_dist = dist_to_edge_A + dist_to_edge_B
        smooth_strength = np.maximum(0, threshold*2 - combined_dist) / (threshold*2)
        contact_strength = smooth_strength * contact_region_inside_pieces

        return np.clip(contact_strength, 0, 1)

    @staticmethod
    def _resize_geometric_features(geometric, target_size):
        """Resize geometric features to target size."""
        from PIL import Image

        resized_channels = []
        for i in range(geometric.shape[0]):
            channel = geometric[i]
            channel_pil = Image.fromarray((channel * 255).astype(np.uint8))
            channel_resized = channel_pil.resize((target_size[1], target_size[0]), Image.BILINEAR)
            channel_array = np.array(channel_resized).astype(np.float32) / 255.0
            resized_channels.append(channel_array)

        return np.stack(resized_channels, axis=0)

    def create_split(self, train_puzzles=None, val_puzzles=None, radius=30, threshold=30,
                     augment=None, augment_cfg=None):
        """
        Create a filtered dataset for train or validation.

        Args:
            train_puzzles: Set of puzzle IDs for training
            val_puzzles: Set of puzzle IDs for validation
            augment: Override augmentation (if None, inherits from this dataset)
            augment_cfg: Augmentation config (if None, inherits from this dataset)

        Returns:
            New dataset instance with filtered pairs
        """
        if train_puzzles is None and val_puzzles is None:
            raise ValueError("Must specify either train_puzzles or val_puzzles")

        # Determine which puzzles to keep
        if train_puzzles is not None:
            keep_puzzles = train_puzzles
            split_name = "train"
        else:
            keep_puzzles = val_puzzles
            split_name = "val"

        # Filter pair_keys
        filtered_keys = [
            key for key in self.pair_keys
            if key.split('|')[0] in keep_puzzles
        ]
        
        # Determine augment settings (override vs inherit)
        use_augment = augment if augment is not None else self.augment
        use_augment_cfg = augment_cfg if augment_cfg is not None else self.augment_cfg

        # Pre-build color augmentation pipeline if needed
        color_augment = None
        if use_augment and use_augment_cfg is not None:
            aug_list = []
            cfg = use_augment_cfg
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
            color_augment = transforms.Compose(aug_list) if aug_list else None

        # Create new instance (shallow copy)
        new_dataset = PrecomposedAlignmentDataset.__new__(PrecomposedAlignmentDataset)
        new_dataset.data_root = self.data_root
        new_dataset.max_negatives_per_positive = self.max_negatives_per_positive
        new_dataset.hard_negative_ratio = self.hard_negative_ratio
        new_dataset.transform = self.transform
        new_dataset.pairs = self.pairs  # Share the pairs dict (read-only)
        new_dataset.pair_keys = filtered_keys
        new_dataset.non_neighbour_pairs = self.non_neighbour_pairs
        new_dataset.neighbour_pairs = self.neighbour_pairs
        new_dataset.radius = radius
        new_dataset.threshold = threshold
        new_dataset.non_neighbour_ratio = self.non_neighbour_pairs / len(self.pairs)
        new_dataset.augment = use_augment
        new_dataset.augment_cfg = use_augment_cfg
        new_dataset.color_augment = color_augment
        new_dataset._non_neighbour_indices = self._non_neighbour_indices

        print(f"Created {split_name} split: {len(filtered_keys)} pairs from {len(keep_puzzles)} puzzles")
        print(f"  Non-neighbour pairs: {self.non_neighbour_pairs}")
        print(f"  Neighbour pairs: {self.neighbour_pairs}")
        if new_dataset.augment and new_dataset.augment_cfg is not None and new_dataset.augment_cfg.ENABLED:
            print(f"  Augmentation: ENABLED")

        return new_dataset

    @staticmethod
    def create_puzzle_split(dataset, train_ratio=0.8, seed=42, radius=30, threshold=30,
                            val_ratio=None, test_ratio=None):
        """
        Split dataset by puzzles (no puzzle appears in both train and val).

        Args:
            dataset: The full dataset
            train_ratio: Fraction of puzzles for training
            seed: Random seed for reproducibility
            val_ratio: Fraction for validation (if None, remaining goes to val)
            test_ratio: Fraction for test (optional 3-way split)

        Returns:
            train_dataset, val_dataset  or  train_dataset, val_dataset, test_dataset
        """
        # Get all unique puzzles
        all_puzzles = list(set(key.split('|')[0] for key in dataset.pair_keys))
        all_puzzles.sort()  # For reproducibility

        # Shuffle with seed
        np.random.seed(seed)
        np.random.shuffle(all_puzzles)

        n = len(all_puzzles)
        n_train = int(n * train_ratio)

        print(f"\n=== Puzzle-Based Split ===")
        print(f"Total puzzles: {n}")

        if val_ratio is not None and test_ratio is not None:
            # 3-way split
            n_val = int(n * val_ratio)
            train_puzzles = set(all_puzzles[:n_train])
            val_puzzles = set(all_puzzles[n_train:n_train + n_val])
            test_puzzles = set(all_puzzles[n_train + n_val:])

            print(f"Train puzzles: {len(train_puzzles)} ({train_ratio:.1%})")
            print(f"Val puzzles: {len(val_puzzles)} ({val_ratio:.1%})")
            print(f"Test puzzles: {len(test_puzzles)} ({test_ratio:.1%})")

            train_dataset = dataset.create_split(train_puzzles=train_puzzles, radius=radius, threshold=threshold)
            val_dataset = dataset.create_split(val_puzzles=val_puzzles, radius=radius, threshold=threshold)
            test_dataset = dataset.create_split(val_puzzles=test_puzzles, radius=radius, threshold=threshold)

            return train_dataset, val_dataset, test_dataset
        else:
            # 2-way split (backwards compatible)
            train_puzzles = set(all_puzzles[:n_train])
            val_puzzles = set(all_puzzles[n_train:])

            print(f"Train puzzles: {len(train_puzzles)} ({train_ratio:.1%})")
            print(f"Val puzzles: {len(val_puzzles)} ({1 - train_ratio:.1%})")

            train_dataset = dataset.create_split(train_puzzles=train_puzzles, radius=radius, threshold=threshold)
            val_dataset = dataset.create_split(val_puzzles=val_puzzles, radius=radius, threshold=threshold)

            return train_dataset, val_dataset

    def save_split(self, path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Create a 3-way puzzle split and save puzzle IDs to JSON.

        The saved JSON can be loaded later with from_split_file() for
        reproducible dataset splits across experiments.

        Args:
            path: Path to save the JSON file
            train_ratio: Fraction of puzzles for training
            val_ratio: Fraction of puzzles for validation
            test_ratio: Fraction of puzzles for testing
            seed: Random seed for reproducibility
        """
        all_puzzles = list(set(key.split('|')[0] for key in self.pair_keys))
        all_puzzles.sort()
        np.random.seed(seed)
        np.random.shuffle(all_puzzles)

        n = len(all_puzzles)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        split = {
            'train': all_puzzles[:n_train],
            'val': all_puzzles[n_train:n_train + n_val],
            'test': all_puzzles[n_train + n_val:],
            'params': {
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'seed': seed,
                'data_root': str(self.data_root),
                'max_negatives_per_positive': self.max_negatives_per_positive,
                'hard_negative_ratio': self.hard_negative_ratio,
                'radius': self.radius,
                'threshold': self.threshold,
            }
        }

        with open(path, 'w') as f:
            json.dump(split, f, indent=2)
        print(f"Split saved to {path}  ({len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test)")

    @staticmethod
    def from_split_file(split_path, data_root=None, max_negatives_per_positive=4,
                         hard_negative_ratio=0.5, radius=20, threshold=15,
                         transform=None, debug_mode=False):
        """
        Load a saved split from JSON and create train/val/test datasets.

        Args:
            split_path: Path to the JSON split file (created by save_split)
            data_root: Override data_root (uses the one from split file if None)
            max_negatives_per_positive: Override if needed
            hard_negative_ratio: Override if needed
            radius: Override if needed
            threshold: Override if needed
            transform: Optional transform (uses default if None)
            debug_mode: If True, limit dataset

        Returns:
            train_dataset, val_dataset, test_dataset
        """
        with open(split_path) as f:
            split = json.load(f)

        params = split['params']
        root = Path(data_root or params['data_root'])
        max_neg = max_negatives_per_positive or params['max_negatives_per_positive']
        hn_ratio = hard_negative_ratio or params['hard_negative_ratio']
        rad = radius or params['radius']
        thr = threshold or params['threshold']

        # Create a temporary full dataset to use create_split
        temp_dataset = PrecomposedAlignmentDataset.__new__(PrecomposedAlignmentDataset)
        full_dataset = PrecomposedAlignmentDataset(
            data_root=root,
            max_negatives_per_positive=max_neg,
            hard_negative_ratio=hn_ratio,
            radius=rad,
            threshold=thr,
            transform=transform,
            debug_mode=debug_mode,
        )

        train_dataset = full_dataset.create_split(
            train_puzzles=set(split['train']), radius=rad, threshold=thr
        )
        val_dataset = full_dataset.create_split(
            val_puzzles=set(split['val']), radius=rad, threshold=thr
        )
        test_dataset = full_dataset.create_split(
            val_puzzles=set(split['test']), radius=rad, threshold=thr
        )

        print(f"\nLoaded split from {split_path}")
        print(f"  Train: {len(train_dataset)} pairs  Val: {len(val_dataset)} pairs  Test: {len(test_dataset)} pairs")

        return train_dataset, val_dataset, test_dataset


# ============================================================================
# Collate Function
# ============================================================================

def collate_alignment_samples(batch_list):
    """
    Collate function for DataLoader.
    Flattens variable-sized groups into single batch.
    
    Handles both standard and non-neighbour batches.
    """
    rgb = torch.cat([item['rgb'] for item in batch_list], dim=0)
    rgb_geometric = torch.cat([item['rgb_geometric'] for item in batch_list], dim=0)
    labels = torch.cat([item['labels'] for item in batch_list], dim=0)
    difficulties = [d for item in batch_list for d in item['difficulties']]
    positions = [p for item in batch_list for p in item['positions']]
    pair_keys = [item['pair_key'] for item in batch_list]
    
    # New fields for non-neighbour support
    # has_positive = [item['has_positive'] for item in batch_list]
    # batch_type = [item['batch_type'] for item in batch_list]
    group_sizes = [len(item['labels']) for item in batch_list]

    result = {
        'rgb': rgb,
        'rgb_geometric': rgb_geometric,
        'labels': labels,
        'difficulties': difficulties,
        'positions': positions,
        'pair_keys': pair_keys,
        # 'has_positive': has_positive,
        # 'batch_type': batch_type,
        'group_sizes': group_sizes
    }

    return result
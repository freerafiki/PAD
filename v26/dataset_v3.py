"""
dataset.py - Updated with pair-based grouping
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_erosion
import torchvision.transforms as transforms
import re


# ============================================================================
# Filename Parsing Utilities
# ============================================================================

def parse_filename(filename):
    """
    Parse alignment filename into components.

    Example: "puzzle_0000029_RP_group_28_vis_8_RPf_00202_7_RPf_00201_493_411_0_gt.png"

    Returns:
        dict with puzzle_id, piece1_id, piece2_id, transform, suffix
        or None if parsing fails
    """
    basename = Path(filename).stem

    # Pattern: puzzle_ID_vis_PIECE1_PIECE2_TRANSFORM_SUFFIX
    pattern = r'(puzzle_\d+_RP_group_\d+)_vis_(\d+_RPf_\d+)_(\d+_RPf_\d+)_(\d+_\d+_\d+)_(.+)$'

    match = re.match(pattern, basename)

    if not match:
        return None

    return {
        'puzzle_id': match.group(1),
        'piece1_id': match.group(2),
        'piece2_id': match.group(3),
        'transform': match.group(4),
        'suffix': match.group(5),
        'basename': basename
    }


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


def classify_file(filename):
    """
    Determine file category: 'positive', 'negative', 'hard_negative', or 'ignore'
    """
    parsed = parse_filename(filename)
    if not parsed:
        return 'ignore'

    suffix = parsed['suffix']

    if suffix == 'gt':
        return 'positive'

    if re.match(r'score\d+', suffix):
        return 'hard_negative'

    if re.match(r'wrong_\d+', suffix):
        return 'negative'

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


from torch.utils.data import Sampler

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
    """

    def __init__(self,
                 data_root,
                 negatives_per_positive=4,  # Reduced from 6 since you have less data
                 hard_negative_ratio=0.6,   # 60% hard, 40% easy
                 radius=20,
                 threshold=15,
                 transform=None
                 ):
        """
        Args:
            data_root: Path to data directory with positive/negative/hard_negative folders
            negatives_per_positive: How many negatives to sample per positive
            hard_negative_ratio: Fraction of negatives that should be hard
            transform: Optional torchvision transforms for RGB images
        """
        self.data_root = Path(data_root)
        self.negatives_per_positive = negatives_per_positive
        self.hard_negative_ratio = hard_negative_ratio
        self.radius = radius
        self.threshold = threshold

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Group files by piece pairs
        print("Loading and grouping files by piece pairs...")
        self.pairs = self._group_by_pairs()

        # Statistics
        self._print_statistics()

        # Convert to list for indexing
        self.pair_keys = list(self.pairs.keys())

    def _group_by_pairs(self):
        """
        Scan all files and group by piece pair.

        Returns:
            dict: {pair_key: {'positive': [...], 'negative': [...], 'hard_negative': [...]}}
        """
        pairs = {}

        for category in ['positive', 'negative', 'hard_negative']:
            images_dir = self.data_root / category / 'images'
            masks_dir = self.data_root / category / 'masks'

            if not images_dir.exists():
                print(f"Warning: {images_dir} does not exist")
                continue

            for img_path in images_dir.glob('*.png'):
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

        # Sort hard negatives by difficulty score (highest = hardest)
        for pair_data in pairs.values():
            if pair_data['hard_negative']:
                pair_data['hard_negative'].sort(
                    key=lambda x: x.get('difficulty_score', 0),
                    reverse=True
                )

        # Filter: only keep pairs with at least 1 positive
        pairs = {k: v for k, v in pairs.items() if len(v['positive']) > 0}

        return pairs

    def _print_statistics(self):
        """Print dataset statistics."""
        total_pos = sum(len(p['positive']) for p in self.pairs.values())
        total_neg = sum(len(p['negative']) for p in self.pairs.values())
        total_hard = sum(len(p['hard_negative']) for p in self.pairs.values())

        print(f"\nDataset Statistics:")
        print(f"  Unique piece pairs: {len(self.pairs)}")
        print(f"  Total positive samples: {total_pos}")
        print(f"  Total negative samples: {total_neg}")
        print(f"  Total hard negative samples: {total_hard}")
        print(f"  Avg negatives per pair: {(total_neg + total_hard) / len(self.pairs):.2f}")

        # Check pairs with insufficient negatives
        insufficient = sum(1 for p in self.pairs.values()
                          if len(p['negative']) + len(p['hard_negative']) < self.negatives_per_positive)
        if insufficient > 0:
            print(f"  ⚠ {insufficient} pairs have fewer than {self.negatives_per_positive} negatives")
            print(f"    (will use sampling with replacement for these)")

    def __len__(self):
        return len(self.pair_keys)

    def __getitem__(self, idx):
        """
        Return samples for one piece pair.

        Returns:
            dict with:
                - rgb: (N, 3, H, W) RGB images
                - rgb_geometric: (N, 6, H, W) RGB + geometric features
                - labels: (N,) 1.0 for positive, 0.0 for negative
                - difficulties: List[str] category names
                - positions: List[int] original positions before shuffle
                - pair_key: str identifier for this pair
        """
        pair_key = self.pair_keys[idx]
        pair_data = self.pairs[pair_key]

        # Get positive sample (should be exactly 1, take first if multiple)
        if len(pair_data['positive']) == 0:
            raise ValueError(f"No positive for pair {pair_key}")

        pos_sample = pair_data['positive'][0]

        # Sample negatives
        n_hard_target = int(self.negatives_per_positive * self.hard_negative_ratio)
        n_easy_target = self.negatives_per_positive - n_hard_target

        selected_negatives = []

        # Sample hard negatives (prefer hardest ones)
        if pair_data['hard_negative']:
            n_hard = min(n_hard_target, len(pair_data['hard_negative']))
            # Take top N hardest (already sorted by score descending)
            selected_negatives.extend(pair_data['hard_negative'][:n_hard])

        # Sample easy negatives (random)
        if pair_data['negative']:
            # If we couldn't get enough hard negatives, compensate with easy ones
            n_easy_actual = self.negatives_per_positive - len(selected_negatives)
            n_easy_actual = min(n_easy_actual, len(pair_data['negative']))

            if n_easy_actual > 0:
                # Sample without replacement if possible
                replace = len(pair_data['negative']) < n_easy_actual

                if replace:
                    # Sample with replacement
                    easy_indices = np.random.choice(
                        len(pair_data['negative']),
                        size=n_easy_actual,
                        replace=True
                    )
                else:
                    # Sample without replacement
                    easy_indices = np.random.choice(
                        len(pair_data['negative']),
                        size=n_easy_actual,
                        replace=False
                    )

                selected_negatives.extend([pair_data['negative'][i] for i in easy_indices])

        # If still not enough negatives, warn and proceed with what we have
        if len(selected_negatives) < self.negatives_per_positive:
            # Could optionally duplicate some negatives here
            # For now, just proceed with what we have
            pass

        # Combine and shuffle
        all_samples = [pos_sample] + selected_negatives
        shuffle_indices = np.random.permutation(len(all_samples))
        all_samples = [all_samples[i] for i in shuffle_indices]
        original_positions = shuffle_indices.tolist()

        # Process samples
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

        # Apply transforms to RGB
        rgb_tensor = self.transform(rgb_resized)

        # Convert geometric to tensor
        geometric_tensor = torch.from_numpy(geometric_resized).float()

        # Combine
        rgb_geometric = torch.cat([rgb_tensor, geometric_tensor], dim=0)

        return rgb_tensor, rgb_geometric

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
            radius: How far outside the mask proximity extends (in pixels)

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
        # import matplotlib.pyplot as plt
        # plt.subplot(321); plt.imshow(close_to_A)
        # plt.subplot(322); plt.imshow(close_to_B)
        # plt.subplot(323); plt.imshow(not_inside)
        # plt.subplot(324); plt.imshow(contact_region_outside_pieces)
        # plt.subplot(325); plt.imshow(contact_region_anywhere)
        # plt.subplot(326); plt.imshow(contact_region_inside_pieces)
        # plt.show()
        # breakpoint()

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

    def create_split(self, train_puzzles=None, val_puzzles=None, radius=30, threshold=30):
        """
        Create a filtered dataset for train or validation.

        Args:
            train_puzzles: Set of puzzle IDs for training
            val_puzzles: Set of puzzle IDs for validation

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

        # Create new instance (shallow copy)
        new_dataset = PrecomposedAlignmentDataset.__new__(PrecomposedAlignmentDataset)
        new_dataset.data_root = self.data_root
        new_dataset.negatives_per_positive = self.negatives_per_positive
        new_dataset.hard_negative_ratio = self.hard_negative_ratio
        new_dataset.transform = self.transform
        new_dataset.pairs = self.pairs  # Share the pairs dict (read-only)
        new_dataset.pair_keys = filtered_keys
        new_dataset.radius = radius
        new_dataset.threshold = threshold

        print(f"Created {split_name} split: {len(filtered_keys)} pairs from {len(keep_puzzles)} puzzles")

        return new_dataset

    @staticmethod
    def create_puzzle_split(dataset, train_ratio=0.8, seed=42, radius=30, threshold=30):
        """
        Split dataset by puzzles (no puzzle appears in both train and val).

        Args:
            dataset: The full dataset
            train_ratio: Fraction of puzzles for training
            seed: Random seed for reproducibility

        Returns:
            train_dataset, val_dataset
        """
        # Get all unique puzzles
        all_puzzles = list(set(key.split('|')[0] for key in dataset.pair_keys))
        all_puzzles.sort()  # For reproducibility

        # Shuffle with seed
        np.random.seed(seed)
        np.random.shuffle(all_puzzles)

        # Split
        split_idx = int(len(all_puzzles) * train_ratio)
        train_puzzles = set(all_puzzles[:split_idx])
        val_puzzles = set(all_puzzles[split_idx:])

        print(f"\n=== Puzzle-Based Split ===")
        print(f"Total puzzles: {len(all_puzzles)}")
        print(f"Train puzzles: {len(train_puzzles)}")
        print(f"Val puzzles: {len(val_puzzles)}")

        # Create split datasets
        train_dataset = dataset.create_split(train_puzzles=train_puzzles, radius = radius, threshold = threshold)
        val_dataset = dataset.create_split(val_puzzles=val_puzzles, radius = radius, threshold = threshold)

        return train_dataset, val_dataset


# ============================================================================
# Collate Function
# ============================================================================

def collate_alignment_samples(batch_list):
    """
    Collate function for DataLoader.
    Flattens variable-sized groups into single batch.
    """
    rgb = torch.cat([item['rgb'] for item in batch_list], dim=0)
    rgb_geometric = torch.cat([item['rgb_geometric'] for item in batch_list], dim=0)
    labels = torch.cat([item['labels'] for item in batch_list], dim=0)
    difficulties = [d for item in batch_list for d in item['difficulties']]
    positions = [p for item in batch_list for p in item['positions']]
    pair_keys = [item['pair_key'] for item in batch_list]

    result = {
        'rgb': rgb,
        'rgb_geometric': rgb_geometric,
        'labels': labels,
        'difficulties': difficulties,
        'positions': positions,
        'pair_keys': pair_keys
    }

    return result

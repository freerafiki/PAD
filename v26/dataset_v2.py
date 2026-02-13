import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_erosion
import torchvision.transforms as transforms

class PrecomposedAlignmentDataset(Dataset):
    """Dataset for alignment scoring with pre-composed images."""

    def __init__(self,
                 data_root,
                 split='train',
                 transform=None,
                 negatives_per_positive=3,
                 radius=20,
                 threshold=15):
        """
        Args:
            data_root: Path to root directory containing:
                - positive/images/*.png
                - positive/masks/*.png
                - negative/images/*.png
                - negative/masks/*.png
                - hard_negative/images/*.png
                - hard_negative/masks/*.png
            split: 'train' or 'val' (for future train/val split)
            transform: torchvision transforms for RGB images
            negatives_per_positive: How many negatives to sample per positive
        """
        self.data_root = Path(data_root)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.negatives_per_positive = negatives_per_positive

        # Load all samples
        self.positive_samples = self._load_samples('positive')
        self.negative_samples = self._load_samples('negative')
        self.hard_negative_samples = self._load_samples('hard_negative')
        self.radius = radius
        self.threshold = threshold

        print(f"Loaded {len(self.positive_samples)} positive samples")
        print(f"Loaded {len(self.negative_samples)} negative samples")
        print(f"Loaded {len(self.hard_negative_samples)} hard negative samples")

    def _load_samples(self, category):
        """Load all image/mask pairs from a category folder."""
        images_dir = self.data_root / category / 'images'
        masks_dir = self.data_root / category / 'masks'

        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist")
            return []

        samples = []
        for img_path in sorted(images_dir.glob('*.png')):
            # Assume mask has same filename
            mask_path = masks_dir / img_path.name #.replace('png', 'jpg')
            if not mask_path.exists():
                print(f"Warning: mask not found for {img_path}")
                continue

            samples.append({
                'image_path': str(img_path),
                'mask_path': str(mask_path),
                'category': category,
                'label': 1.0 if category == 'positive' else 0.0
            })

        return samples

    def __len__(self):
        # One "item" = 1 positive + N negatives
        return len(self.positive_samples)

    def __getitem__(self, idx):
        """
        Returns a dict containing:
        - Multiple samples (1 positive + several negatives)
        - Each sample has: rgb, rgb_geometric, label, difficulty
        """
        # Get the positive sample
        pos_sample = self.positive_samples[idx]

        # Sample negatives (same as before)
        n_hard = self.negatives_per_positive // 2
        n_easy = self.negatives_per_positive - n_hard

        selected_negatives = []

        if len(self.hard_negative_samples) > 0:
            n_hard = min(n_hard, len(self.hard_negative_samples))
            hard_indices = np.random.choice(len(self.hard_negative_samples),
                                        size=n_hard,
                                        replace=False)
            selected_negatives.extend([self.hard_negative_samples[i] for i in hard_indices])

        if len(self.negative_samples) > 0:
            n_easy = min(n_easy, len(self.negative_samples))
            easy_indices = np.random.choice(len(self.negative_samples),
                                        size=n_easy,
                                        replace=False)
            selected_negatives.extend([self.negative_samples[i] for i in easy_indices])

        # Combine positive + negatives
        all_samples = [pos_sample] + selected_negatives

        original_positions = list(range(len(all_samples)))  # [0, 1, 2, 3, ...]

        # # DEBUG: Print before shuffle
        # print(f"\n=== Dataset[{idx}] BEFORE shuffle ===")
        # print(f"Number of samples: {len(all_samples)}")
        # print(f"Labels before: {[s['label'] for s in all_samples]}")
        # print(f"Positions before: {original_positions}")

        # CORRECT:
        shuffle_indices = np.random.permutation(len(all_samples))
        all_samples = [all_samples[i] for i in shuffle_indices]
        # DON'T shuffle original_positions! Just assign the indices themselves:
        original_positions = shuffle_indices.tolist()  # These ARE the original positions

        # # DEBUG: Print after shuffle
        # print(f"Labels after: {[s['label'] for s in all_samples]}")
        # print(f"Positions after: {original_positions}")
        # print(f"Positive is now at index: {[s['label'] for s in all_samples].index(1.0)}")

        # Process each sample
        batch = {
            'rgb': [],
            'rgb_geometric': [],
            'labels': [],
            'difficulties': [],
            'positions': []  # *** NEW: track position in batch ***
        }

        for sample, pos in zip(all_samples, original_positions):
            rgb, rgb_geom = self._process_sample(sample)

            batch['rgb'].append(rgb)
            batch['rgb_geometric'].append(rgb_geom)
            batch['labels'].append(sample['label'])
            batch['difficulties'].append(sample['category'])
            batch['positions'].append(pos)  # *** NEW ***

        # Stack into tensors
        return {
            'rgb': torch.stack(batch['rgb']),
            'rgb_geometric': torch.stack(batch['rgb_geometric']),
            'labels': torch.tensor(batch['labels'], dtype=torch.float32),
            'difficulties': batch['difficulties'],
            'positions': batch['positions']  # *** NEW: list of ints ***
        }

    def __getitem__v2(self, idx):
        """
        Returns a dict containing:
        - Multiple samples (1 positive + several negatives)
        - Each sample has: rgb, rgb_geometric, label, difficulty
        """
        # Get the positive sample
        pos_sample = self.positive_samples[idx]

        # Sample negatives (same as before)
        n_hard = self.negatives_per_positive // 2
        n_easy = self.negatives_per_positive - n_hard

        selected_negatives = []

        if len(self.hard_negative_samples) > 0:
            n_hard = min(n_hard, len(self.hard_negative_samples))
            hard_indices = np.random.choice(len(self.hard_negative_samples),
                                            size=n_hard,
                                            replace=False)
            selected_negatives.extend([self.hard_negative_samples[i] for i in hard_indices])

        if len(self.negative_samples) > 0:
            n_easy = min(n_easy, len(self.negative_samples))
            easy_indices = np.random.choice(len(self.negative_samples),
                                            size=n_easy,
                                            replace=False)
            selected_negatives.extend([self.negative_samples[i] for i in easy_indices])

        # Combine positive + negatives
        all_samples = [pos_sample] + selected_negatives

        # *** NEW: SHUFFLE THE ORDER ***
        # Keep track of which was positive
        shuffle_indices = np.random.permutation(len(all_samples))
        all_samples = [all_samples[i] for i in shuffle_indices]

        # Process each sample
        batch = {
            'rgb': [],
            'rgb_geometric': [],
            'labels': [],
            'difficulties': []
        }

        for sample in all_samples:
            rgb, rgb_geom = self._process_sample(sample)

            batch['rgb'].append(rgb)
            batch['rgb_geometric'].append(rgb_geom)
            batch['labels'].append(sample['label'])
            batch['difficulties'].append(sample['category'])

        # Stack into tensors
        return {
            'rgb': torch.stack(batch['rgb']),
            'rgb_geometric': torch.stack(batch['rgb_geometric']),
            'labels': torch.tensor(batch['labels'], dtype=torch.float32),
            'difficulties': batch['difficulties']
        }

    def __getitem__v1(self, idx):
        """
        Returns a dict containing:
        - Multiple samples (1 positive + several negatives)
        - Each sample has: rgb, rgb_geometric, label, difficulty
        """
        # Get the positive sample
        pos_sample = self.positive_samples[idx]

        # Sample negatives
        # Mix of hard and easy negatives
        n_hard = self.negatives_per_positive // 2
        n_easy = self.negatives_per_positive - n_hard

        selected_negatives = []

        # Sample hard negatives
        if len(self.hard_negative_samples) > 0:
            n_hard = min(n_hard, len(self.hard_negative_samples))
            hard_indices = np.random.choice(len(self.hard_negative_samples),
                                           size=n_hard,
                                           replace=False)
            selected_negatives.extend([self.hard_negative_samples[i] for i in hard_indices])

        # Sample easy negatives
        if len(self.negative_samples) > 0:
            n_easy = min(n_easy, len(self.negative_samples))
            easy_indices = np.random.choice(len(self.negative_samples),
                                           size=n_easy,
                                           replace=False)
            selected_negatives.extend([self.negative_samples[i] for i in easy_indices])

        # Combine positive + negatives
        all_samples = [pos_sample] + selected_negatives

        # Process each sample
        batch = {
            'rgb': [],
            'rgb_geometric': [],
            'labels': [],
            'difficulties': []
        }

        for sample in all_samples:
            rgb, rgb_geom = self._process_sample(sample)

            batch['rgb'].append(rgb)
            batch['rgb_geometric'].append(rgb_geom)
            batch['labels'].append(sample['label'])
            batch['difficulties'].append(sample['category'])

        # Stack into tensors
        return {
            'rgb': torch.stack(batch['rgb']),                    # (N, 3, H, W)
            'rgb_geometric': torch.stack(batch['rgb_geometric']), # (N, 6, H, W)
            'labels': torch.tensor(batch['labels'], dtype=torch.float32), # (N,)
            'difficulties': batch['difficulties']                 # List of strings
        }

    def _process_sample(self, sample):
        """
        Load and process one sample.

        Returns:
            rgb: (3, H, W) normalized RGB tensor
            rgb_geometric: (6, H, W) RGB + 3 geometric channels
        """
        # Load RGB image
        rgb_image = Image.open(sample['image_path']).convert('RGB')

        # Load mask (should have 2 pieces with different values)
        mask_image = Image.open(sample['mask_path']).convert('L')
        mask_array = np.array(mask_image)

        # Create geometric features BEFORE resizing
        geometric_features = self._create_geometric_features(mask_array)

        # Now resize everything to 224x224
        rgb_resized = rgb_image.resize((224, 224), Image.BILINEAR)
        geometric_resized = self._resize_geometric_features(geometric_features, (224, 224))

        # Apply transforms to RGB (normalize)
        rgb_tensor = self.transform(rgb_resized)  # (3, 224, 224)

        # Convert geometric to tensor
        geometric_tensor = torch.from_numpy(geometric_resized).float()  # (3, 224, 224)

        # Combine: RGB + geometric
        rgb_geometric = torch.cat([rgb_tensor, geometric_tensor], dim=0)  # (6, 224, 224)

        return rgb_tensor, rgb_geometric

    def _compute_proximity_with_overlap(self, mask, other_mask, relaxation=3.0):
        """
        Compute proximity that handles overlap correctly.

        Args:
            mask: Binary mask of the piece we're computing proximity to
            other_mask: Binary mask of the other piece
            relaxation: How far proximity extends

        Returns:
            proximity: High values near this piece's surface
        """
        from scipy.ndimage import binary_dilation, binary_erosion

        # Extract the SURFACE of the piece (its boundary)
        surface = mask & ~binary_erosion(mask)

        # Compute distance to surface
        dist_to_surface = distance_transform_edt(~surface)

        # Apply relaxation
        relaxed_dist = dist_to_surface / relaxation

        # Normalize and invert
        max_dist = relaxed_dist.max() + 1e-6
        proximity = 1.0 - np.clip(relaxed_dist / max_dist, 0, 1)

        # SPECIAL HANDLING FOR OVERLAP:
        # Pixels inside the OTHER piece but near THIS piece's surface
        # should also have high proximity
        overlap_region = mask & other_mask

        if overlap_region.any():
            # In overlap region, set proximity to 1.0 (they're touching!)
            proximity[overlap_region] = 1.0

        return proximity

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

    #@staticmethod
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

    def _create_geometric_features(self, mask_array):
        """
        Create 3 geometric feature channels.

        UPDATED: More relaxed proximity, handles overlap correctly.
        """
        unique_values = np.unique(mask_array)
        unique_values = unique_values[unique_values > 0]

        if len(unique_values) < 2:
            return np.zeros((3, mask_array.shape[0], mask_array.shape[1]), dtype=np.float32)

        val_A = unique_values[0]
        val_B = unique_values[1]

        mask_A = mask_array == val_A
        mask_B = mask_array == val_B

        # Channel 0: Proximity to piece A (INCLUSIVE, with relaxation)
        proximity_A = self._compute_proximity_inclusive(mask_A, mask_B,)

        # Channel 1: Proximity to piece B
        proximity_B = self._compute_proximity_inclusive(mask_B, mask_A)

        # Channel 2: Contact region (where pieces meet)
        contact_strength = self._compute_contact_region_edge_based(mask_A, mask_B)

        geometric = np.stack([proximity_A, proximity_B, contact_strength], axis=0)

        return geometric.astype(np.float32)

    def _create_geometric_features_v2(self, mask_array):
        """
        Create 3 geometric feature channels from a mask.
        """
        # Separate the two pieces
        unique_values = np.unique(mask_array)
        unique_values = unique_values[unique_values > 0]

        if len(unique_values) < 2:
            return np.zeros((3, mask_array.shape[0], mask_array.shape[1]), dtype=np.float32)

        val_A = 1 #unique_values[0]
        val_B = 2 #unique_values[1]

        mask_A = mask_array == val_A
        mask_B = mask_array == val_B

        # Channel 0: Proximity to piece A (anywhere on piece A's boundary)
        boundary_A = self._extract_boundary(mask_A)
        dist_A = distance_transform_edt(~boundary_A)
        proximity_A = 1.0 - np.clip(dist_A / (dist_A.max() + 1e-6), 0, 1)

        # Channel 1: Proximity to piece B
        boundary_B = self._extract_boundary(mask_B)
        dist_B = distance_transform_edt(~boundary_B)
        proximity_B = 1.0 - np.clip(dist_B / (dist_B.max() + 1e-6), 0, 1)

        # Channel 2: Contact region (FIXED - only where pieces meet)
        contact_strength = self._compute_contact_region_edge_based(mask_A, mask_B, threshold=10)

        geometric = np.stack([proximity_A, proximity_B, contact_strength], axis=0)

        return geometric.astype(np.float32)

    def _create_geometric_features_old(self, mask_array):
        """
        Create 3 geometric feature channels from a mask.

        Args:
            mask_array: (H, W) numpy array with:
                - 0 = background
                - 128 = piece A (or some value)
                - 255 = piece B (or another value)

        Returns:
            geometric: (3, H, W) numpy array with:
                - Channel 0: proximity to piece A boundary
                - Channel 1: proximity to piece B boundary
                - Channel 2: contact region strength
        """
        # Separate the two pieces
        # Assuming: background=0, piece_A=some_value, piece_B=another_value
        unique_values = np.unique(mask_array)
        unique_values = unique_values[unique_values > 0]  # Remove background

        if len(unique_values) < 2:
            # Only one piece visible (or error) - return zeros
            return np.zeros((3, mask_array.shape[0], mask_array.shape[1]), dtype=np.float32)

        # Assume first non-zero value is piece A, second is piece B
        val_A = unique_values[0]
        val_B = unique_values[1]

        mask_A = mask_array == val_A
        mask_B = mask_array == val_B

        contact_strength = self._compute_contact_region_edge_based(mask_A, mask_B)

        # # Extract boundaries
        # boundary_A = self._extract_boundary(mask_A)
        # boundary_B = self._extract_boundary(mask_B)

        # # Compute distance to boundaries
        # dist_A = distance_transform_edt(~boundary_A)
        # dist_B = distance_transform_edt(~boundary_B)

        # # Normalize to [0, 1], with 1 = close to boundary
        # max_dist_A = dist_A.max() + 1e-6
        # max_dist_B = dist_B.max() + 1e-6

        # proximity_A = 1.0 - np.clip(dist_A / max_dist_A, 0, 1)
        # proximity_B = 1.0 - np.clip(dist_B / max_dist_B, 0, 1)

        # # Contact region: where both boundaries are close
        # # Define "close" as within 10 pixels
        # threshold = 10.0

        # contact_strength = np.maximum(0, threshold - (dist_A + dist_B)) / threshold
        # contact_strength = np.clip(contact_strength, 0, 1)

        # Stack channels
        geometric = np.stack([proximity_A, proximity_B, contact_strength], axis=0)

        return geometric.astype(np.float32)

    @staticmethod
    def _extract_boundary(mask):
        """Extract boundary pixels from a binary mask."""
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        return boundary

    @staticmethod
    def _resize_geometric_features(geometric, target_size):
        """
        Resize geometric features to target size.

        Args:
            geometric: (3, H, W) numpy array
            target_size: (H_new, W_new)

        Returns:
            resized: (3, H_new, W_new) numpy array
        """
        from PIL import Image

        resized_channels = []
        for i in range(geometric.shape[0]):
            channel = geometric[i]
            # Convert to PIL, resize, convert back
            channel_pil = Image.fromarray((channel * 255).astype(np.uint8))
            channel_resized = channel_pil.resize((target_size[1], target_size[0]), Image.BILINEAR)
            channel_array = np.array(channel_resized).astype(np.float32) / 255.0
            resized_channels.append(channel_array)

        return np.stack(resized_channels, axis=0)


# Custom collate function for DataLoader
def collate_alignment_samples(batch_list):
    """
    Collate function to handle variable-length batches.
    """
    rgb = torch.cat([item['rgb'] for item in batch_list], dim=0)
    rgb_geometric = torch.cat([item['rgb_geometric'] for item in batch_list], dim=0)
    labels = torch.cat([item['labels'] for item in batch_list], dim=0)
    difficulties = [d for item in batch_list for d in item['difficulties']]
    positions = [p for item in batch_list for p in item['positions']]  # *** NEW ***

    return {
        'rgb': rgb,
        'rgb_geometric': rgb_geometric,
        'labels': labels,
        'difficulties': difficulties,
        'positions': positions  # *** NEW ***
    }

def collate_alignment_samples_v1(batch_list):
    """
    Collate function to handle variable-length batches.

    Each item in batch_list contains multiple samples (1 pos + N neg).
    Flatten them into a single batch.
    """
    rgb = torch.cat([item['rgb'] for item in batch_list], dim=0)
    rgb_geometric = torch.cat([item['rgb_geometric'] for item in batch_list], dim=0)
    labels = torch.cat([item['labels'] for item in batch_list], dim=0)
    difficulties = [d for item in batch_list for d in item['difficulties']]

    return {
        'rgb': rgb,
        'rgb_geometric': rgb_geometric,
        'labels': labels,
        'difficulties': difficulties
    }

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
                 negatives_per_positive=3):
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
            mask_path = masks_dir / img_path.name
            
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
    
    def _create_geometric_features(self, mask_array):
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
        
        # Extract boundaries
        boundary_A = self._extract_boundary(mask_A)
        boundary_B = self._extract_boundary(mask_B)
        
        # Compute distance to boundaries
        dist_A = distance_transform_edt(~boundary_A)
        dist_B = distance_transform_edt(~boundary_B)
        
        # Normalize to [0, 1], with 1 = close to boundary
        max_dist_A = dist_A.max() + 1e-6
        max_dist_B = dist_B.max() + 1e-6
        
        proximity_A = 1.0 - np.clip(dist_A / max_dist_A, 0, 1)
        proximity_B = 1.0 - np.clip(dist_B / max_dist_B, 0, 1)
        
        # Contact region: where both boundaries are close
        # Define "close" as within 10 pixels
        threshold = 10.0
        contact_strength = np.maximum(0, threshold - (dist_A + dist_B)) / threshold
        contact_strength = np.clip(contact_strength, 0, 1)
        
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
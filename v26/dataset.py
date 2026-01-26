import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

"""
Target distribution per batch:

1 positive sample (correct alignment)
2-3 hard negatives (near-miss alignments, within 10-20° rotation)
2-3 medium negatives (moderate misalignment, 20-50° rotation)
2-3 easy negatives (random alignments)
"""

class AlignmentDataset(Dataset):
    """Dataset for pairwise alignment scoring."""
    
    def __init__(self, 
                 positive_samples,      # List of correct alignments
                 negative_samples,      # List of incorrect alignments
                 hard_negative_ratio=0.3,
                 medium_negative_ratio=0.3,
                 transform=None):
        """
        Args:
            positive_samples: List of dicts with keys:
                - 'piece_A': path to piece A image
                - 'piece_B': path to piece B image
                - 'transform': ground truth transformation
                - 'label': 1.0
            negative_samples: Same structure, but with:
                - 'difficulty': 'hard', 'medium', or 'easy'
                - 'label': 0.0
        """
        self.positive_samples = positive_samples
        
        # Split negatives by difficulty
        self.hard_negatives = [s for s in negative_samples if s['difficulty'] == 'hard']
        self.medium_negatives = [s for s in negative_samples if s['difficulty'] == 'medium']
        self.easy_negatives = [s for s in negative_samples if s['difficulty'] == 'easy']
        
        self.hard_ratio = hard_negative_ratio
        self.medium_ratio = medium_negative_ratio
        self.transform = transform
    
    def __len__(self):
        return len(self.positive_samples)
    
    def __getitem__(self, idx):
        """
        Returns a batch item containing:
        - 1 positive sample
        - Multiple negative samples of varying difficulty
        """
        # Get positive sample
        pos_sample = self.positive_samples[idx]
        
        # Sample negatives for the same puzzle/piece pair
        puzzle_id = pos_sample['puzzle_id']
        
        # Get negatives for this specific puzzle
        puzzle_hard_negs = [s for s in self.hard_negatives if s['puzzle_id'] == puzzle_id]
        puzzle_med_negs = [s for s in self.medium_negatives if s['puzzle_id'] == puzzle_id]
        puzzle_easy_negs = [s for s in self.easy_negatives if s['puzzle_id'] == puzzle_id]
        
        # Sample negatives
        n_hard = np.random.poisson(2)  # Average 2 hard negatives
        n_medium = np.random.poisson(2)
        n_easy = np.random.poisson(2)
        
        hard_samples = np.random.choice(puzzle_hard_negs, min(n_hard, len(puzzle_hard_negs)), replace=False)
        medium_samples = np.random.choice(puzzle_med_negs, min(n_medium, len(puzzle_med_negs)), replace=False)
        easy_samples = np.random.choice(puzzle_easy_negs, min(n_easy, len(puzzle_easy_negs)), replace=False)
        
        # Combine all samples
        all_samples = [pos_sample] + list(hard_samples) + list(medium_samples) + list(easy_samples)
        
        # Load and process each sample
        batch = {
            'rgb': [],
            'rgb_geometric': [],
            'labels': [],
            'difficulties': []
        }
        
        for sample in all_samples:
            # Load images and create composite
            rgb, rgb_geom = self.create_alignment_input(sample)
            
            batch['rgb'].append(rgb)
            batch['rgb_geometric'].append(rgb_geom)
            batch['labels'].append(sample['label'])
            batch['difficulties'].append(sample.get('difficulty', 'positive'))
        
        # Stack into tensors
        return {
            'rgb': torch.stack(batch['rgb']),
            'rgb_geometric': torch.stack(batch['rgb_geometric']),
            'labels': torch.tensor(batch['labels'], dtype=torch.float32),
            'difficulties': batch['difficulties']
        }
    
    def create_alignment_input(self, sample):
        """
        Create RGB and RGB+Geometric inputs for one alignment.
        
        Returns:
            rgb: (3, H, W) tensor
            rgb_geometric: (6, H, W) tensor
        """
        # Load pieces
        piece_A = Image.open(sample['piece_A']).convert('RGB')
        piece_B = Image.open(sample['piece_B']).convert('RGB')
        mask_A = Image.open(sample['mask_A']).convert('L')
        mask_B = Image.open(sample['mask_B']).convert('L')
        
        # Apply transformation to create composite
        composite_rgb, composite_mask_A, composite_mask_B = self.create_composite(
            piece_A, piece_B, mask_A, mask_B, sample['transform']
        )
        
        # Create geometric features
        geometric_features = self.create_geometric_features(
            composite_mask_A, composite_mask_B
        )
        
        # Convert to tensors
        if self.transform:
            composite_rgb = self.transform(composite_rgb)
        
        # Combine RGB + geometric
        rgb_geometric = torch.cat([composite_rgb, geometric_features], dim=0)
        
        return composite_rgb, rgb_geometric
    
    def create_geometric_features(self, mask_A, mask_B):
        """
        Create 3 geometric feature channels.
        
        Returns:
            geometric: (3, H, W) tensor
        """
        from scipy.ndimage import distance_transform_edt, binary_erosion
        
        mask_A = np.array(mask_A) > 128
        mask_B = np.array(mask_B) > 128
        
        # Extract boundaries
        boundary_A = mask_A & ~binary_erosion(mask_A)
        boundary_B = mask_B & ~binary_erosion(mask_B)
        
        # Distance transforms
        dist_A = distance_transform_edt(~boundary_A)
        dist_B = distance_transform_edt(~boundary_B)
        
        # Normalize
        dist_A = 1 - (dist_A / (dist_A.max() + 1e-6))
        dist_B = 1 - (dist_B / (dist_B.max() + 1e-6))
        
        # Contact region: where both boundaries are close
        contact = np.maximum(0, 10 - (distance_transform_edt(~boundary_A) + 
                                       distance_transform_edt(~boundary_B))) / 10
        
        # Stack and convert to tensor
        geometric = np.stack([dist_A, dist_B, contact], axis=0)
        return torch.tensor(geometric, dtype=torch.float32)
    
    # You'd implement create_composite() to merge pieces with transformation
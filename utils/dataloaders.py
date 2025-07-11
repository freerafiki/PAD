
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, Optional
import os
from pathlib import Path
from PIL import Image

class BinaryDataset(Dataset):
    """Custom dataset class for binary classification"""
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        train: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train
        
        # Get class directories
        self.classes = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all image paths
        self.samples = []
        for class_dir in self.classes:
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    self.samples.append((img_path, self.class_to_idx[class_dir]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[index]
        
        # Load image using PIL
        image = Image.open(img_path)
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to binary label (0 or 1)
        label = torch.tensor(1 if label == 0 else 0)
        
        return image, label



def create_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4, 
    force_validation_shuffle: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Creates training and validation data loaders"""
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BinaryDataset(train_dir, train_transforms, train=True)
    val_dataset = BinaryDataset(val_dir, val_transforms, train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=force_validation_shuffle,
        num_workers=num_workers,
        # pin_memory=True
    )
    
    return train_loader, val_loader

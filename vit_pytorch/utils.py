import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
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

class PatchEmbed(nn.Module):
    """Patch embedding layer for transformer input"""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # Number of patches
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 3-channel and 1-channel images
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Patch embedding
        x = self.proj(x).flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class SmallTransformer(nn.Module):
    """Small transformer model for binary classification"""
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embedding
        x = x + self.pos_embed[:, 1:, :]
        
        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x

def create_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Creates training and validation data loaders"""
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True
    )
    
    return train_loader, val_loader

def train_binary_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    patience: int = 5
) -> None:
    """Trains the binary classifier with early stopping and learning rate scheduling"""
    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Initialize training variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Train loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target.view(-1, 1).float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == target.view(-1, 1).float()).sum().item()
            total += target.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target.view(-1, 1).float())
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == target.view(-1, 1).float()).sum().item()
                val_total += target.size(0)
        
        # Calculate epoch statistics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        train_acc = correct / total
        val_acc = val_correct / val_total
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Check for improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model_vit.pth')
        else:
            epochs_without_improvement += 1
            
            # Reduce learning rate
            if epochs_without_improvement % 2 == 0:
                learning_rate *= 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
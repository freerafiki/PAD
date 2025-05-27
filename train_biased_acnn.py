import os
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the biased spatial attention layer
class BiasedSpatialAttention(nn.Module):
    def __init__(self, channels, focus_region=(0.0, 1.0, 0.0, 1.0)):
        super(BiasedSpatialAttention, self).__init__()
        self.spatial_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.focus_x_min, self.focus_x_max, self.focus_y_min, self.focus_y_max = focus_region
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate spatial weights
        spatial_weights = torch.sigmoid(self.spatial_conv(x))
        
        # Add initial bias
        x_coords = torch.linspace(0, 1, width).to(x.device)
        y_coords = torch.linspace(0, 1, height).to(x.device)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords)
        
        bias_mask = ((x_grid >= self.focus_x_min) & 
                    (x_grid <= self.focus_x_max) &
                    (y_grid >= self.focus_y_min) & 
                    (y_grid <= self.focus_y_max)).float().unsqueeze(0)
        
        combined_weights = spatial_weights * 0.7 + bias_mask * 0.3
        return x * combined_weights

# Define the CNN model
class AttentionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AttentionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            BiasedSpatialAttention(128),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*28*28, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train(model, device, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Calculate loss
        loss = criterion(outputs, target)
        
        # Backward pass
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        
        # Print progress
        if batch_idx % 100 == 99:
            print(f'Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
            running_loss = 0.0
            
    return running_loss / len(loader)

def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy

def plot_training_progress(train_losses, val_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(122)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')

# Main training script
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data augmentation and normalization
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(root='/media/lucap/big_data/datasets/repair/ground_truth/pairwise_alignments_dataset',
                                       transform=data_transforms)
    val_dataset = datasets.ImageFolder(root='/media/lucap/big_data/datasets/repair/ground_truth/pairwise_alignments_dataset',
                                     transform=data_transforms)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)
    val_loader = DataLoader(dataset=val_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4)
    
    # Initialize model, criterion, and optimizer
    model = AttentionCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    n_epochs = 10
    train_losses = []
    val_accs = []
    
    print(f'Starting training for {n_epochs} epochs...')
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        epoch_loss = train(model, device, train_loader, criterion, optimizer)
        train_losses.append(epoch_loss)
        val_acc = evaluate(model, device, val_loader)
        val_accs.append(val_acc)
    
    # Plot training progress
    plot_training_progress(train_losses, val_accs)
    
    # Save trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'data_transforms': data_transforms,
        'model_architecture': 'AttentionCNN',
        'num_classes': 2,
        'focus_region': (0.0, 1.0, 0.0, 1.0),
    }, 'attention_cnn_repair.pth')

if __name__ == '__main__':
    main()
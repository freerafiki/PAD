import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple, Optional
import os 
from torch.utils.data import Dataset, DataLoader
from utils.predictions import save_predictions

def train_binary_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    patience: int = 5,
    save_visualization: bool = False,
    save_vis_each: int = 5
) -> None:
    """Trains the binary classifier with early stopping and learning rate scheduling"""
    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
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
        
        if save_visualization == True and epoch % save_vis_each == 0:
            save_predictions(model, train_loader, 8, subfolder='training', suffix=f'train_{epoch}epochs')
            save_predictions(model, val_loader, 8, subfolder='training', suffix=f'val_{epoch}epochs')

        # Check for improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            # Save best model
            target_dir = os.path.join('checkpoints', f'{model.model_name}')
            os.makedirs(target_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(target_dir, 'best_model.pth'))
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
import torch.nn as nn
import torch 
import torchvision
from typing import Tuple, Optional
from torchvision.models import resnet50, ResNet50_Weights

class ResNetBinaryClassifier(nn.Module):
    """ResNet50-based binary classifier"""
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (3, 224, 224),
        freeze_base: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.base_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model_name = 'ResNet50'

        # Freeze base layers if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Replace final layer with binary classification head
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
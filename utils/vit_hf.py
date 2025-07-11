from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch 
import torch.nn as nn

class HuggingFaceTransformer(nn.Module):
    """Hugging Face transformer model for binary classification"""
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = "HuggingFaceTransformer"
        # Load pre-trained model and feature extractor
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        
        # Modify classification head for binary classification
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.model.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs.logits
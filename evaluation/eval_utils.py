import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple, Optional
import os
from pathlib import Path

def evaluate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_type: str = "resnet"
) -> Tuple[dict, np.ndarray]:
    """Evaluates the model on validation set and returns metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            if model_type == "transformer":
                outputs = model(data)
            else:  # resnet
                outputs = model(data)
            
            # Convert to probabilities
            probabilities = torch.sigmoid(outputs).squeeze()
            
            # Store results
            all_predictions.extend((probabilities > 0.5).cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    return calculate_metrics(all_predictions, all_labels, all_probabilities)

def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray
) -> dict:
    """Calculates various metrics for model evaluation."""
    metrics = {
        'accuracy': np.mean(predictions == labels),
        'precision': np.sum((predictions == 1) & (labels == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0,
        'recall': np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1) if np.sum(labels == 1) > 0 else 0,
        'f1_score': 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if metrics['precision'] + metrics['recall'] > 0 else 0,
        'auc': roc_auc_score(labels, probabilities)
    }
    
    return metrics

def print_evaluation_results(
    metrics: dict,
    confusion_matrix: np.ndarray,
    class_report: str
) -> None:
    """Prints evaluation results in a nicely formatted way."""
    print("\n=== Model Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    print("\nClassification Report:")
    print(class_report)

def load_and_evaluate_model(
    model_path: str,
    model_type: str = "resnet",
    device: Optional[torch.device] = None
) -> None:
    """Loads a saved model and evaluates it on the validation set."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = torch.load(model_path, map_location=device)
    model.to(device)
    
    # Load validation data
    _, val_loader = create_data_loaders(
        train_dir="./path/to/train/directory",
        val_dir="./path/to/validation/directory",
        batch_size=32,
        num_workers=4
    )
    
    # Evaluate model
    metrics, confusion_matrix = evaluate_model(model, val_loader, device, model_type)
    class_report = classification_report(confusion_matrix[:, 1], confusion_matrix[0, :] + confusion_matrix[1, :], output_dict=True)
    
    # Print results
    print_evaluation_results(metrics, confusion_matrix, class_report)
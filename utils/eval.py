import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from typing import Tuple, Optional
import os
from pathlib import Path
from utils.dataloaders import create_data_loaders
from utils.models import load_model
import json 

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
            probabilities = outputs #torch.sigmoid(outputs).squeeze()
            
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
    accuracy = np.mean(predictions == labels)
    true_positive = np.sum(np.squeeze(predictions == 1) == np.squeeze(labels == 1))
    precision = true_positive / len(predictions)
    false_negatives = np.sum(np.squeeze(predictions == 0) == np.squeeze(labels == 1))
    recall = false_negatives / len(predictions)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0,
        'auc': roc_auc_score(labels, probabilities),
        'true_positive': true_positive.astype(np.float64),
        'false_negatives': false_negatives.astype(np.float64)
    }

    cm = confusion_matrix(labels, predictions)
    
    return metrics, cm

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

    model = load_model(model_type=model_type, trained_model_path=model_path)    
    model.to(device)
    
    # Load validation data
    _, val_loader = create_data_loaders(
        train_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/train',
        val_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/validation',
        batch_size=32,
        num_workers=4
    )
    
    # Evaluate model
    metrics, confusion_matrix = evaluate_model(model, val_loader, device, model_type)
    class_report = classification_report(confusion_matrix[:, 1], confusion_matrix[0, :] + confusion_matrix[1, :], output_dict=True)
    
    # Print results
    print_evaluation_results(metrics, confusion_matrix, class_report)

    with open(os.path.join('checkpoints', model.model_name, 'metrics.json'), 'w') as jf:
        json.dump(metrics, jf, indent=3)


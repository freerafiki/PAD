import torch 
from utils import load_and_evaluate_model

# Example usage
if __name__ == "__main__":
    model_path = "./best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Starting model evaluation...")
    load_and_evaluate_model(
        model_path=model_path,
        model_type="resnet",  # or "transformer"
        device=device
    )
    print("Evaluation completed!")
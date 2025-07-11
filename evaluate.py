import torch 
from utils.eval import load_and_evaluate_model
import argparse 

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Starting model evaluation...")
    load_and_evaluate_model(
        model_type=args.model,
        model_path=args.weights,  # or "transformer"
        device=device
    )
    print("Evaluation completed!")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-M', type=str, default='')
    parser.add_argument('--weights', '-W', type=str, default='best_model.pth')
    args = parser.parse_args()
    main(args)
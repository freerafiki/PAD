import torch
from utils.dataloaders import create_data_loaders
from utils.training import train_binary_classifier
from utils.models import load_model
import argparse
import os 
import yaml 

def main(args):
    
    with open("input.yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    print('data loaders')
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/train',
        val_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/validation',
        batch_size=parameters['batch_size'],
        num_workers=parameters['num_workers']
    )

    model = load_model(args.model, args.continue_from_trained_model)
    model.to(device)

    print('training')
    # Train the model
    train_binary_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=parameters['num_epochs'],
        learning_rate=parameters['learning_rate'],
        patience=parameters['patience'],
        save_visualization=parameters['save_visualization'],
        save_vis_each=parameters['save_vis_each']
    )

    target_dir = os.path.join('checkpoints', f'{model.model_name}')
    os.makedirs(target_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(target_dir, 'last_model.pth'))

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-M', type=str, default='')
    parser.add_argument('--continue_from_trained_model', '-C', type=str, default=None)
    args = parser.parse_args()
    main(args)
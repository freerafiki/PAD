import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import random
from utils.dataloaders import create_data_loaders
from utils.models import load_model
import os 
from utils.predictions import save_predictions

def main(args):

    # Load model
    model = load_model(args.model, args.weights)
    model.eval()

    # Create dataloader
    _, val_dataloader = create_data_loaders(        
        train_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/train',
        val_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/validation',
        batch_size=8,
        num_workers=4, 
        force_validation_shuffle=True)
    
    # Visualize predictions
    save_predictions(model, val_dataloader, 8, suffix='1')
    save_predictions(model, val_dataloader, 8, suffix='2')


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-M', type=str, default='')
    parser.add_argument('--weights', '-W', type=str, default='best_model.pth')
    args = parser.parse_args()
    main(args)
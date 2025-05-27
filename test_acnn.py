import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from train_biased_acnn import AttentionCNN
import random 

def load_image(image_path, transform=None):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def visualize_predictions(model, image_paths, device, transform=None):
    """Visualize model predictions on multiple images."""
    # Create figure with subplots
    n_images = len(image_paths)
    random.shuffle(image_paths)
    fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
    axes = axes.ravel()
    
    # Load and preprocess images
    images = []
    for path in image_paths:
        img = load_image(path, transform)
        if transform:
            img = img.unsqueeze(0).to(device)
        images.append(img)
    
    # Get predictions
    with torch.no_grad():
        model.eval()
        outputs = []
        for img in images:
            output = model(img)
            prob = torch.nn.functional.softmax(output, dim=1)
            outputs.append(prob)
    
    # Plot original images and predictions
    for i, (img, prob) in enumerate(zip(images, outputs)):
        # Plot original image
        img = img.squeeze(0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title('Original')
        
        # Plot prediction
        pred_class = torch.argmax(prob)
        confidence = prob.max().item()
        axes[i + n_images].imshow(img)
        axes[i + n_images].axis('off')
        axes[i + n_images].set_title(f'Prediction: {pred_class}\nConfidence: {confidence:.2f}')
    
    plt.tight_layout()
    return fig

def load_model(model_path, device):
    """Load model with architecture and state dictionary."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance with saved architecture parameters
    model = AttentionCNN(
        num_classes=checkpoint['num_classes'],
        # focus_region=checkpoint['focus_region']
    ).to(device)
    
    # Load state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['data_transforms']

def test_model(model_path, test_folder, device, n_images=15):
    """Test model on a folder of images."""
    # Load model and transforms
    model, transform = load_model(model_path, device)
    model.eval()
    
    # # Data transforms
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                        std=[0.229, 0.224, 0.225])
    # ])
    
    # Get test images
    image_paths = [os.path.join(test_folder, f) for f in os.listdir(test_folder)][:n_images]
    
    # Visualize predictions
    fig = visualize_predictions(model, image_paths, device, transform)
    plt.show()

# Example usage
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model('attention_cnn_repair.pth', '/media/lucap/big_data/datasets/repair/ground_truth/pairwise_alignments_dataset/incorrect', device)
import torch 
import matplotlib.pyplot as plt 
import os 

def save_predictions(model, dataloader, num_images, subfolder='preds', suffix=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Get random batch
    batch = next(iter(dataloader))
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        predicted = outputs > 0.5
        # _, predicted = torch.max(outputs, 1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(num_images):
        axes[i].imshow(images[i].cpu().permute(1, 2, 0))
        axes[i].set_title(f'Pred: {predicted[i].item()}, Label: {labels[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    target_dir = os.path.join('checkpoints', subfolder, model.model_name)
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(os.path.join(target_dir, f'predictions_{suffix}.png'))
    plt.close()
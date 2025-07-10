import torch
from utils import create_data_loaders, HuggingFaceTransformer, train_binary_classifier

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/dataset/train',
    val_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/dataset/validation',
    batch_size=32,
    num_workers=4
)

# Create and move model to device
model = HuggingFaceTransformer(
    model_name="google/vit-base-patch16-224-in21k",
    num_classes=2,
    dropout=0.1
)
model.to(device)

# Train the model
train_binary_classifier(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=10,
    learning_rate=1e-4,
    patience=5
)

torch.save(model.state_dict(), 'last_model_vit_hf.pth')
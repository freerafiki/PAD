import torch
from utils import create_data_loaders, SmallTransformer, train_binary_classifier

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/train',
    val_dir='/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/validation',
    batch_size=32,
    num_workers=4
)

# Create and move model to device
model = SmallTransformer(
    patch_size=16,
    embed_dim=128,
    depth=4,
    num_heads=4,
    mlp_ratio=2.0,
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

torch.save(model.state_dict(), 'last_model_vit.pth')
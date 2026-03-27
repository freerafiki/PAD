from torch.utils.data import DataLoader
from dataset_v3 import PrecomposedAlignmentDataset, collate_alignment_samples, ShuffledBatchSampler
import matplotlib.pyplot as plt
import numpy as np


# dataset_name = 'escher'
# Create dataset
dataset = PrecomposedAlignmentDataset(
    data_root='/media/lucap/big_data/datasets/wikiart_PAD/PAD_dataset__Wikiart',
    # '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v4',
    max_negatives_per_positive=15,  # Will sample 3 hard + 3 easy
    radius = 50,
    threshold = 50
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,  # 4 positives, each with 6 negatives = 28 total samples per batch
    shuffle=False,
    sampler=ShuffledBatchSampler(dataset, shuffle=True, seed=42),  # *** NEW ***
    collate_fn=collate_alignment_samples,
    num_workers=4
)

# Test it
for batch in dataloader:
    print(f"RGB shape: {batch['rgb'].shape}")              # e.g., (28, 3, 224, 224)
    print(f"RGB+Geom shape: {batch['rgb_geometric'].shape}") # e.g., (28, 6, 224, 224)
    print(f"Labels shape: {batch['labels'].shape}")        # e.g., (28,)
    print(f"Difficulties: {batch['difficulties'][:5]}")    # e.g., ['positive', 'hard_negative', ...]
    break


# # Test the dataset before training
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     dataset = PrecomposedAlignmentDataset(
#         data_root='./data',
#         negatives_per_positive=4
#     )

# Get one batch
sample = dataset[0]

print(f"Batch contains {len(sample['rgb'])} samples")
print(f"Labels: {sample['labels']}")
print(f"Difficulties: {sample['difficulties']}")

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(15, 6))

for i in range(min(4, len(sample['rgb']))):
    # RGB (denormalize for visualization)
    rgb = sample['rgb'][i].permute(1, 2, 0).numpy()
    rgb = rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    rgb = np.clip(rgb, 0, 1)

    axes[0, i].imshow(rgb)
    axes[0, i].set_title(f"{sample['difficulties'][i]}\nLabel: {sample['labels'][i]:.1f}")
    axes[0, i].axis('off')

    # Contact region (channel 5 of rgb_geometric)
    contact = sample['rgb_geometric'][i, 5].numpy()
    axes[1, i].imshow(contact, cmap='hot')
    axes[1, i].set_title('Contact Region')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('dataset_sample.png')
print("Saved visualization to dataset_sample.png")

# Enhanced visualization
fig, axes = plt.subplots(4, 6, figsize=(15, 9))

for i in range(min(6, len(sample['rgb']))):
    # RGB
    rgb = sample['rgb'][i].permute(1, 2, 0).numpy()
    rgb = rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    rgb = np.clip(rgb, 0, 1)

    axes[0, i].imshow(rgb)
    axes[0, i].set_title(f"{sample['difficulties'][i]}")
    axes[0, i].axis('off')

    # Proximity to A (channel 3)
    prox_A = sample['rgb_geometric'][i, 3].numpy()
    axes[1, i].imshow(prox_A, cmap='Reds', vmin=0, vmax=1)
    axes[1, i].set_title('Proximity to A')
    axes[1, i].axis('off')

    # Proximity to B (channel 4)
    prox_B = sample['rgb_geometric'][i, 4].numpy()
    axes[2, i].imshow(prox_B, cmap='Reds', vmin=0, vmax=1)
    axes[2, i].set_title('Proximity to B')
    axes[2, i].axis('off')

    # Contact region (channel 5)
    contact = sample['rgb_geometric'][i, 5].numpy()
    axes[3, i].imshow(contact, cmap='hot', vmin=0, vmax=1)
    axes[3, i].set_title(f'Contact (max={contact.max():.2f})')
    axes[3, i].axis('off')

plt.tight_layout()
plt.savefig(f'dataset_sample_debug_r{dataset.radius}_t{dataset.threshold}.png', dpi=150)

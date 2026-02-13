# test_shuffle.py

from dataset_v2 import PrecomposedAlignmentDataset

DATA_ROOT = '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v4'  # Your data directory

# Create dataset
dataset = PrecomposedAlignmentDataset(
    data_root=DATA_ROOT,
    negatives_per_positive=6
)

# Get one sample multiple times to see if shuffle changes
print("Testing shuffle - getting same item 5 times:\n")

for trial in range(5):
    sample = dataset[0]  # Get first item

    labels = sample['labels'].numpy()
    positions = sample['positions']

    positive_idx = (labels == 1.0).argmax()

    print(f"Trial {trial+1}:")
    print(f"  Labels: {labels}")
    print(f"  Positions: {positions}")
    print(f"  Positive at index: {positive_idx}")
    print(f"  Positive's original position: {positions[positive_idx]}")
    print()

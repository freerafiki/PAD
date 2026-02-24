from dataset_v3 import PrecomposedAlignmentDataset
import matplotlib.pyplot as plt 
import numpy as np 

def analyze_group_size_distribution(dataset):
    """
    Analyze the distribution of group sizes in the dataset.
    """
    group_sizes = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        group_size = len(sample['labels'])
        group_sizes.append(group_size)
    
    group_sizes = np.array(group_sizes)
    
    print("\n=== Group Size Distribution ===")
    print(f"Total pairs: {len(group_sizes)}")
    print(f"Min group size: {group_sizes.min()}")
    print(f"Max group size: {group_sizes.max()}")
    print(f"Mean group size: {group_sizes.mean():.2f}")
    print(f"Median group size: {np.median(group_sizes):.0f}")
    
    # Histogram
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(group_sizes, bins=range(1, group_sizes.max() + 2), edgecolor='black')
    plt.xlabel('Group Size (1 positive + N negatives)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Group Sizes')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Mean: {group_sizes.mean():.1f}\nMedian: {np.median(group_sizes):.0f}\nMin: {group_sizes.min()}\nMax: {group_sizes.max()}"
    plt.text(0.95, 0.95, stats_text, 
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=12)
    
    plt.savefig('group_size_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved distribution plot to group_size_distribution.png")
    
    # Show pairs with very few or very many negatives
    small_groups = np.where(group_sizes < 3)[0]
    large_groups = np.where(group_sizes > 15)[0]
    
    if len(small_groups) > 0:
        print(f"\n⚠️  {len(small_groups)} pairs have <3 negatives:")
        for idx in small_groups[:5]:
            sample = dataset[idx]
            print(f"  Pair {idx} ({sample['pair_key']}): {group_sizes[idx]} samples")
    
    if len(large_groups) > 0:
        print(f"\n✓ {len(large_groups)} pairs have >15 negatives:")
        for idx in large_groups[:5]:
            sample = dataset[idx]
            print(f"  Pair {idx} ({sample['pair_key']}): {group_sizes[idx]} samples")

# Use it
DATA_ROOT = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v4"

dataset = PrecomposedAlignmentDataset(
    data_root=DATA_ROOT,
    max_negatives_per_positive=None  # Use all available
)

analyze_group_size_distribution(dataset)
# ```

# Expected output:
# ```
# === Group Size Distribution ===
# Total pairs: 1039
# Min group size: 2
# Max group size: 18
# Mean group size: 7.3
# Median group size: 6

# ⚠️  23 pairs have <3 negatives:
#   Pair 12 (puzzle_0001_RP_group_5|3_RPf_101|4_RPf_102): 2 samples
#   Pair 45 (puzzle_0003_RP_group_8|7_RPf_201|8_RPf_202): 3 samples
#   ...

# ✓ 87 pairs have >15 negatives:
#   Pair 234 (puzzle_0015_RP_group_12|12_RPf_501|13_RPf_502): 18 samples
#   ...
def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['baseline', 'geometric', 'multimodal'],
                       help='Type of model')
    parser.add_argument('--data_root', type=str, default='./data', help='Path to data')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of pairs to visualize')  # Changed from num_batches
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--analyze_failures', action='store_true', help='Find and visualize failures')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for pair selection')

    args = parser.parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir) / Path(args.model).stem
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir}")

    # Set seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Load model
    model = load_model(args.model, args.model_type, device)

    # Load dataset
    dataset = PrecomposedAlignmentDataset(
        data_root=args.data_root,
        negatives_per_positive=4
    )

    print(f"Dataset: {len(dataset)} pairs")

    # *** Randomly select pairs ***
    num_pairs = len(dataset)
    num_to_viz = min(args.num_samples, num_pairs)

    random_indices = np.random.choice(num_pairs, size=num_to_viz, replace=False)

    print(f"\nVisualizing {num_to_viz} random pairs...")
    print(f"Selected pair indices: {sorted(random_indices.tolist())}")

    # Visualize each selected pair
    for viz_idx, pair_idx in enumerate(random_indices):
        # Get pair data
        sample = dataset[pair_idx]

        # Create batch dict
        batch = {
            'rgb': sample['rgb'].to(device),
            'rgb_geometric': sample['rgb_geometric'].to(device),
            'labels': sample['labels'].to(device),
            'difficulties': sample['difficulties'],
            'positions': sample['positions'],
            'pair_keys': [sample['pair_key']]
        }

        print(f"\nPair {viz_idx+1}/{num_to_viz}: {sample['pair_key']}")

        visualize_batch(
            model,
            batch,
            device,
            output_dir,
            batch_idx=viz_idx,
            model_type=args.model_type
        )

    print(f"\nDone! Visualizations saved to {output_dir}")

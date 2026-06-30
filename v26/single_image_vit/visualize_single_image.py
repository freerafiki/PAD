"""
Visualize ViT model predictions for the single-image dataset.

Usage:
    python single_image_vit/visualize_single_image.py --model checkpoints/Option1_RGB_best.pth
    python single_image_vit/visualize_single_image.py --model checkpoints/Option1_RGB_best.pth --inspect-batch
    python single_image_vit/visualize_single_image.py --model checkpoints/Option1_RGB_best.pth --show-batch
"""

import sys
from pathlib import Path
_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

from single_image_utils.dataset_single import SingleImageDataset
from single_image_utils.vis_utils import (
    visualize_predictions,
    visualize_score_distribution,
    analyze_failures,
    inspect_batch_channels,
)
from single_image_vit.models import RGBScorer, GeometricScorer
from single_image_vit.config_vit import Config


def load_model(checkpoint_path, device="cuda", model_type=None):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("model_state_dict", ckpt)

    if model_type is None:
        model_type = "geometric" if "projection.weight" in state_dict else "RGB"

    cfg = Config()
    if model_type == "geometric":
        model = GeometricScorer(
            pretrained_name=cfg.model.VIT_MODEL,
            geometric_channel_scale=cfg.model.GEOMETRIC_CHANNEL_SCALE,
        )
    else:
        model = RGBScorer(
            pretrained_vit_name=cfg.model.VIT_MODEL,
            freeze_vit_layers=cfg.model.FROZEN_LAYERS,
            dropout=cfg.model.DROPOUT,
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if "val_accuracy" in ckpt:
        print(f"  Val Accuracy: {ckpt['val_accuracy']:.3f}")

    print(f"Loaded {model_type} model from {checkpoint_path}")
    return model, model_type, ckpt.get("history")


def extract_attention_maps(model, rgb, rgb_geometric, model_type="RGB"):
    model.eval()
    vit_model = model.vit

    if model_type == "geometric":
        with torch.no_grad():
            vit_input = model.projection(rgb_geometric)
    else:
        vit_input = rgb

    last_attn = vit_model.encoder.layer[-1].attention.attention

    old_attn_impl = vit_model.config._attn_implementation
    vit_model.config._attn_implementation = "eager"

    captured = []

    def hook(_m, _i, output):
        captured.append(output[1].detach())

    handle = last_attn.register_forward_hook(hook)

    with torch.no_grad():
        vit_model(vit_input)

    handle.remove()
    vit_model.config._attn_implementation = old_attn_impl

    if not captured:
        B = rgb.shape[0]
        return np.zeros((B, 14, 14))

    attn_probs = captured[0]
    attn = attn_probs.mean(dim=1)
    cls_attn = attn[:, 0, 1:]
    num_patches_side = int(np.sqrt(cls_attn.shape[1]))
    attn_maps = cls_attn.reshape(-1, num_patches_side, num_patches_side)

    return attn_maps.cpu().numpy()


def overlay_attention(image, attention_map, alpha=0.6):
    import matplotlib.cm as cm

    H, W = image.shape[:2]
    from skimage.transform import resize as sk_resize
    attn_resized = sk_resize(attention_map, (H, W), order=3, preserve_range=True)
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    heatmap = cm.jet(attn_norm)[:, :, :3]
    overlayed = alpha * heatmap + (1 - alpha) * image
    return np.clip(overlayed, 0, 1)


def run_inference(model, dataloader, device, model_type="RGB"):
    all_samples = []

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            rgb_geometric = batch["rgb_geometric"].to(device)
            labels = batch["labels"]
            categories = batch["category"]
            pair_keys = batch["pair_key"]

            if model_type == "geometric":
                logits = model(rgb_geometric).squeeze()
            else:
                logits = model(rgb).squeeze()

            scores = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(scores)):
                all_samples.append({
                    'rgb': rgb[i].cpu(),
                    'rgb_geometric': rgb_geometric[i].cpu(),
                    'score': float(scores[i]),
                    'label': float(labels_np[i]),
                    'category': categories[i],
                    'pair_key': pair_keys[i],
                })

    groups = defaultdict(list)
    for s in all_samples:
        groups[s['pair_key']].append(s)

    return all_samples, dict(groups)


def main():
    parser = argparse.ArgumentParser(description="Visualize ViT single-image model predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["RGB", "geometric"],
                        help="Override model type (auto-detected from checkpoint if not set)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=5,
                        help="Number of batches to process for batch visualization")
    parser.add_argument("--max-groups", type=int, default=20,
                        help="Max groups to visualize")
    parser.add_argument("--inspect-batch", action="store_true",
                        help="Visualize first batch channels")
    parser.add_argument("--show-batch", action="store_true",
                        help="Display batch inspection interactively instead of saving")
    parser.add_argument("--show-attention", action="store_true", default=True,
                        help="Overlay ViT attention heatmaps (default: on; use --hide-attention to disable)")
    parser.add_argument("--hide-attention", dest="show_attention", action="store_false",
                        help="Disable attention heatmap overlays")
    parser.add_argument("--analyze-failures", action="store_true",
                        help="Analyze and visualize failure cases")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for classification")
    parser.add_argument("--limit", type=int, default=600,
                        help="Limit total images (0 = all, default 600 for speed)")
    parser.add_argument("--debug", action="store_true",
                        help="Use debug mode (200 images per category)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) / Path(args.model).stem
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    model, model_type, history = load_model(args.model, device, model_type=args.model_type)
    print(f"Model type: {model_type}")

    cfg = Config()
    cache_dir = None
    if cfg.data.CACHE_DIR:
        cd = Path(cfg.data.CACHE_DIR)
        cache_dir = cd if cd.is_absolute() else Path(cfg.data.DATA_ROOT) / cd
    val_dataset = SingleImageDataset(
        data_root=cfg.data.DATA_ROOT,
        use_geometric=True,
        radius=cfg.data.RADIUS,
        threshold=cfg.data.THRESHOLD,
        debug=args.debug,
        limit=args.limit if not args.debug else 0,
        cache_dir=cache_dir,
    )

    val_dataset.augment = False

    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Val dataset: {len(val_dataset)} images")

    if args.inspect_batch or args.show_batch:
        first_batch = next(iter(dataloader))
        save_path = None
        if not args.show_batch:
            save_path = output_dir / "batch_inspection.png"
        inspect_batch_channels(
            first_batch, save_path=save_path, show=args.show_batch,
            model=model, device=device, use_geom=(model_type == "geometric"),
        )

    print("\nRunning inference on validation set...")
    all_samples, groups = run_inference(model, dataloader, device, model_type)

    # Pre-compute attention and geometric maps per group
    attn_maps_dict = {}
    geom_maps_dict = {}
    for pair_key, samples in groups.items():
        rgbs = torch.stack([s['rgb'] for s in samples]).to(device)
        rg_geom = torch.stack([s['rgb_geometric'] for s in samples]).to(device)
        if args.show_attention:
            attn_maps_dict[pair_key] = extract_attention_maps(model, rgbs, rg_geom, model_type)
        geom = []
        for i in range(len(samples)):
            geom.append((
                rg_geom[i, 3].cpu().numpy(),
                rg_geom[i, 4].cpu().numpy(),
                rg_geom[i, 5].cpu().numpy(),
            ))
        geom_maps_dict[pair_key] = geom

    print(f"\nVisualizing group predictions ({args.max_groups} groups)...")
    visualize_predictions(groups, output_dir / "groups", max_groups=args.max_groups,
                          threshold=args.threshold,
                          attn_maps_dict=attn_maps_dict if args.show_attention else None,
                          geom_maps_dict=geom_maps_dict)

    print("\nPlotting score distribution...")
    visualize_score_distribution(all_samples, output_dir / "score_distribution.png")

    if args.analyze_failures:
        print("\nAnalyzing failures...")
        analyze_failures(groups, output_dir / "failures", max_failures=20,
                         threshold=args.threshold,
                         attn_maps_dict=attn_maps_dict if args.show_attention else None,
                         geom_maps_dict=geom_maps_dict)

    pos_scores = [s['score'] for s in all_samples if s['label'] == 1.0]
    neg_scores = [s['score'] for s in all_samples if s['label'] == 0.0]
    all_labels = np.array([s['label'] for s in all_samples])
    all_preds = np.array([s['score'] > args.threshold for s in all_samples], dtype=float)
    accuracy = (all_preds == all_labels).mean()
    pos_acc = (all_preds[all_labels == 1.0] == 1.0).mean() if (all_labels == 1.0).sum() > 0 else 0.0
    neg_acc = (all_preds[all_labels == 0.0] == 0.0).mean() if (all_labels == 0.0).sum() > 0 else 0.0

    print(f"\n{'=' * 50}")
    print(f"{'Validation Metrics':^50}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:           {accuracy:.3f}")
    print(f"  Positive Acc:       {pos_acc:.3f}")
    print(f"  Negative Acc:       {neg_acc:.3f}")
    print(f"  Avg Pos Score:      {np.mean(pos_scores):.3f}")
    print(f"  Avg Neg Score:      {np.mean(neg_scores):.3f}")

    rank_correct = 0
    rank_total = 0
    for pair_key, samples in groups.items():
        scores = np.array([s['score'] for s in samples])
        labels = np.array([s['label'] for s in samples])
        pos_mask = labels == 1.0
        if pos_mask.sum() >= 1:
            best_idx = np.argmax(scores)
            if labels[best_idx] == 1.0:
                rank_correct += 1
            rank_total += 1
    print(f"  Ranking Accuracy:   {rank_correct / rank_total:.3f} ({rank_correct}/{rank_total})")
    print(f"{'=' * 50}")

    print(f"\nDone! Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

"""
Visualize CNN model predictions for the single-image dataset.

Usage:
    python single_image_cnn/visualize_cnn.py --model checkpoints/Option3_CNN_best.pth
    python single_image_cnn/visualize_cnn.py --model checkpoints/Option3_CNN_best.pth --inspect-batch
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

import torch.nn.functional as F
from single_image_utils.dataset_single import SingleImageDataset
from single_image_utils.vis_utils import (
    visualize_predictions,
    visualize_score_distribution,
    analyze_failures,
    inspect_batch_channels,
)
from single_image_cnn.resnet_models import PairwiseCompatibilityModel, PairwiseCompatibilityDualModel
from single_image_cnn.config_cnn import Config


def load_model(checkpoint_path, device="cuda"):
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("model_state_dict", ckpt)

    cfg = Config()
    if cfg.model.TYPE == 'single':
        model = PairwiseCompatibilityModel()
    elif cfg.model.TYPE == 'dual':
        raise NotImplementedError("Dual model visualization not implemented yet")
    else:
        raise ValueError(f"Unknown model type: {cfg.model.TYPE}")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if "val_accuracy" in ckpt:
        print(f"  Val Accuracy: {ckpt['val_accuracy']:.3f}")

    print(f"Loaded CNN model from {checkpoint_path}")
    return model, ckpt.get("history")


def run_inference(model, dataloader, device):
    all_samples = []

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            rgb_geometric = batch["rgb_geometric"].to(device)
            labels = batch["labels"]
            categories = batch["category"]
            pair_keys = batch["pair_key"]

            guidance_map = rgb_geometric[:, 5:6]
            logits = model(rgb, guidance_map).squeeze()

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


def extract_gradcam_maps(model, rgb, guidance_map):
    model.eval()
    target_layer = model.encoder.layer4[-1]

    features = []
    grads = []

    def fwd_hook(m, i, o):
        features.append(o)

    def bwd_hook(m, gi, go):
        grads.append(go[0])

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    rgb = rgb.clone().requires_grad_(True)
    logits = model(rgb, guidance_map)

    model.zero_grad()
    logits.sum().backward()

    fh.remove()
    bh.remove()

    feat = features[0]
    grad = grads[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat).sum(dim=1)
    cam = F.relu(cam)

    cam_maps = cam.detach().cpu().numpy()
    for i in range(cam_maps.shape[0]):
        c = cam_maps[i]
        c_min, c_max = c.min(), c.max()
        if c_max > c_min:
            cam_maps[i] = (c - c_min) / (c_max - c_min)
        else:
            cam_maps[i] = 0

    return cam_maps


def extract_gate_maps(model, rgb, guidance_map):
    model.eval()

    last_block = model.encoder.layer4[-1]
    if hasattr(last_block, 'gconv2'):
        target_conv = last_block.gconv2
    else:
        target_conv = last_block.conv2

    gate_conv = target_conv.gate_conv
    gate_outputs = []

    def hook(m, i, o):
        gate_outputs.append(o.detach())

    handle = gate_conv.register_forward_hook(hook)

    with torch.no_grad():
        model(rgb, guidance_map)

    handle.remove()

    gate_raw = gate_outputs[0]
    gate = torch.sigmoid(gate_raw)
    gate_map = gate.mean(dim=1)

    return gate_map.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Visualize CNN single-image model predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="visualizations_cnn",
                        help="Output directory for visualizations")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-groups", type=int, default=20,
                        help="Max groups to visualize")
    parser.add_argument("--inspect-batch", action="store_true",
                        help="Visualize first batch channels")
    parser.add_argument("--show-batch", action="store_true",
                        help="Display batch inspection interactively instead of saving")
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

    model, history = load_model(args.model, device)

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
            model=model, device=device, use_geom=True,
        )

    print("\nRunning inference on validation set...")
    all_samples, groups = run_inference(model, dataloader, device)

    # Pre-compute geometric, Grad-CAM, and gate maps per group
    geom_maps_dict = {}
    gradcam_maps_dict = {}
    gate_maps_dict = {}
    for pair_key, samples in groups.items():
        rgbs = torch.stack([s['rgb'] for s in samples]).to(device)
        rg_geom = torch.stack([s['rgb_geometric'] for s in samples]).to(device)
        guidance_map = rg_geom[:, 5:6]

        geom = []
        for i in range(len(samples)):
            geom.append((
                rg_geom[i, 3].cpu().numpy(),
                rg_geom[i, 4].cpu().numpy(),
                rg_geom[i, 5].cpu().numpy(),
            ))
        geom_maps_dict[pair_key] = geom

        gradcam_maps_dict[pair_key] = extract_gradcam_maps(model, rgbs, guidance_map)
        gate_maps_dict[pair_key] = extract_gate_maps(model, rgbs, guidance_map)

    print(f"\nVisualizing group predictions ({args.max_groups} groups)...")
    visualize_predictions(groups, output_dir / "groups", max_groups=args.max_groups,
                          threshold=args.threshold,
                          gradcam_maps_dict=gradcam_maps_dict,
                          gate_maps_dict=gate_maps_dict,
                          geom_maps_dict=geom_maps_dict)

    print("\nPlotting score distribution...")
    visualize_score_distribution(all_samples, output_dir / "score_distribution.png")

    if args.analyze_failures:
        print("\nAnalyzing failures...")
        analyze_failures(groups, output_dir / "failures", max_failures=20,
                         threshold=args.threshold,
                         gradcam_maps_dict=gradcam_maps_dict,
                         gate_maps_dict=gate_maps_dict,
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

import sys
from pathlib import Path
_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from rich.console import Console

from single_image_utils.dataset_single import SingleImageDataset, SamePairBatchSampler
from single_image_utils.train_utils import train_model
from single_image_vit.models import RGBScorer, GeometricScorer
from single_image_vit.config_vit import Config


def main():
    parser = argparse.ArgumentParser(description="Train ViT single-image scoring model")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Override NUM_IMAGES (train images, 0=all, maintains ratio)')
    parser.add_argument('--num-images-val', type=int, default=None,
                        help='Override NUM_IMAGES_VAL (val images)')
    args = parser.parse_args()

    cfg = Config()
    num_images = args.num_images if args.num_images is not None else cfg.data.NUM_IMAGES
    num_images_val = args.num_images_val if args.num_images_val is not None else cfg.data.NUM_IMAGES_VAL

    console = Console()
    console.print(f"Using device: {cfg.training.DEVICE}")
    if num_images > 0:
        console.print(f"Train: {num_images} images  Val: {num_images_val} images  "
                      f"({cfg.data.POSITIVE_RATIO:.0%} positive)")

    cache_dir = None
    if cfg.data.CACHE_DIR:
        cd = Path(cfg.data.CACHE_DIR)
        cache_dir = cd if cd.is_absolute() else Path(cfg.data.DATA_ROOT) / cd
    train_dataset, val_dataset = SingleImageDataset.create_puzzle_split(
        data_root=cfg.data.DATA_ROOT,
        train_ratio=cfg.data.TRAIN_RATIO,
        seed=cfg.data.SEED,
        use_geometric=cfg.data.USE_GEOMETRIC,
        radius=cfg.data.RADIUS,
        threshold=cfg.data.THRESHOLD,
        debug=cfg.data.DEBUG,
        augment=cfg.augmentation.ENABLED,
        augment_cfg=cfg.augmentation,
        num_images=num_images,
        num_images_val=num_images_val,
        positive_ratio=cfg.data.POSITIVE_RATIO,
        cache_dir=cache_dir,
    )

    if cfg.data.SAME_PAIR_BATCH:
        if not cfg.augmentation.ENABLED:
            console.print("[yellow]same_pair_batch=True — enabling augmentation for variability[/]")
            cfg.augmentation.ENABLED = True
            train_dataset.augment = True
            train_dataset.color_augment = train_dataset._build_color_augment()
            val_dataset.augment = False
        batch_size = cfg.training.BATCH_SIZE
        train_sampler = SamePairBatchSampler(train_dataset, batch_size, shuffle=True)
        val_sampler = SamePairBatchSampler(val_dataset, batch_size, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                                num_workers=4, pin_memory=True)
        console.print(f"Using same-pair batching ({len(train_sampler)} train / {len(val_sampler)} val batches)")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    console.print(f"\n=== Dataset Ready ===")
    console.print(f"Train: {len(train_dataset)} images")
    console.print(f"Val: {len(val_dataset)} images")

    train_puzzles = set(s['puzzle_id'] for s in train_dataset.samples if s['puzzle_id'])
    val_puzzles = set(s['puzzle_id'] for s in val_dataset.samples if s['puzzle_id'])
    overlap = train_puzzles & val_puzzles
    if overlap:
        console.print(f"WARNING: {len(overlap)} puzzles appear in both train and val!")
    else:
        console.print("No puzzle overlap between train and val")

    use_geom = cfg.data.USE_GEOMETRIC
    console.print(f"\n{'=' * 60}")
    console.print(f"TRAINING: {'RGB+Geometric' if use_geom else 'RGB only'} + BCE")
    console.print(f"{'=' * 60}")

    if use_geom:
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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.LEARNING_RATE,
        weight_decay=cfg.training.WEIGHT_DECAY,
    )

    start_epoch = 1
    initial_history = None
    initial_best_val_acc = 0.0
    initial_patience = 0

    if args.resume:
        console.print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=cfg.training.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for pg in optimizer.param_groups:
            pg['lr'] = cfg.training.LEARNING_RATE
        device = torch.device(cfg.training.DEVICE)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        ckpt_epoch = ckpt.get('epoch', 0)
        start_epoch = ckpt_epoch + 1
        initial_best_val_acc = ckpt.get('val_accuracy', 0.0)
        initial_history = ckpt.get('history', None)
        hist = initial_history
        if hist is not None and len(hist.get('val_accuracy', [])) > 0:
            recent_accs = hist['val_accuracy'][-cfg.training.EARLY_STOPPING_PATIENCE:]
            if all(a <= initial_best_val_acc for a in recent_accs):
                initial_patience = len(recent_accs)
        console.print(f"  Resuming from epoch {start_epoch}/{cfg.training.NUM_EPOCHS}")
        console.print(f"  Previous best val accuracy: {initial_best_val_acc:.3f}")

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        num_epochs=cfg.training.NUM_EPOCHS,
        lr=cfg.training.LEARNING_RATE,
        weight_decay=cfg.training.WEIGHT_DECAY,
        use_geom=use_geom,
        early_stopping_patience=cfg.training.EARLY_STOPPING_PATIENCE,
        model_name=cfg.name,
        max_norm=cfg.training.GRAD_CLIP_MAX_NORM,
        pos_weight_val_BCE=cfg.training.BCE_POS_WEIGHT,
        start_epoch=start_epoch,
        initial_history=initial_history,
        initial_best_val_acc=initial_best_val_acc,
        initial_patience=initial_patience,
    )


if __name__ == "__main__":
    main()

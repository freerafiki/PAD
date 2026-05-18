from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_ranking import (
    PrecomposedAlignmentDataset,
    ShuffledBatchSampler,
    collate_alignment_samples,
)
from models_ranking import MultiModalScorerV2_Practical, MultiModalScorerWeightedVit, MultiModalScorerWeightedViTFiLM, GeometricScorer, BaselineScorer
from loss_ranking import AdaptiveTopNRankingLoss, BoundaryPairwiseCorrelationLoss, BoundaryPseudoRankingLoss
from training_utils import evaluate_ranking, estimate_data_needs, diagnose_data_sufficiency, plot_training_history, plot_new_accuracy_metrics
from config import Config


def train_model(
    model,
    train_dataset,
    val_dataset,
    optimizer=None,
    num_epochs=50,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    save_dir="checkpoints",
    model_name="model",
    early_stopping_patience=10,
    bce_weight=0.15,
    ranking_weight=0.55,
    ranking_margin=0.3,
    boundary_weight=0.3,
    hard_negative_weight=2.0,
    top_n=3,
    temperature=1.0,
    max_norm=1.0,
    pos_weight_val_BCE=4.0,
    start_epoch=1,
    initial_history=None,
    initial_best_val_acc=0.0,
    initial_patience=0,
    reset_scheduler=False,
):
    """
    Training with combined BCE + AdaptiveTopNRankingLoss and early stopping for small datasets.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=ShuffledBatchSampler(train_dataset, shuffle=True, seed=42),
        collate_fn=collate_alignment_samples,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=ShuffledBatchSampler(val_dataset, shuffle=False, seed=42),
        collate_fn=collate_alignment_samples,
        num_workers=4,
        pin_memory=True,
    )

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val_BCE]).to(device))
    ranking_criterion = AdaptiveTopNRankingLoss(
        top_n=top_n,
        margin=ranking_margin,
        hard_negative_weight=hard_negative_weight,
        temperature=temperature,
    )
    boundary_criterion = BoundaryPairwiseCorrelationLoss()
    # boundary_criterion = BoundaryPseudoRankingLoss()

    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    remaining_epochs = num_epochs - start_epoch + 1
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=remaining_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    model = model.to(device)

    best_val_acc = initial_best_val_acc
    patience_counter = initial_patience

    if initial_history is not None:
        history = initial_history
    else:
        history = {
            "train_loss": [],
            "train_bce_loss": [],
            "train_ranking_loss": [],
            "train_boundary_loss": [],
            "val_loss": [],
            "val_bce_loss": [],
            "val_ranking_loss": [],
            "val_boundary_loss": [],
            "val_accuracy": [],
            "val_top3_accuracy": [],
            "val_top5_accuracy": [],
            "val_pos_score": [],
            "val_neg_score": [],
            "val_hard_neg_score": [],
            "val_single_align_accuracy": [],
            "val_single_pos_accuracy": [],
            "val_single_neg_accuracy": [],
            "val_single_hard_neg_accuracy": [],
            "val_align_set_score": [],
            "learning_rates": [],
        }

    if start_epoch > 1:
        print(f"Resuming {model_name} from epoch {start_epoch} (total: {num_epochs} epochs)")
    else:
        print(f"Training {model_name} for up to {num_epochs} epochs")
    print(f"Early stopping patience: {early_stopping_patience}")
    if initial_best_val_acc > 0:
        print(f"Previous best val accuracy: {initial_best_val_acc:.3f}")
    print(f"Loss: BCE*{bce_weight} + AdaptiveTopNRankingLoss*{ranking_weight} + BoundaryPairwiseCorrelationLoss*{boundary_weight}")
    print(f"  AdaptiveTopNRankingLoss: top_n={top_n}, margin={ranking_margin}, temperature={temperature}")
    print(f"Train samples: {len(train_dataset)} pairs")
    print(f"Val samples: {len(val_dataset)} pairs")
    print("-" * 60)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_bce_loss = 0.0
        train_ranking_loss = 0.0
        train_boundary_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch in pbar:
            rgb = batch["rgb"].to(device)
            rgb_geometric = batch["rgb_geometric"].to(device)
            labels = batch["labels"].to(device)
            difficulties = batch["difficulties"]
            group_sizes = batch['group_sizes']

            optimizer.zero_grad()

            if isinstance(model, MultiModalScorerV2_Practical) or isinstance(model, MultiModalScorerWeightedVit):
                logits = model(rgb, rgb_geometric).squeeze()
            elif isinstance(model, GeometricScorer):
                logits = model(rgb_geometric).squeeze()
            else:
                logits = model(rgb).squeeze()

            scores = torch.sigmoid(logits).squeeze()

            bce_loss = bce_criterion(logits, labels.squeeze())
            ranking_loss = ranking_criterion(scores, labels.squeeze(), difficulties, group_sizes=group_sizes)
            boundary_loss = boundary_criterion(rgb, rgb_geometric, scores, group_sizes=group_sizes)

            loss = bce_weight * bce_loss + ranking_weight * ranking_loss + boundary_weight * boundary_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_bce_loss += bce_loss.item()
            train_ranking_loss += ranking_loss.item()
            train_boundary_loss += boundary_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "bce": f"{bce_loss.item():.4f}",
                "rank": f"{ranking_loss.item():.4f}",
                "boundary": f"{boundary_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
            })

        avg_train_loss = train_loss / num_batches
        avg_train_bce = train_bce_loss / num_batches
        avg_train_ranking = train_ranking_loss / num_batches
        avg_train_boundary = train_boundary_loss / num_batches

        model.eval()
        val_loss = 0.0
        val_bce_loss = 0.0
        val_ranking_loss = 0.0
        val_boundary_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device)
                rgb_geometric = batch["rgb_geometric"].to(device)
                labels = batch["labels"].to(device)
                difficulties = batch["difficulties"]
                group_sizes = batch['group_sizes']

                if isinstance(model, MultiModalScorerV2_Practical) or isinstance(model, MultiModalScorerWeightedVit):
                    logits = model(rgb, rgb_geometric).squeeze()
                elif isinstance(model, GeometricScorer):
                    logits = model(rgb_geometric).squeeze()
                else:
                    logits = model(rgb).squeeze()

                scores = torch.sigmoid(logits).squeeze()

                bce_loss = bce_criterion(logits, labels.squeeze())
                ranking_loss = ranking_criterion(scores, labels.squeeze(), difficulties, group_sizes=group_sizes)
                boundary_loss = boundary_criterion(rgb, rgb_geometric, scores, group_sizes=group_sizes)
                loss = bce_weight * bce_loss + ranking_weight * ranking_loss + boundary_weight * boundary_loss

                val_loss += loss.item()
                val_bce_loss += bce_loss.item()
                val_ranking_loss += ranking_loss.item()
                val_boundary_loss += boundary_loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches
        avg_val_bce = val_bce_loss / num_batches
        avg_val_ranking = val_ranking_loss / num_batches
        avg_val_boundary = val_boundary_loss / num_batches

        if isinstance(model, MultiModalScorerV2_Practical) or isinstance(model, MultiModalScorerWeightedVit):
            model_type = "multimodal"
        elif isinstance(model, GeometricScorer):
            model_type = "geometric"
        else:
            model_type = "baseline"
        val_acc, val_top3_acc, val_top5_acc, avg_pos, avg_neg, avg_hard_neg, single_align_acc, single_pos_acc, single_neg_acc, single_hard_neg_acc, align_set_acc, align_set_breakdown = evaluate_ranking(
            model, val_loader, device, model_type=model_type
        )

        history["train_loss"].append(avg_train_loss)
        history["train_bce_loss"].append(avg_train_bce)
        history["train_ranking_loss"].append(avg_train_ranking)
        history["train_boundary_loss"].append(avg_train_boundary)
        history["val_loss"].append(avg_val_loss)
        history["val_bce_loss"].append(avg_val_bce)
        history["val_ranking_loss"].append(avg_val_ranking)
        history["val_boundary_loss"].append(avg_val_boundary)
        history["val_accuracy"].append(val_acc)
        history["val_top3_accuracy"].append(val_top3_acc)
        history["val_top5_accuracy"].append(val_top5_acc)
        history["val_pos_score"].append(avg_pos)
        history["val_neg_score"].append(avg_neg)
        history["val_hard_neg_score"].append(avg_hard_neg)
        history["val_single_align_accuracy"].append(single_align_acc)
        history["val_single_pos_accuracy"].append(single_pos_acc)
        history["val_single_neg_accuracy"].append(single_neg_acc)
        history["val_single_hard_neg_accuracy"].append(single_hard_neg_acc)
        history["val_align_set_score"].append(align_set_acc)
        history["learning_rates"].append(scheduler.get_last_lr()[0])

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce:.4f}, Rank: {avg_train_ranking:.4f}, Boundary: {avg_train_boundary:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Ranking Acc: {val_acc:.3f} (Top3: {val_top3_acc:.3f}, Top5: {val_top5_acc:.3f}) | "
            f"Single Align: {single_align_acc:.3f} (Pos: {single_pos_acc:.3f}, Neg: {single_neg_acc:.3f}) | "
            f"Align Set: {align_set_acc:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "history": history,
            }
            torch.save(checkpoint, save_dir / f"{model_name}_best.pth")
            torch.save(checkpoint, save_dir / f"{model_name}_best_at_epoch_{epoch}.pth")
            print(f"  → Saved best model (acc: {val_acc:.3f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.3f}")
            break

    print("-" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.3f}")

    plot_training_history(history, save_dir / f"{model_name}_history.png")
    plot_new_accuracy_metrics(history, save_dir / f"{model_name}_new_accuracy_metrics.png")

    return model, history


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train ranking model")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-split', type=str, default=None,
                        help='Create and save a 3-way puzzle split to JSON file')
    parser.add_argument('--load-split', type=str, default=None,
                        help='Load a pre-saved puzzle split from JSON file')
    args = parser.parse_args()

    cfg = Config()

    print(f"Using device: {cfg.training.DEVICE}")

    full_dataset = PrecomposedAlignmentDataset(
        data_root=cfg.data.DATA_ROOT,
        max_negatives_per_positive=cfg.data.MAX_NEGATIVES_PER_POSITIVE,
        radius=cfg.data.RADIUS,
        threshold=cfg.data.THRESHOLD,
        debug_mode=cfg.data.DEBUG,
    )

    if args.load_split:
        # Load pre-saved split
        train_dataset, val_dataset, _ = PrecomposedAlignmentDataset.from_split_file(
            args.load_split,
            max_negatives_per_positive=cfg.data.MAX_NEGATIVES_PER_POSITIVE,
            hard_negative_ratio=0.5,
            radius=cfg.data.RADIUS,
            threshold=cfg.data.THRESHOLD,
            debug_mode=cfg.data.DEBUG,
        )
    else:
        # Create split on the fly
        train_dataset, val_dataset = PrecomposedAlignmentDataset.create_puzzle_split(
            full_dataset, radius=cfg.data.RADIUS, threshold=cfg.data.THRESHOLD,
            train_ratio=cfg.data.TRAIN_RATIO, seed=cfg.data.SEED,
        )

    # Enable augmentation only on training set
    if cfg.augmentation.ENABLED:
        train_dataset = full_dataset.create_split(
            train_puzzles=set(k.split("|")[0] for k in train_dataset.pair_keys),
            augment=True, augment_cfg=cfg.augmentation,
        )

    if args.save_split:
        full_dataset.save_split(args.save_split)

    print(f"\n=== Dataset Ready ===")
    print(f"Train: {len(train_dataset)} pairs")
    print(f"Val: {len(val_dataset)} pairs")

    train_puzzles = set(k.split("|")[0] for k in train_dataset.pair_keys)
    val_puzzles = set(k.split("|")[0] for k in val_dataset.pair_keys)
    overlap = train_puzzles & val_puzzles

    if overlap:
        print(f"WARNING: {len(overlap)} puzzles appear in both train and val!")
    else:
        print("No puzzle overlap between train and val")

    print("\n" + "=" * 60)
    if cfg.model.FILM_ENABLED:
        print("TRAINING MODEL: RGB + Geometry + DINO + FiLM (with contact-weighted pooling)")
    else:
        print("TRAINING MODEL: RGB + Geometry + DINO (with geometric channel scaling)")
    print("=" * 60)

    if cfg.model.FILM_ENABLED:
        model = MultiModalScorerWeightedViTFiLM(
            dino_model=cfg.model.DINO_MODEL,
            geometric_vit=cfg.model.VIT_MODEL,
            freeze_vit_layers=cfg.model.FROZEN_LAYERS,
            dropout=cfg.model.DROPOUT,
            geometric_channel_scale=cfg.model.GEOMETRIC_CHANNEL_SCALE,
            t_dim=cfg.model.FILM_T_DIM,
            film_layers=cfg.model.FILM_LAYERS,
        )
    else:
        model = MultiModalScorerV2_Practical(
            dino_model=cfg.model.DINO_MODEL,
            geometric_vit=cfg.model.VIT_MODEL,
            freeze_vit_layers=cfg.model.FROZEN_LAYERS,
            dropout=cfg.model.DROPOUT,
            geometric_channel_scale=cfg.model.GEOMETRIC_CHANNEL_SCALE,
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
        print(f"\nResuming from checkpoint: {args.resume}")
        try:
            ckpt = torch.load(args.resume, map_location=cfg.training.DEVICE, weights_only=True)
        except Exception:
            ckpt = torch.load(args.resume, map_location=cfg.training.DEVICE, weights_only=False)

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        for pg in optimizer.param_groups:
            pg['lr'] = cfg.training.LEARNING_RATE

        # Move optimizer state tensors to the correct device after loading.
        # Adam's load_state_dict does not auto-move internal state buffers,
        # which causes device mismatch errors on the first optimizer.step() call.
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

        print(f"  Resuming from epoch {start_epoch}/{cfg.training.NUM_EPOCHS}")
        print(f"  Previous best val accuracy: {initial_best_val_acc:.3f}")
        print(f"  LR set to: {cfg.training.LEARNING_RATE}")

    estimate_data_needs(model, train_dataset)

    model, history = train_model(
        model,
        train_dataset,
        val_dataset,
        optimizer=optimizer,
        num_epochs=cfg.training.NUM_EPOCHS,
        batch_size=cfg.training.BATCH_SIZE,
        lr=cfg.training.LEARNING_RATE,
        weight_decay=cfg.training.WEIGHT_DECAY,
        early_stopping_patience=cfg.training.EARLY_STOPPING_PATIENCE,
        model_name="multimodal_bceW_wikiart_frozen8_resumed",
        bce_weight=cfg.loss.BCE_WEIGHT,
        ranking_weight=cfg.loss.RANKING_WEIGHT,
        boundary_weight=cfg.loss.BOUNDARY_WEIGHT,
        ranking_margin=cfg.loss.RANKING_MARGIN,
        hard_negative_weight=cfg.loss.HARD_NEGATIVE_WEIGHT,
        top_n=cfg.loss.TOP_N,
        temperature=cfg.loss.TEMPERATURE,
        max_norm=cfg.training.GRAD_CLIP_MAX_NORM,
        pos_weight_val_BCE=cfg.training.BCE_POS_WEIGHT,
        start_epoch=start_epoch,
        initial_history=initial_history,
        initial_best_val_acc=initial_best_val_acc,
        initial_patience=initial_patience,
    )

    diagnose_data_sufficiency(history)

    # ============================================================
    # Alternative model: MultiModalScorerWeightedVit
    # Uses contact-region weighted pooling of ViT patch tokens
    # in addition to the CLS token and DINO features.
    # Uncomment below to use it instead of MultiModalScorerV2_Practical.
    # ============================================================
    # model = MultiModalScorerWeightedVit(
    #     dino_model=cfg.model.DINO_MODEL,
    #     geometric_vit=cfg.model.VIT_MODEL,
    #     freeze_vit_layers=cfg.model.FROZEN_LAYERS,
    #     dropout=cfg.model.DROPOUT,
    #     geometric_channel_scale=cfg.model.GEOMETRIC_CHANNEL_SCALE,
    # )
    
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=cfg.training.LEARNING_RATE,
    #     weight_decay=cfg.training.WEIGHT_DECAY,
    # )
    
    # estimate_data_needs(model, train_dataset)
    
    # model, history = train_model(
    #     model,
    #     train_dataset,
    #     val_dataset,
    #     optimizer=optimizer,
    #     num_epochs=cfg.training.NUM_EPOCHS,
    #     batch_size=cfg.training.BATCH_SIZE,
    #     lr=cfg.training.LEARNING_RATE,
    #     weight_decay=cfg.training.WEIGHT_DECAY,
    #     early_stopping_patience=cfg.training.EARLY_STOPPING_PATIENCE,
    #     model_name="multimodal_weighted_vit",
    #     bce_weight=cfg.loss.BCE_WEIGHT,
    #     ranking_weight=cfg.loss.RANKING_WEIGHT,
    #     boundary_weight=cfg.loss.BOUNDARY_WEIGHT,
    #     ranking_margin=cfg.loss.RANKING_MARGIN,
    #     hard_negative_weight=cfg.loss.HARD_NEGATIVE_WEIGHT,
    #     top_n=cfg.loss.TOP_N,
    #     temperature=cfg.loss.TEMPERATURE,
    #     max_norm=cfg.training.GRAD_CLIP_MAX_NORM,
    #     pos_weight_val_BCE=cfg.training.BCE_POS_WEIGHT,
    # )
    
    # diagnose_data_sufficiency(history)


if __name__ == "__main__":
    main()

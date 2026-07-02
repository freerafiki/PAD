import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import json
import contextlib
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


def evaluate_single_image(model, val_loader, criterion, device, use_geom=False,
                          backbone="vit", split_pieces=False):
    model.eval()
    all_logits = []
    all_labels = []
    all_categories = []
    all_pair_keys = []
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device)
            rgb_geometric = batch["rgb_geometric"].to(device)
            labels = batch["labels"].to(device)
            categories = batch["category"]
            pair_keys = batch["pair_key"]

            if backbone == 'vit':
                if use_geom:
                    logits = model(rgb_geometric).squeeze()
                else:
                    logits = model(rgb).squeeze()
            elif backbone == 'resnet':
                guidance_map = rgb_geometric[:, 5:6]
                if split_pieces:
                    raise NotImplementedError("split_pieces not implemented yet")
                else:
                    logits = model(rgb, guidance_map).squeeze()

            loss = criterion(logits, labels.squeeze())
            val_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_categories.extend(categories)
            all_pair_keys.extend(pair_keys)

    avg_val_loss = val_loss / num_batches
    all_logits = torch.cat(all_logits)
    all_scores = torch.sigmoid(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = (all_scores >= 0.5).astype(float)

    accuracy = (all_preds == all_labels).mean()

    pos_mask = all_labels == 1.0
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    pos_accuracy = (all_preds[pos_mask] == all_labels[pos_mask]).mean() if n_pos > 0 else 0.0
    neg_accuracy = (all_preds[neg_mask] == all_labels[neg_mask]).mean() if n_neg > 0 else 0.0
    false_positive_rate = (all_preds[neg_mask] == 1.0).mean() if n_neg > 0 else 0.0

    groups = defaultdict(list)
    for i in range(len(all_scores)):
        groups[all_pair_keys[i]].append({
            'score': all_scores[i],
            'label': all_labels[i],
            'category': all_categories[i],
        })

    ranking_correct = 0
    ranking_total = 0
    total_false_positives = 0
    total_groups = 0

    for pair_key, samples in groups.items():
        scores = np.array([s['score'] for s in samples])
        labels = np.array([s['label'] for s in samples])

        pos_mask_g = labels == 1.0
        if pos_mask_g.sum() >= 1:
            best_idx = np.argmax(scores)
            if labels[best_idx] == 1.0:
                ranking_correct += 1
            ranking_total += 1

        neg_preds = (scores > 0.5) & (labels == 0.0)
        total_false_positives += neg_preds.sum()
        total_groups += 1

    ranking_accuracy = ranking_correct / ranking_total if ranking_total > 0 else 0.0
    avg_fp_per_group = total_false_positives / total_groups if total_groups > 0 else 0.0

    return {
        'val_loss': avg_val_loss,
        'val_accuracy': accuracy,
        'val_pos_accuracy': pos_accuracy,
        'val_neg_accuracy': neg_accuracy,
        'val_false_positive_rate': false_positive_rate,
        'val_ranking_accuracy': ranking_accuracy,
        'val_avg_fp_per_group': avg_fp_per_group,
    }


def plot_single_image_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    n_plots = 8
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    plot_idx = 0

    ax = axes[plot_idx]; plot_idx += 1
    ax.plot(epochs, history["train_loss"], label="Train", color="steelblue")
    ax.plot(epochs, history["val_loss"], label="Val", color="coral", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("BCE Loss")
    ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[plot_idx]; plot_idx += 1
    ax.plot(epochs, history.get("val_accuracy", []), label="Overall", color="#1f77b4", linewidth=2)
    ax.plot(epochs, history.get("val_pos_accuracy", []), label="Positive", color="#2ca02c", linewidth=2)
    ax.plot(epochs, history.get("val_neg_accuracy", []), label="Negative", color="#d62728", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Per-Image Accuracy")
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.4)

    ax = axes[plot_idx]; plot_idx += 1
    ax.plot(epochs, history.get("val_false_positive_rate", []), color="#d62728", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("FPR")
    ax.set_title("False Positive Rate")
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.4)

    ax = axes[plot_idx]; plot_idx += 1
    ax.plot(epochs, history.get("val_ranking_accuracy", []), color="#9467bd", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Per-Group: Positive Ranked #1")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.4)

    ax = axes[plot_idx]; plot_idx += 1
    ax.plot(epochs, history.get("val_avg_fp_per_group", []), color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg FP / Group")
    ax.set_title("False Positives per Group")
    ax.grid(True, alpha=0.4)

    ax = axes[plot_idx]; plot_idx += 1
    train_arr = np.array(history["train_loss"])
    val_arr = np.array(history["val_loss"])
    gap = val_arr - train_arr
    ax.plot(epochs, gap, color="darkorange", linewidth=2)
    ax.fill_between(epochs, 0, gap, alpha=0.2, color="darkorange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val - Train Loss")
    ax.set_title("Overfitting Gap")
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.4)

    ax = axes[plot_idx]; plot_idx += 1
    ax.plot(epochs, history.get("learning_rates", []), color="gray", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("Learning Rate")
    ax.grid(True, alpha=0.4)

    ax = axes[plot_idx]; plot_idx += 1
    if "val_pos_score" in history and "val_neg_score" in history:
        ax.plot(epochs, np.array(history["val_pos_score"]) - np.array(history["val_neg_score"]),
                color="#9467bd", linewidth=2)
        ax.set_ylabel("Pos - Neg Score")
    ax.set_xlabel("Epoch")
    ax.set_title("Score Margin (placeholder)")
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.4)

    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Single-Image Training History", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training history plot to {save_path}")


def save_batch_samples_figure(batch_data, save_path, max_samples=32):
    rgb = batch_data['rgb']
    labels = batch_data['labels']
    logits = batch_data['logits']
    scores = torch.sigmoid(logits).numpy()
    n = len(scores)
    n_show = min(n, max_samples)

    best_cols = min(4, n_show)
    best_rows = int(np.ceil(n_show / best_cols))

    fig, axes = plt.subplots(best_rows, best_cols, figsize=(best_cols * 3.5, best_rows * 3.5))
    axes_flat = axes.flatten() if best_rows > 1 or best_cols > 1 else [axes]

    for i in range(best_rows * best_cols):
        ax = axes_flat[i]
        if i >= n_show:
            ax.set_visible(False)
            continue
        img = rgb[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.clip(img * std + mean, 0, 1)
        label = labels[i].item()
        score = float(scores[i])
        correct = (score > 0.5) == (label == 1.0)
        color = 'green' if correct else 'red'

        ax.imshow(img)
        gt = "pos" if label == 1.0 else "neg"
        ax.set_title(f"label={gt}\nscore={score:.3f}", fontsize=8, color=color, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer=None,
    num_epochs=50,
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    use_geom=False,
    split_pieces=False,
    save_dir="checkpoints",
    model_name="model",
    backbone="vit",
    early_stopping_patience=5,
    max_norm=1.0,
    pos_weight_val_BCE=4.0,
    start_epoch=1,
    initial_history=None,
    initial_best_val_acc=0.0,
    initial_patience=0,
    log_batch_every_n=0,
    profile_steps=0,
    profile_dir=None,
):
    console = Console()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = save_dir / model_name
    log_dir.mkdir(exist_ok=True, parents=True)

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val_BCE]).to(device))

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
            "val_loss": [],
            "val_accuracy": [],
            "val_pos_accuracy": [],
            "val_neg_accuracy": [],
            "val_false_positive_rate": [],
            "val_ranking_accuracy": [],
            "val_avg_fp_per_group": [],
            "learning_rates": [],
        }

    table = Table(title=f"[bold]Training: {model_name}[/]", title_justify="left")
    table.add_column("Ep", justify="right", style="cyan", no_wrap=True)
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Acc", justify="right")
    table.add_column("Pos", justify="right")
    table.add_column("Neg", justify="right")
    table.add_column("FPR", justify="right")
    table.add_column("Rank", justify="right")
    table.add_column("FP/Gr", justify="right")
    table.add_column("LR", justify="right")
    table.add_column("", justify="center")

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("L: {task.fields[loss]:.4f}"),
        TextColumn("LR: {task.fields[lr]:.2e}"),
        console=console,
    )

    console.print(f"Training for up to [bold]{num_epochs}[/] epochs "
                  f"(patience: {early_stopping_patience})")
    if initial_best_val_acc > 0:
        console.print(f"Previous best val accuracy: [yellow]{initial_best_val_acc:.3f}[/]")
    console.print(f"Train: {len(train_loader.dataset)} samples  "
                  f"Val: {len(val_loader.dataset)} samples")
    console.print("")

    with Live(Group(progress, table), refresh_per_second=4, console=console):
        for epoch in range(start_epoch, num_epochs + 1):
            model.train()
            train_loss = 0.0
            num_batches = 0

            task = progress.add_task(
                f"Epoch {epoch}/{num_epochs}",
                total=len(train_loader),
                loss=0.0,
                lr=0.0,
            )

            should_log = (epoch == 1) or (log_batch_every_n > 0 and (epoch - 1) % log_batch_every_n == 0)
            sample_batch = None

            # Profile only the very first epoch we run in this call, and only if
            # profile_steps > 0. Uses a wait/warmup/active schedule so the first
            # (slow, cudnn-autotuning) iterations don't pollute the trace.
            do_profile = profile_steps > 0 and epoch == start_epoch
            if do_profile:
                prof_dir = Path(profile_dir) if profile_dir else (log_dir / "profiler")
                prof_dir.mkdir(exist_ok=True, parents=True)
                prof_schedule = schedule(wait=1, warmup=1, active=profile_steps, repeat=1)
                profiler_ctx = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=prof_schedule,
                    on_trace_ready=tensorboard_trace_handler(str(prof_dir)),
                    record_shapes=True,
                    with_stack=False,
                )
                console.print(f"  [yellow]Profiling first {profile_steps} steps -> {prof_dir}[/]")
            else:
                profiler_ctx = contextlib.nullcontext()

            with profiler_ctx as prof:
                for batch in train_loader:
                    rgb = batch["rgb"].to(device)
                    rgb_geometric = batch["rgb_geometric"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()

                    if backbone == 'vit':
                        if use_geom:
                            logits = model(rgb_geometric).squeeze()
                        else:
                            logits = model(rgb).squeeze()
                    elif backbone == 'resnet':
                        guidance_map = rgb_geometric[:, 5:6]
                        if split_pieces:
                            raise NotImplementedError("split_pieces not implemented yet")
                        else:
                            logits = model(rgb, guidance_map).squeeze()

                    if should_log and sample_batch is None:
                        sample_batch = {
                            'rgb': rgb.detach().cpu(),
                            'labels': labels.detach().cpu(),
                            'logits': logits.detach().cpu(),
                        }

                    loss = bce_criterion(logits, labels.squeeze())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    optimizer.step()
                    scheduler.step()

                    loss_val = loss.item()  # single sync point, reused below (was called twice before)
                    train_loss += loss_val
                    num_batches += 1

                    progress.update(task, advance=1, loss=loss_val, lr=scheduler.get_last_lr()[0])

                    if do_profile:
                        prof.step()
                        # wait(1) + warmup(1) + active(profile_steps) -> stop early once done
                        if num_batches >= 2 + profile_steps:
                            break

            if do_profile:
                console.print(f"  [yellow]Profiler trace written to {prof_dir} "
                               f"(open with: tensorboard --logdir {prof_dir})[/]")
                console.print(prof.key_averages().table(
                    sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total",
                    row_limit=15,
                ))

            progress.remove_task(task)
            avg_train_loss = train_loss / num_batches

            if sample_batch is not None:
                batch_dir = log_dir / 'batch_samples'
                batch_dir.mkdir(exist_ok=True)
                save_batch_samples_figure(sample_batch, batch_dir / f'epoch_{epoch:03d}.png')

            eval_metrics = evaluate_single_image(
                model, val_loader, bce_criterion, device, use_geom=use_geom,
                backbone=backbone, split_pieces=split_pieces,
            )

            history["train_loss"].append(avg_train_loss)
            for key in ["val_loss", "val_accuracy", "val_pos_accuracy", "val_neg_accuracy",
                         "val_false_positive_rate", "val_ranking_accuracy", "val_avg_fp_per_group"]:
                history[key].append(eval_metrics[key])
            history["learning_rates"].append(scheduler.get_last_lr()[0])

            val_acc = eval_metrics["val_accuracy"]
            is_best = val_acc > best_val_acc
            best_mark = "[bold green]BEST[/]" if is_best else ""

            table.add_row(
                str(epoch),
                f"{avg_train_loss:.4f}",
                f"{eval_metrics['val_loss']:.4f}",
                f"{val_acc:.3f}",
                f"{eval_metrics['val_pos_accuracy']:.3f}",
                f"{eval_metrics['val_neg_accuracy']:.3f}",
                f"{eval_metrics['val_false_positive_rate']:.3f}",
                f"{eval_metrics['val_ranking_accuracy']:.3f}",
                f"{eval_metrics['val_avg_fp_per_group']:.2f}",
                f"{scheduler.get_last_lr()[0]:.2e}",
                best_mark,
            )

            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "history": history,
                }
                torch.save(checkpoint, log_dir / f"{model_name}_best.pth")
                torch.save(checkpoint, log_dir / f"{model_name}_best_at_epoch_{epoch}.pth")
                console.print(f"  [green]>> Saved best model (acc: {val_acc:.3f})[/]")
            else:
                patience_counter += 1
                console.print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

            if patience_counter >= early_stopping_patience:
                console.print(f"\n[bold red]Early stopping triggered[/] after {epoch} epochs")
                console.print(f"Best validation accuracy: {best_val_acc:.3f}")
                break

    console.print(f"\n[bold]Training complete![/] Best val accuracy: {best_val_acc:.3f}")

    plot_single_image_history(history, log_dir / "history.png")

    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history

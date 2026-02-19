import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your dataset and models
from dataset_v3 import PrecomposedAlignmentDataset, collate_alignment_samples, ShuffledBatchSampler
from models import MultiModalScorerV2

def train_model(
    model,
    train_dataset,
    val_dataset,
    num_epochs=50,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    device='cuda',
    save_dir='checkpoints',
    model_name='model',
    early_stopping_patience=10  # NEW: Stop if no improvement
):
    """
    Training with early stopping for small datasets.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=ShuffledBatchSampler(train_dataset, shuffle=True, seed=42),
        collate_fn=collate_alignment_samples,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=ShuffledBatchSampler(val_dataset, shuffle=False),
        collate_fn=collate_alignment_samples,
        num_workers=4,
        pin_memory=True
    )
    
    # Loss and optimizer (UPDATED)
    criterion = nn.BCEWithLogitsLoss()  # More stable than BCE + Sigmoid
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    model = model.to(device)
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    print(f"Training {model_name} for up to {num_epochs} epochs")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Train samples: {len(train_dataset)} pairs")
    print(f"Val samples: {len(val_dataset)} pairs")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            rgb_geometric = batch['rgb_geometric'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)  # (B, 1)
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(rgb, rgb_geometric)
            
            # Loss (BCEWithLogitsLoss combines sigmoid + BCE)
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping (helps with small data)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        avg_train_loss = train_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                rgb_geometric = batch['rgb_geometric'].to(device)
                labels = batch['labels'].to(device).unsqueeze(1)
                
                logits = model(rgb, rgb_geometric)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        
        # Ranking accuracy
        val_acc, avg_pos, avg_neg = evaluate_ranking(model, val_loader, device)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.3f} | "
              f"Pos/Neg: {avg_pos:.3f}/{avg_neg:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'history': history
            }
            torch.save(checkpoint, save_dir / f'{model_name}_best.pth')
            print(f"  → Saved best model (acc: {val_acc:.3f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.3f}")
            break
    
    print("-" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.3f}")
    
    # Plot training curves
    plot_training_history(history, save_dir / f'{model_name}_history.png')
    
    return model, history


def evaluate_ranking(model, dataloader, device):
    """
    Evaluate ranking accuracy: is positive ranked first in each group?
    """
    model.eval()
    
    correct = 0
    total_groups = 0
    all_pos_scores = []
    all_neg_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device)
            rgb_geometric = batch['rgb_geometric'].to(device)
            labels = batch['labels'].to(device)
            
            # Get logits and convert to probabilities
            logits = model(rgb, rgb_geometric).squeeze()
            scores = torch.sigmoid(logits)  # Convert to [0, 1]
            
            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Find groups
            positive_indices = np.where(labels_np == 1.0)[0]
            
            for i, pos_idx in enumerate(positive_indices):
                if i < len(positive_indices) - 1:
                    next_pos_idx = positive_indices[i + 1]
                else:
                    next_pos_idx = len(scores_np)
                
                group_scores = scores_np[pos_idx:next_pos_idx]
                group_labels = labels_np[pos_idx:next_pos_idx]
                
                # Is positive ranked first?
                if group_scores[0] == group_scores.max():
                    correct += 1
                
                total_groups += 1
                
                # Collect scores
                all_pos_scores.append(group_scores[0])
                all_neg_scores.extend(group_scores[1:])
    
    accuracy = correct / total_groups if total_groups > 0 else 0.0
    avg_pos_score = np.mean(all_pos_scores) if all_pos_scores else 0.0
    avg_neg_score = np.mean(all_neg_scores) if all_neg_scores else 0.0
    
    return accuracy, avg_pos_score, avg_neg_score

def diagnose_data_sufficiency(history):
    """
    Analyze learning curves to diagnose data issues.
    """
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    print("\n=== Data Sufficiency Diagnosis ===")
    
    # Check 1: Overfitting
    final_gap = val_loss[-1] - train_loss[-1]
    if final_gap > 0.2:
        print("❌ SEVERE OVERFITTING: Val loss >> Train loss")
        print("   → Need MORE DATA or MORE REGULARIZATION")
    elif final_gap > 0.1:
        print("⚠️  Moderate overfitting")
        print("   → Could benefit from more data")
    else:
        print("✓ No major overfitting")
    
    # Check 2: Convergence
    if val_loss[-1] < val_loss[2]:
        print("✓ Model is learning (val loss decreasing)")
    else:
        print("❌ Val loss not improving")
        print("   → Model may be too complex for data size")
    
    # Check 3: Early stopping
    best_epoch = np.argmin(val_loss) + 1
    total_epochs = len(val_loss)
    
    if best_epoch < total_epochs * 0.3:
        print(f"❌ Best epoch: {best_epoch}/{total_epochs} (very early)")
        print("   → DEFINITELY need more data")
    elif best_epoch < total_epochs * 0.6:
        print(f"⚠️  Best epoch: {best_epoch}/{total_epochs}")
        print("   → Data size is marginal")
    else:
        print(f"✓ Best epoch: {best_epoch}/{total_epochs}")
        print("   → Data size seems adequate")

def estimate_data_needs(model, train_dataset):
    """
    Rough estimate of data needs based on model parameters.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== Data Needs Estimate ===")
    print(f"Trainable parameters: {total_params:,}")
    
    # Rule of thumb: 10-20 samples per parameter (conservative)
    min_samples = total_params * 10
    recommended_samples = total_params * 20
    
    print(f"Minimum data (10x params): {min_samples:,} samples")
    print(f"Recommended data (20x params): {recommended_samples:,} samples")
    
    current_samples = len(train_dataset) * 5  # Rough estimate (pairs × samples per pair)
    print(f"Current data: ~{current_samples:,} samples")
    
    if current_samples < min_samples:
        print("❌ Likely INSUFFICIENT data")
        print(f"   Need {(min_samples - current_samples):,} more samples")
    elif current_samples < recommended_samples:
        print("⚠️  Marginal data size")
        print(f"   Could benefit from {(recommended_samples - current_samples):,} more samples")
    else:
        print("✓ Data size seems adequate")



def main():
    """Main training script."""

    # Configuration
    DATA_ROOT = '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v4'
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NEGATIVES_PER_POSITIVE = 4  # Adjusted for your data size

    # DATASET PARAMETERS
    RADIUS = 30
    THRESHOLD = 30

    # DINO PARAMETERS
    DINO_MODEL = 'facebook/dinov2-base'

    print(f"Using device: {DEVICE}")

    # Load full dataset
    full_dataset = PrecomposedAlignmentDataset(
        data_root=DATA_ROOT,
        negatives_per_positive=NEGATIVES_PER_POSITIVE,
        hard_negative_ratio=0.6,
        radius = RADIUS,
        threshold = THRESHOLD
    )

    # *** CHANGED: Split by puzzles, not random ***
    train_dataset, val_dataset = PrecomposedAlignmentDataset.create_puzzle_split(
        full_dataset,
        radius = RADIUS,
        threshold = THRESHOLD,
        train_ratio=0.8,
        seed=42
    )

    print(f"\n=== Dataset Ready ===")
    print(f"Train: {len(train_dataset)} pairs")
    print(f"Val: {len(val_dataset)} pairs")

    # Verify no overlap
    train_puzzles = set(k.split('|')[0] for k in train_dataset.pair_keys)
    val_puzzles = set(k.split('|')[0] for k in val_dataset.pair_keys)
    overlap = train_puzzles & val_puzzles

    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} puzzles appear in both train and val!")
    else:
        print("✓ No puzzle overlap between train and val")


    print("\n" + "="*60)
    print("TRAINING MODEL 4: RGB + Geometry + DINO (v2)")
    print("="*60)
    # Version without cross-attention (Option A + B)
    model_v2 = MultiModalScorerV2(
        use_cross_attention=False,
        dropout=0.3  # Higher dropout for small data
    )

    # Version with cross-attention (Option A + B + C)
    # model_v3 = MultiModalScorerV2(
    #     use_cross_attention=True,
    #     dropout=0.3
    # )

    estimate_data_needs(model_v2)

    # Train
    model, history = train_model(
        model_v2,
        train_dataset,
        val_dataset,
        num_epochs=25,
        batch_size=8,
        lr=1e-4,
        weight_decay=1e-4,
        early_stopping_patience=5,
        model_name='multimodal_v2'
    )

    # Use it
    diagnose_data_sufficiency(history)    


if __name__ == '__main__':
    main()

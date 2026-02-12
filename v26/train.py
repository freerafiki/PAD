import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, train_dataset, val_dataset, 
                num_epochs=50, lr=1e-4, device='cuda'):
    """
    Train alignment scoring model.
    """
    # Custom collate function to handle variable-length batches
    def collate_fn(batch_list):
        # Each item in batch_list is a dict with multiple samples
        # Flatten into single batch
        rgb = torch.cat([item['rgb'] for item in batch_list], dim=0)
        rgb_geometric = torch.cat([item['rgb_geometric'] for item in batch_list], dim=0)
        labels = torch.cat([item['labels'] for item in batch_list], dim=0)
        difficulties = [d for item in batch_list for d in item['difficulties']]
        
        return {
            'rgb': rgb,
            'rgb_geometric': rgb_geometric,
            'labels': labels,
            'difficulties': difficulties
        }
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,  # 4 puzzles, each with ~8 samples = 32 total samples
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Loss and optimizer
    criterion = RankingLoss(margin=0.3, hard_negative_weight=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            rgb = batch['rgb'].to(device)
            rgb_geometric = batch['rgb_geometric'].to(device)
            labels = batch['labels'].to(device)
            difficulties = batch['difficulties']
            
            optimizer.zero_grad()
            
            # Forward pass (depends on model version)
            if hasattr(model, 'dino'):  # Version 3
                scores = model(rgb, rgb_geometric).squeeze()
            else:  # Version 1 or 2
                scores = model(rgb_geometric).squeeze()
            
            loss = criterion(scores, labels, difficulties)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                rgb_geometric = batch['rgb_geometric'].to(device)
                labels = batch['labels'].to(device)
                difficulties = batch['difficulties']
                
                if hasattr(model, 'dino'):
                    scores = model(rgb, rgb_geometric).squeeze()
                else:
                    scores = model(rgb_geometric).squeeze()
                
                loss = criterion(scores, labels, difficulties)
                val_loss += loss.item()
                
                # Accuracy: for each group, is the positive ranked highest?
                # Group by puzzle (assuming batch structure)
                # Simplified: binary classification accuracy
                predictions = (scores > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    return model


# Usage
if __name__ == '__main__':
    # Load your data
    # positive_samples = load_positive_alignments()
    # negative_samples = load_negative_alignments()
    
    # Create datasets
    # train_dataset = AlignmentDataset(pos_train, neg_train)
    # val_dataset = AlignmentDataset(pos_val, neg_val)
    
    # Initialize model (Version 2 for example)
    model = GeometricScorer()
    
    # Train
    trained_model = train_model(model, train_dataset, val_dataset)
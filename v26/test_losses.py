# Sanity check: does the loss behave as expected?
def test_loss():
    criterion = AdaptiveTopNRankingLoss(top_n=3, margin=0.3, temperature=0.5)
    
    # Scenario 1: Perfect (positive ranks first)
    scores = torch.tensor([0.9, 0.3, 0.2, 0.1])  # pos at rank 1
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
    difficulties = ['positive', 'negative', 'negative', 'negative']
    loss1 = criterion(scores, labels, difficulties)
    print(f"Positive at rank 1: loss = {loss1.item():.4f}")
    
    # Scenario 2: OK (positive at rank 2)
    scores = torch.tensor([0.85, 0.9, 0.3, 0.2])  # pos at rank 2
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
    difficulties = ['positive', 'negative', 'negative', 'negative']
    loss2 = criterion(scores, labels, difficulties)
    print(f"Positive at rank 2: loss = {loss2.item():.4f}")
    
    # Scenario 3: Bad (positive at rank 5)
    scores = torch.tensor([0.5, 0.9, 0.8, 0.7, 0.6])  # pos at rank 5
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    difficulties = ['positive', 'negative', 'negative', 'negative', 'negative']
    loss3 = criterion(scores, labels, difficulties)
    print(f"Positive at rank 5: loss = {loss3.item():.4f}")
    
    assert loss1 < loss2 < loss3, "Loss should increase with worse ranking!"
    print("✓ Loss behaves correctly")

test_loss()
import torch.nn as nn
import torch

class RankingLoss(nn.Module):
    """Ranking loss: positive should score higher than negatives."""

    def __init__(self, margin=0.3, hard_negative_weight=2.0):
        super().__init__()
        self.margin = margin
        self.hard_weight = hard_negative_weight

    def forward(self, scores, labels, difficulties):
        """
        Args:
            scores: (N,) predicted scores for N samples
            labels: (N,) ground truth (1.0 for positive, 0.0 for negative)
            difficulties: List of N strings ('positive', 'negative', 'hard_negative')

        Returns:
            loss: scalar tensor
        """
        # Separate positive and negative scores
        pos_mask = labels == 1.0
        neg_mask = labels == 0.0

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # For each positive, it should score higher than all negatives
        # Expand dimensions for pairwise comparison
        pos_expanded = pos_scores.unsqueeze(1)  # (n_pos, 1)
        neg_expanded = neg_scores.unsqueeze(0)  # (1, n_neg)

        # Hinge loss: max(0, margin + neg_score - pos_score)
        # We want: pos_score > neg_score + margin
        pairwise_losses = torch.clamp(self.margin + neg_expanded - pos_expanded, min=0)

        # Weight hard negatives more heavily
        weights = torch.ones(len(neg_scores), device=scores.device)

        # Find hard negatives and increase their weight
        neg_difficulties = [d for d, l in zip(difficulties, labels) if l == 0.0]
        for i, diff in enumerate(neg_difficulties):
            if diff == 'hard_negative':
                weights[i] = self.hard_weight

        # Apply weights
        weighted_losses = pairwise_losses * weights.unsqueeze(0)

        return weighted_losses.mean()

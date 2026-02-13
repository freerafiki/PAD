import torch.nn as nn
import torch

class RankingLoss(nn.Module):
    """Contrastive ranking loss for alignment scoring."""

    def __init__(self, margin=0.5, hard_negative_weight=2.0):
        super().__init__()
        self.margin = margin
        self.hard_weight = hard_negative_weight

    def forward(self, scores, labels, difficulties):
        """
        Args:
            scores: (N,) predicted scores
            labels: (N,) ground truth (1.0 for positive, 0.0 for negative)
            difficulties: (N,) list of difficulty strings

        Returns:
            loss: scalar
        """
        # Separate positive and negative scores
        pos_mask = labels == 1.0
        neg_mask = labels == 0.0

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=scores.device)

        # Ranking loss: positive should be higher than negatives by margin
        # For each positive, compare with all negatives
        pos_scores_expanded = pos_scores.unsqueeze(1)  # (n_pos, 1)
        neg_scores_expanded = neg_scores.unsqueeze(0)  # (1, n_neg)

        # Hinge loss: max(0, margin + neg_score - pos_score)
        losses = torch.clamp(self.margin + neg_scores_expanded - pos_scores_expanded, min=0)

        # Weight hard negatives more
        weights = torch.ones_like(neg_scores)
        for i, diff in enumerate(difficulties):
            if diff == 'hard':
                weights[i] = self.hard_weight

        weighted_losses = losses * weights.unsqueeze(0)

        return weighted_losses.mean()


# Alternative: InfoNCE (contrastive) loss
class InfoNCELoss(nn.Module):
    """InfoNCE loss for alignment scoring."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores, labels):
        """
        Args:
            scores: (N,) scores for [1 positive, K negatives]
            labels: (N,) should be [1, 0, 0, ..., 0]
        """
        # Positive is first, negatives are rest
        logits = scores / self.temperature

        # Cross entropy: positive should have highest probability
        loss = -torch.log_softmax(logits, dim=0)[0]

        return loss

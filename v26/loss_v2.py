import torch.nn as nn
import torch


import torch
import torch.nn as nn
import numpy as np


class TopNRankingLoss(nn.Module):
    """
    Ranking loss that penalizes when positive is not in top N positions.
    
    Penalty structure:
    - Position 1 (best): No penalty
    - Position 2 to N: Increasing penalty
    - Position > N: Maximum penalty
    
    This encourages the model to put the positive in top N, with preference for top 1.
    """
    
    def __init__(self, 
                 top_n=3, 
                 margin=0.3,
                 hard_negative_weight=2.0,
                 position_penalty_scale=1.0):
        """
        Args:
            top_n: Target top-N positions (e.g., 3 means we want positive in top 3)
            margin: Margin for ranking loss
            hard_negative_weight: Weight for hard negatives
            position_penalty_scale: How much to penalize based on position
                                   (higher = stronger penalty for being far from top)
        """
        super().__init__()
        self.top_n = top_n
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
        self.position_penalty_scale = position_penalty_scale
    
    def forward(self, scores, labels, difficulties):
        """
        Args:
            scores: (B,) predicted scores for B samples in batch
            labels: (B,) ground truth (1.0 for positive, 0.0 for negative)
            difficulties: List[str] of length B
        
        Returns:
            loss: scalar tensor
        """
        # Separate positive and negative indices
        pos_mask = labels == 1.0
        neg_mask = labels == 0.0
        
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        total_loss = 0.0
        num_groups = 0
        
        # Process each positive (each group)
        for i, pos_idx in enumerate(pos_indices):
            # Find negatives in this group
            # Assumption: groups are contiguous, next positive marks new group
            if i < len(pos_indices) - 1:
                group_end = pos_indices[i + 1]
            else:
                group_end = len(scores)
            
            # Get negatives in this group
            group_neg_mask = (neg_indices >= pos_idx) & (neg_indices < group_end)
            group_neg_indices = neg_indices[group_neg_mask]
            
            if len(group_neg_indices) == 0:
                continue
            
            pos_score = scores[pos_idx]
            neg_scores = scores[group_neg_indices]
            
            # Get difficulties for weighting
            group_difficulties = [difficulties[idx.item()] for idx in group_neg_indices]
            
            # ============================================
            # Position-Aware Ranking Loss
            # ============================================
            
            # Compute how many negatives score higher than positive
            higher_negs = (neg_scores > pos_score).float()
            position = higher_negs.sum() + 1  # Position of positive (1-indexed)
            
            # Position penalty: grows from 0 (position 1) to max (position > N)
            if position <= 1:
                position_penalty = 0.0
            elif position <= self.top_n:
                # Linear growth from position 2 to N
                # position 2: penalty = 1/N
                # position N: penalty = 1
                position_penalty = (position - 1) / (self.top_n - 1)
            else:
                # Beyond top-N: maximum penalty
                position_penalty = 1.0 + (position - self.top_n) * 0.1  # Grows slowly beyond N
            
            position_penalty = position_penalty * self.position_penalty_scale
            
            # ============================================
            # Standard Pairwise Ranking Loss
            # ============================================
            
            # For each negative, compute hinge loss
            neg_scores_expanded = neg_scores.unsqueeze(0)  # (1, K)
            pos_score_expanded = pos_score.unsqueeze(0).unsqueeze(1)  # (1, 1)
            
            # Hinge loss: max(0, margin + neg_score - pos_score)
            pairwise_losses = torch.clamp(self.margin + neg_scores_expanded - pos_score_expanded, min=0)
            
            # Weight by difficulty
            weights = torch.ones(len(neg_scores), device=scores.device)
            for j, diff in enumerate(group_difficulties):
                if diff == 'hard_negative':
                    weights[j] = self.hard_negative_weight
            
            weighted_losses = pairwise_losses.squeeze(0) * weights
            
            # ============================================
            # Combine: Pairwise loss * Position penalty
            # ============================================
            
            # Base ranking loss
            base_loss = weighted_losses.mean()
            
            # Apply position penalty multiplier
            group_loss = base_loss * (1.0 + position_penalty)
            
            total_loss += group_loss
            num_groups += 1
        
        return total_loss / num_groups if num_groups > 0 else torch.tensor(0.0, device=scores.device)


class AdaptiveTopNRankingLoss(nn.Module):
    """
    Advanced version: penalty curve is differentiable and smooth.
    """
    
    def __init__(self, 
                 top_n=3, 
                 margin=0.3,
                 hard_negative_weight=2.0,
                 temperature=1.0):
        """
        Args:
            top_n: Target top-N positions
            margin: Margin for ranking loss
            hard_negative_weight: Weight for hard negatives
            temperature: Controls smoothness of position penalty (lower = sharper)
        """
        super().__init__()
        self.top_n = top_n
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
        self.temperature = temperature
    
    def forward(self, scores, labels, difficulties):
        """Differentiable version using soft ranking."""
        pos_mask = labels == 1.0
        neg_mask = labels == 0.0
        
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        total_loss = 0.0
        num_groups = 0
        
        for i, pos_idx in enumerate(pos_indices):
            # Find group
            if i < len(pos_indices) - 1:
                group_end = pos_indices[i + 1]
            else:
                group_end = len(scores)
            
            group_neg_mask = (neg_indices >= pos_idx) & (neg_indices < group_end)
            group_neg_indices = neg_indices[group_neg_mask]
            
            if len(group_neg_indices) == 0:
                continue
            
            pos_score = scores[pos_idx]
            neg_scores = scores[group_neg_indices]
            
            # ============================================
            # Soft Position Estimate (Differentiable)
            # ============================================
            
            # Soft count of how many negatives are "higher" using sigmoid
            # This is differentiable unlike hard thresholding
            score_diffs = neg_scores - pos_score  # (K,)
            soft_higher = torch.sigmoid(score_diffs / self.temperature)  # (K,)
            soft_position = soft_higher.sum() + 1.0
            
            # Smooth position penalty
            # Using smooth exponential curve
            if self.top_n > 1:
                # Penalty grows exponentially beyond top-N
                normalized_pos = (soft_position - 1.0) / (self.top_n - 1.0)
                position_penalty = torch.exp(torch.clamp(normalized_pos - 1.0, min=0.0)) - 1.0
            else:
                position_penalty = soft_position - 1.0
            
            # ============================================
            # Pairwise Ranking Loss
            # ============================================
            
            group_difficulties = [difficulties[idx.item()] for idx in group_neg_indices]
            
            pairwise_losses = torch.clamp(self.margin + neg_scores - pos_score, min=0)
            
            # Weights
            weights = torch.ones_like(neg_scores)
            for j, diff in enumerate(group_difficulties):
                if diff == 'hard_negative':
                    weights[j] = self.hard_negative_weight
            
            weighted_losses = pairwise_losses * weights
            base_loss = weighted_losses.mean()
            
            # ============================================
            # Combine with position penalty
            # ============================================
            
            group_loss = base_loss * (1.0 + position_penalty)
            
            total_loss += group_loss
            num_groups += 1
        
        return total_loss / num_groups if num_groups > 0 else torch.tensor(0.0, device=scores.device)

class RankingLoss(nn.Module):
    """Ranking loss: positive should score higher than negatives.
    
    Includes penalty for positive not being in top-N positions.
    """

    def __init__(self, margin=0.3, hard_negative_weight=2.0, top_n_penalty_threshold=5, top_n_penalty_weight=2.0):
        """
        Args:
            margin: Margin for hinge loss (positive should be above negative by this margin)
            hard_negative_weight: Weight multiplier for hard negatives
            top_n_penalty_threshold: N for top-N penalty (default 5)
            top_n_penalty_weight: Additional penalty weight when positive is not in top-N
        """
        super().__init__()
        self.margin = margin
        self.hard_weight = hard_negative_weight
        self.top_n = top_n_penalty_threshold
        self.top_n_penalty = top_n_penalty_weight

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

        # Apply weights for hard negatives
        weighted_losses = pairwise_losses * weights.unsqueeze(0)

        # Top-N penalty: count how many negatives score higher than positive
        # If more than (N-1) negatives score higher, positive is not in top-N
        for i, pos_score in enumerate(pos_scores):
            # Count negatives that score higher than this positive
            num_higher_negatives = (neg_scores > pos_score).sum().item()
            
            # If positive would be ranked below position N, add extra penalty
            if num_higher_negatives >= self.top_n:
                # Scale penalty by how far outside top-N the positive is
                penalty_factor = self.top_n_penalty * (1 + (num_higher_negatives - self.top_n + 1) * 0.5)
                weighted_losses[i, :] *= penalty_factor

        return weighted_losses.mean()

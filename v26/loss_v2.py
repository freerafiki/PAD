import torch.nn as nn
import torch
import numpy as np


class BoundaryConsistencyLoss(nn.Module):
    """
    Penalize visual discontinuity across piece boundaries.
    
    Measures feature similarity in the contact region between two pieces.
    """
    
    def __init__(self, feature_extractor='dino', consistency_weight=1.0):
        """
        Args:
            feature_extractor: 'dino' or 'vit' - which features to use
            consistency_weight: How much to weight this loss
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.consistency_weight = consistency_weight
    
    def extract_boundary_features(self, model, rgb, rgb_geometric, contact_mask):
        """
        Extract features in the contact region for each piece.
        
        Args:
            model: Your trained model
            rgb: (B, 3, H, W)
            rgb_geometric: (B, 6, H, W)
            contact_mask: (B, H, W) - binary mask of contact region
        
        Returns:
            features_A: (B, D) - average features on piece A side of boundary
            features_B: (B, D) - average features on piece B side of boundary
        """
        B = rgb.shape[0]
        
        # Get DINO features (use frozen DINO for consistency)
        if self.feature_extractor == 'dino':
            with torch.no_grad():
                outputs = model.dino(rgb, output_hidden_states=True)
                # Get patch features (not pooled)
                patch_features = outputs.last_hidden_state[:, 1:, :]  # (B, num_patches, 768)
        elif self.feature_extractor == 'vit':
            # Use geometric ViT
            vit_input = model.projection(rgb_geometric) if hasattr(model, 'projection') else rgb_geometric[:, :3]
            outputs = model.geometric_vit(vit_input, output_hidden_states=True)
            patch_features = outputs.last_hidden_state[:, 1:, :]
        
        # Reshape to spatial grid (14x14 for ViT-Base)
        num_patches_side = int(np.sqrt(patch_features.shape[1]))
        D = patch_features.shape[2]
        patch_features = patch_features.reshape(B, num_patches_side, num_patches_side, D)
        
        # Upsample contact mask to patch resolution
        contact_mask_resized = F.interpolate(
            contact_mask.unsqueeze(1).float(),
            size=(num_patches_side, num_patches_side),
            mode='bilinear'
        ).squeeze(1)  # (B, 14, 14)
        
        # Get piece masks
        mask_A = rgb_geometric[:, 3]  # Proximity to A
        mask_B = rgb_geometric[:, 4]  # Proximity to B
        
        # Resize to patch resolution
        mask_A_resized = F.interpolate(
            mask_A.unsqueeze(1),
            size=(num_patches_side, num_patches_side),
            mode='bilinear'
        ).squeeze(1) > 0.5
        
        mask_B_resized = F.interpolate(
            mask_B.unsqueeze(1),
            size=(num_patches_side, num_patches_side),
            mode='bilinear'
        ).squeeze(1) > 0.5
        
        # Extract features near boundary on each side
        features_A_list = []
        features_B_list = []
        
        for b in range(B):
            # Contact region for this sample
            contact_b = contact_mask_resized[b] > 0.3
            
            # Piece A side: contact region AND on piece A
            boundary_A = contact_b & mask_A_resized[b]
            
            # Piece B side: contact region AND on piece B
            boundary_B = contact_b & mask_B_resized[b]
            
            if boundary_A.sum() > 0:
                feats_A = patch_features[b][boundary_A].mean(dim=0)  # (D,)
            else:
                feats_A = torch.zeros(D, device=rgb.device)
            
            if boundary_B.sum() > 0:
                feats_B = patch_features[b][boundary_B].mean(dim=0)  # (D,)
            else:
                feats_B = torch.zeros(D, device=rgb.device)
            
            features_A_list.append(feats_A)
            features_B_list.append(feats_B)
        
        features_A = torch.stack(features_A_list)  # (B, D)
        features_B = torch.stack(features_B_list)  # (B, D)
        
        return features_A, features_B
    
    def forward(self, model, rgb, rgb_geometric, labels):
        """
        Compute boundary consistency loss.
        
        For positive samples: features should be similar across boundary
        For negative samples: we don't care (or penalize similarity)
        """
        # Extract contact region from geometric features
        contact_mask = rgb_geometric[:, 5]  # Channel 5 is contact region
        
        # Get boundary features from both sides
        features_A, features_B = self.extract_boundary_features(
            model, rgb, rgb_geometric, contact_mask
        )
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(features_A, features_B, dim=1)  # (B,)
        
        # For positive samples: maximize similarity
        # For negative samples: minimize similarity (or ignore)
        pos_mask = labels == 1.0
        neg_mask = labels == 0.0
        
        loss = 0.0
        
        if pos_mask.sum() > 0:
            # Positive: want high similarity (close to 1)
            pos_similarity = similarity[pos_mask]
            pos_loss = (1.0 - pos_similarity).mean()
            loss += pos_loss
        
        if neg_mask.sum() > 0:
            # Negative: want low similarity (close to 0)
            # Use hinge loss: only penalize if similarity is too high
            neg_similarity = similarity[neg_mask]
            neg_loss = torch.clamp(neg_similarity - 0.3, min=0).mean()  # Margin at 0.3
            loss += neg_loss * 0.5  # Weight negatives less
        
        return loss * self.consistency_weight

class ContrastiveBoundaryLoss(nn.Module):
    """
    Contrastive loss for boundary features.
    
    Pull together: features across boundary in positive samples
    Push apart: features across boundary in negative samples
    """
    
    def __init__(self, temperature=0.1, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, model, rgb, rgb_geometric, labels):
        """
        Contrastive loss in the boundary region.
        """
        # Extract boundary features (same as before)
        contact_mask = rgb_geometric[:, 5]
        
        with torch.no_grad():
            outputs = model.dino(rgb, output_hidden_states=True)
            patch_features = outputs.last_hidden_state[:, 1:, :]
        
        # ... [same feature extraction as BoundaryConsistencyLoss]
        
        features_A, features_B = self.extract_boundary_features(...)
        
        # Normalize features
        features_A = F.normalize(features_A, dim=1)
        features_B = F.normalize(features_B, dim=1)
        
        # Compute all pairwise similarities
        # For each positive, compare with all negatives
        pos_mask = labels == 1.0
        neg_mask = labels == 0.0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=rgb.device)
        
        # For positives: similarity should be high
        pos_sim = (features_A[pos_mask] * features_B[pos_mask]).sum(dim=1)
        
        # For negatives: similarity should be low
        neg_sim = (features_A[neg_mask] * features_B[neg_mask]).sum(dim=1)
        
        # Contrastive loss
        pos_loss = -torch.log(torch.exp(pos_sim / self.temperature).mean() + 1e-8)
        neg_loss = torch.clamp(self.margin - neg_sim, min=0).mean()
        
        return pos_loss + neg_loss

class PerceptualBoundaryLoss(nn.Module):
    """
    Simple perceptual loss: RGB values should be similar across boundary.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, rgb, rgb_geometric, labels):
        """
        Measure RGB similarity across contact boundary.
        """
        contact_mask = rgb_geometric[:, 5]  # (B, H, W)
        mask_A = rgb_geometric[:, 3] > 0.5  # Piece A region
        mask_B = rgb_geometric[:, 4] > 0.5  # Piece B region
        
        B = rgb.shape[0]
        losses = []
        
        for b in range(B):
            # Skip if not positive
            if labels[b] != 1.0:
                continue
            
            # Contact region
            contact_b = contact_mask[b] > 0.3
            
            # Dilate to get regions on both sides
            from scipy.ndimage import binary_dilation
            contact_dilated = torch.from_numpy(
                binary_dilation(contact_b.cpu().numpy(), iterations=3)
            ).to(rgb.device)
            
            # Side A: contact region intersect piece A
            side_A = contact_dilated & mask_A[b]
            
            # Side B: contact region intersect piece B
            side_B = contact_dilated & mask_B[b]
            
            if side_A.sum() == 0 or side_B.sum() == 0:
                continue
            
            # Get RGB values
            rgb_A = rgb[b][:, side_A].mean(dim=1)  # (3,)
            rgb_B = rgb[b][:, side_B].mean(dim=1)  # (3,)
            
            # L2 distance
            color_diff = ((rgb_A - rgb_B) ** 2).mean()
            losses.append(color_diff)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=rgb.device)
        
        return torch.stack(losses).mean()

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
    Top-N ranking loss that works with explicit group boundaries.
    """
    
    def __init__(self, 
                 top_n=5, 
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
    
    def forward(self, scores, labels, difficulties, group_sizes=None):
        """
        Differentiable version using soft ranking.
        Args:
            scores: (B,) predicted scores
            labels: (B,) ground truth
            difficulties: List[str] of length B
            group_sizes: List[int] - sizes of each group (NEW!)
        """
        if group_sizes is None:
            # Fallback: try to infer groups from positives
            return self._forward_inferred_groups(scores, labels, difficulties)
        
        # Process each group explicitly
        total_loss = 0.0
        num_groups = 0
        
        start_idx = 0
        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size
            
            # Extract group
            group_scores = scores[start_idx:end_idx]
            group_labels = labels[start_idx:end_idx]
            group_difficulties = difficulties[start_idx:end_idx]
            
            # Validate: should have exactly 1 positive
            pos_mask = group_labels == 1.0
            if pos_mask.sum() != 1:
                print(f"⚠️  Warning: Group {group_idx} has {pos_mask.sum().item()} positives")
                start_idx = end_idx
                continue
            
            # Get positive and negative scores
            pos_idx = torch.where(pos_mask)[0][0]
            neg_mask = ~pos_mask
            
            pos_score = group_scores[pos_idx]
            neg_scores = group_scores[neg_mask]
            
            if len(neg_scores) == 0:
                start_idx = end_idx
                continue
            
            # ============================================
            # Soft Position Estimate
            # ============================================
            
            score_diffs = neg_scores - pos_score
            soft_higher = torch.sigmoid(score_diffs / self.temperature)
            soft_position = soft_higher.sum() + 1.0
            
            # Position penalty (exponential beyond top-N)
            if self.top_n > 1:
                normalized_pos = (soft_position - 1.0) / (self.top_n - 1.0)
                position_penalty = torch.exp(torch.clamp(normalized_pos - 1.0, min=0.0)) - 1.0
            else:
                position_penalty = soft_position - 1.0
            
            # ============================================
            # Pairwise Ranking Loss
            # ============================================
            
            pairwise_losses = torch.clamp(self.margin + neg_scores - pos_score, min=0)
            
            # Weight by difficulty
            weights = torch.ones_like(neg_scores)
            neg_difficulties = [group_difficulties[i] for i in range(group_size) if i != pos_idx.item()]
            
            for i, diff in enumerate(neg_difficulties):
                if diff == 'hard_negative':
                    weights[i] = self.hard_negative_weight
            
            weighted_losses = pairwise_losses * weights
            base_loss = weighted_losses.mean()
            
            # Combine with position penalty
            group_loss = base_loss * (1.0 + position_penalty)
            
            total_loss += group_loss
            num_groups += 1
            
            start_idx = end_idx
        
        return total_loss / num_groups if num_groups > 0 else torch.tensor(0.0, device=scores.device, requires_grad=True)
    
    def _forward_inferred_groups(self, scores, labels, difficulties):
        """
        Fallback: infer groups from positives (old behavior).
        Use this if group_sizes not provided.

        This is very likely to fail!
        """
        print("\n\n WARNING: \n\n")
        print("`group_sizes` not provided, so groups are inferred based on positive samples.")
        print("This is likely to produce mixed groups within batches if shuffle is used\n\n")
        pos_indices = torch.where(labels == 1.0)[0]
        
        if len(pos_indices) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        total_loss = 0.0
        num_groups = 0
        
        for i, pos_idx in enumerate(pos_indices):
            # Find group boundaries
            if i < len(pos_indices) - 1:
                group_end = pos_indices[i + 1]
            else:
                group_end = len(scores)
            
            # Extract group
            group_scores = scores[pos_idx:group_end]
            group_labels = labels[pos_idx:group_end]
            group_difficulties = difficulties[pos_idx:group_end]
            
            # ... [same processing as above]
            # this code below is copied from above
            # TODO: re-use with function!
            # Validate: should have exactly 1 positive
            pos_mask = group_labels == 1.0
            if pos_mask.sum() != 1:
                print(f"⚠️  Warning: Group {group_idx} has {pos_mask.sum().item()} positives")
                continue
            
            # Get positive and negative scores
            pos_idx = torch.where(pos_mask)[0][0]
            neg_mask = ~pos_mask
            
            pos_score = group_scores[pos_idx]
            neg_scores = group_scores[neg_mask]
            
            if len(neg_scores) == 0:
                continue
            
            # ============================================
            # Soft Position Estimate
            # ============================================
            
            score_diffs = neg_scores - pos_score
            soft_higher = torch.sigmoid(score_diffs / self.temperature)
            soft_position = soft_higher.sum() + 1.0
            
            # Position penalty (exponential beyond top-N)
            if self.top_n > 1:
                normalized_pos = (soft_position - 1.0) / (self.top_n - 1.0)
                position_penalty = torch.exp(torch.clamp(normalized_pos - 1.0, min=0.0)) - 1.0
            else:
                position_penalty = soft_position - 1.0
            
            # ============================================
            # Pairwise Ranking Loss
            # ============================================
            
            pairwise_losses = torch.clamp(self.margin + neg_scores - pos_score, min=0)
            
            # Weight by difficulty
            weights = torch.ones_like(neg_scores)
            neg_difficulties = [group_difficulties[i] for i in range(group_size) if i != pos_idx.item()]
            
            for i, diff in enumerate(neg_difficulties):
                if diff == 'hard_negative':
                    weights[i] = self.hard_negative_weight
            
            weighted_losses = pairwise_losses * weights
            base_loss = weighted_losses.mean()
            
            # Combine with position penalty
            group_loss = base_loss * (1.0 + position_penalty)
            
            total_loss += group_loss
            num_groups += 1
                    
        return total_loss / num_groups if num_groups > 0 else torch.tensor(0.0, device=scores.device, requires_grad=True)

# class AdaptiveTopNRankingLoss(nn.Module):
#     """
#     Advanced version: penalty curve is differentiable and smooth.
#     """
    
#     def __init__(self, 
#                  top_n=3, 
#                  margin=0.3,
#                  hard_negative_weight=2.0,
#                  temperature=1.0):
#         """
#         Args:
#             top_n: Target top-N positions
#             margin: Margin for ranking loss
#             hard_negative_weight: Weight for hard negatives
#             temperature: Controls smoothness of position penalty (lower = sharper)
#         """
#         super().__init__()
#         self.top_n = top_n
#         self.margin = margin
#         self.hard_negative_weight = hard_negative_weight
#         self.temperature = temperature
    
#     def forward(self, scores, labels, difficulties):
#         """"""
#         pos_mask = labels == 1.0
#         neg_mask = labels == 0.0
        
#         pos_indices = torch.where(pos_mask)[0]
#         neg_indices = torch.where(neg_mask)[0]
        
#         if len(pos_indices) == 0 or len(neg_indices) == 0:
#             return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
#         total_loss = 0.0
#         num_groups = 0
        
#         for i, pos_idx in enumerate(pos_indices):
#             # Find group
#             if i < len(pos_indices) - 1:
#                 group_end = pos_indices[i + 1]
#             else:
#                 group_end = len(scores)
            
#             group_neg_mask = (neg_indices >= pos_idx) & (neg_indices < group_end)
#             group_neg_indices = neg_indices[group_neg_mask]
            
#             if len(group_neg_indices) == 0:
#                 continue
            
#             pos_score = scores[pos_idx]
#             neg_scores = scores[group_neg_indices]
            
#             # ============================================
#             # Soft Position Estimate (Differentiable)
#             # ============================================
            
#             # Soft count of how many negatives are "higher" using sigmoid
#             # This is differentiable unlike hard thresholding
#             score_diffs = neg_scores - pos_score  # (K,)
#             soft_higher = torch.sigmoid(score_diffs / self.temperature)  # (K,)
#             soft_position = soft_higher.sum() + 1.0
            
#             # Smooth position penalty
#             # Using smooth exponential curve
#             if self.top_n > 1:
#                 # Penalty grows exponentially beyond top-N
#                 normalized_pos = (soft_position - 1.0) / (self.top_n - 1.0)
#                 position_penalty = torch.exp(torch.clamp(normalized_pos - 1.0, min=0.0)) - 1.0
#             else:
#                 position_penalty = soft_position - 1.0
            
#             # ============================================
#             # Pairwise Ranking Loss
#             # ============================================
            
#             group_difficulties = [difficulties[idx.item()] for idx in group_neg_indices]
            
#             pairwise_losses = torch.clamp(self.margin + neg_scores - pos_score, min=0)
            
#             # Weights
#             weights = torch.ones_like(neg_scores)
#             for j, diff in enumerate(group_difficulties):
#                 if diff == 'hard_negative':
#                     weights[j] = self.hard_negative_weight
            
#             weighted_losses = pairwise_losses * weights
#             base_loss = weighted_losses.mean()
            
#             # ============================================
#             # Combine with position penalty
#             # ============================================
            
#             group_loss = base_loss * (1.0 + position_penalty)
            
#             total_loss += group_loss
#             num_groups += 1
        
#         return total_loss / num_groups if num_groups > 0 else torch.tensor(0.0, device=scores.device)

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

import matplotlib.pyplot as plt
import numpy as np

def visualize_position_penalties():
    """Show how penalty grows with position."""
    positions = np.arange(1, 11)
    top_n = 3
    
    # Version 1: Discrete penalty
    penalties_v1 = []
    for pos in positions:
        if pos <= 1:
            penalty = 0.0
        elif pos <= top_n:
            penalty = (pos - 1) / (top_n - 1)
        else:
            penalty = 1.0 + (pos - top_n) * 0.1
        penalties_v1.append(penalty)
    
    # Version 2: Smooth exponential
    penalties_v2 = []
    for pos in positions:
        normalized_pos = (pos - 1.0) / (top_n - 1.0)
        penalty = np.exp(max(normalized_pos - 1.0, 0.0)) - 1.0
        penalties_v2.append(penalty)
    
    plt.figure(figsize=(10, 6))
    plt.plot(positions, penalties_v1, 'o-', label='TopNRankingLoss (discrete)', linewidth=2, markersize=8)
    plt.plot(positions, penalties_v2, 's-', label='AdaptiveTopNRankingLoss (smooth)', linewidth=2, markersize=8)
    
    plt.axvline(x=top_n, color='red', linestyle='--', alpha=0.5, label=f'Top-{top_n} cutoff')
    plt.axhspan(0, 0.3, alpha=0.1, color='green', label='Low penalty zone')
    plt.axhspan(1.0, max(penalties_v1[-1], penalties_v2[-1]), alpha=0.1, color='red', label='High penalty zone')
    
    plt.xlabel('Position of Positive Sample', fontsize=12)
    plt.ylabel('Position Penalty Multiplier', fontsize=12)
    plt.title(f'Position Penalty Curve (Top-N = {top_n})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(positions)
    
    plt.savefig('position_penalty_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_position_penalties()
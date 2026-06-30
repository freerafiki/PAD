# Single Image with CNN

Here the code to train CNN-based model on the single image classification task. 

Beyond *standard* CNN, there is code to experiment with a *guidance map* (the contact region in our case), see below for more discussion.

## SPADE vs Guidance Gated Convolutions

| | SPADE | Guidance Gated Conv2d |
|---|---|---|
| **Mechanism** | Modulates normalization (BN/IN) per-pixel | Modulates feature values per-pixel via soft gate |
| **Guidance role** | Predicts scale+shift for every feature channel | Directly opens/closes feature gates |
| **Integration** | Replaces ResBlock's BN — drop-in | Replaces Conv2d — more structural change |
| **Extra params** | ~2×(hidden_nc × C) per block (~small) | ~C² per gated layer (~small) |
| **Works on encoders** | Yes (designed for generators but trivially adapted) | Yes (designed for any setting) |
| **License** | CC BY-NC-SA 4.0 (NVlabs) — academic only | Community impls — check individual repo |
| **Batch size sensitivity** | BN inside SPADE is sensitive to small batch | No BN inside gate — safer at batch size 8–16 |

## Discussion on Batches and Normalization

BatchNorm vs InstanceNorm in SPADE — the real story
This is worth understanding deeply, because the right choice depends on what SPADE is actually doing inside.

### What the base norm layer does in SPADE

The key insight is that in SPADE, the normalization layer is used with affine=False — it contributes no learned γ or β of its own. Those are entirely overridden by the guidance-map-derived parameters. The base norm layer only does one thing: zero-mean, unit-variance normalization of the feature map before the guided modulation is applied. So the question becomes: "what statistics should we normalize by?"
```
Input features x
       │
  ┌────▼─────────────────────────────┐
  │  param_free_norm(x)              │  ← only zero-centers + unit-variance
  │  (affine=False, no γ/β here)     │
  └────────────┬─────────────────────┘
               │
  guidance_map ──→ MLP conv → gamma [B,C,H,W]
                          └─→ beta  [B,C,H,W]
               │
  out = x_norm * (1 + gamma) + beta  ← ALL learned affine comes from guidance
```
This means the base norm's statistics directly set the scale at which the guidance-map modulation operates.

### BatchNorm: what it computes and why it can fight SPADE

BatchNorm computes statistics across a batch — with large batch sizes, the batch normalized models are trained effectively due to more accurate estimation of the batch statistics. Using small batch sizes, BatchNorm causes reduction in model accuracy due to dramatic fluctuations in the batch statistics. GitHub
But there is a deeper, more fundamental issue specifically for SPADE: BN normalizes across samples in the batch. After BN, every sample has been "pulled toward" the batch mean. Then SPADE immediately applies per-sample modulation (each sample has its own guidance map → its own γ and β). This creates a conceptual tug-of-war:
```
BN: pulls sample A and sample B toward a shared mean  ←──────┐ fighting
SPADE: pushes sample A and sample B apart via their own γ/β  ←──────┘
```
In generative use (the original SPADE paper's context), this was partially handled by SyncBatchNorm across 8 GPUs — with 128+ images per batch, BN statistics are stable enough that this tension is manageable. But for a discriminative encoder where your per-sample guidance maps are structurally diverse (different contact regions per puzzle pair), BN's cross-sample mixing actively corrupts the signal before SPADE can recover it.

### InstanceNorm: why it's the correct default for SPADE

InstanceNorm computes statistics of individual samples — the goal is to gather statistics of a single image and how the style can affect the result. Using BatchNorm's batch-level statistics can lead to the loss of high-frequency information, hence the choice to use InstanceNorm for per-sample conditioning. Internet Archive

With InstanceNorm inside SPADE:
```
IN: normalizes each sample A independently    ─→ sample A has zero mean, unit var
IN: normalizes each sample B independently    ─→ sample B has zero mean, unit var
SPADE: applies sample A's gamma/beta to A   ─→ sample A modulated by its contact map
SPADE: applies sample B's gamma/beta to B   ─→ sample B modulated by its contact map
```
No cross-contamination. The per-sample modulation from SPADE lands on a per-sample normalized base, which is exactly what you want.

InstanceNorm discards global statistics and with its learned affine parameters intends to close the style gap between each sample — it resists the effect of style discrepancy but can damage discrimination simultaneously. Considering the advantages of BN and IN, IBN-Net integrates both IN and BN as building blocks to extract style-invariable features while maintaining discriminative power. Paperspace

That last point is the one nuance against pure InstanceNorm: it can reduce feature discriminability. For a classifier/encoder (vs a generator), you typically want the network to maintain inter-sample discriminative statistics — which BN preserves and IN erodes. This is why IN is the standard in style transfer and GANs (you want style-invariance) but BN is the standard in ResNets for classification (you want to maintain the discriminative signal across the dataset).

### Your situation: large batches, classification task

With large batches (≥32) and a classification objective (compatibility scoring), the trade-offs shift:

| | **Batch size dependency** | **Per-sample isolation** | **Discriminative power** | **Verdict for you** |
|---|---|---|---|
| **BatchNorm** | Need ≥32 for stable stats | None — mixes samples | ✅ Best | Works if batch ≥32, but fights SPADE |
| **InstanceNorm** | None | ✅ Perfect | Weakened | Best for SPADE conditioning specifically |
| **GroupNorm** | None | ✅ Partial (within groups) | Good | Best all-round compromise |

GroupNorm is the pragmatic recommendation for your case. GroupNorm offers a balance between BatchNorm and LayerNorm, providing the flexibility of LayerNorm while maintaining the stability of BatchNorm. It does not depend on the batch size, making it suitable for tasks with small batch sizes, and works well for object detection and segmentation. With 32 groups, GroupNorm approximates InstanceNorm's per-sample behavior while preserving some cross-channel structure that helps discrimination.
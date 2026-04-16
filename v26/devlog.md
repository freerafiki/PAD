# WIKIART 

### Experiment with `train_ranking.py` (14/04/2026)

###### Dataset
```
Dataset Statistics:
  Unique piece pairs: 415241
    Neighbour pairs: 27906
    Non Neighbour pairs: 387335
    Neigh / Non-Neigh split: 6.72% / 93.28%
    Non-Neighbour percentage: 93.28%
  Total positive samples: 27906
  Total negative samples: 139459
  Total hard negative samples: 829573
    Avg negatives per pair: 2.33
  Warning: 387345 pairs have fewer than 5 negatives
    (will use sampling with replacement for these)

=== Puzzle-Based Split ===
Total puzzles: 1648
Train puzzles: 1318
Val puzzles: 330
Created train split: 335220 pairs from 1318 puzzles
  Non-neighbour pairs: 387335
  Neighbour pairs: 27906
Created val split: 80021 pairs from 330 puzzles
  Non-neighbour pairs: 387335
  Neighbour pairs: 27906

=== Dataset Ready ===
Train: 335220 pairs
Val: 80021 pairs
✓ No puzzle overlap between train and val
```
##### Model
```
============================================================
TRAINING MODEL 5: RGB + Geometry + DINO (with more frozen layers)
============================================================
Some weights of ViTModel were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
ViT: 86,389,248 trainable / 86,389,248 total (100.0%)

=== Data Needs Estimate ===
Total parameters: 174,434,116
  ├── Trainable: 87,853,636
  │   ├── Fine-tuned (pre-trained backbones): 86,465,280
  │   └── New layers (from scratch): 1,388,356
  └── Frozen: 86,580,480

Data requirements:
  Minimum (conservative):
    ├── Fine-tuned params: 86,465,280 × 1 = 86,465,280 samples
    └── New layers: 1,388,356 × 10 = 13,883,560 samples
    └── Total minimum: 100,348,840 samples
  Recommended:
    ├── Fine-tuned params: 86,465,280 × 5 = 432,326,400 samples
    └── New layers: 1,388,356 × 20 = 27,767,120 samples
    └── Total recommended: 460,093,520 samples

Current training data: 335,220 samples
❌ INSUFFICIENT data (need at least 100,013,620 more samples)
   Consider: more data, stronger regularization, or freezing more layers
Training multimodal_boundary_wikiart for up to 25 epochs
Early stopping patience: 5
Loss: BCE*0.15 + AdaptiveTopNRankingLoss*0.55 + PerceptualBoundaryLoss*0.3
  AdaptiveTopNRankingLoss: top_n=3, margin=0.3, temperature=1.0
Train samples: 335220 pairs
Val samples: 80021 pairs
```


### Experiment with `train_v3.py` (27/03/2026)

#### Dataset
```
Dataset Statistics:
  Unique piece pairs: 3489
  Total positive samples: 3489
  Total negative samples: 17438
  Total hard negative samples: 6962
  Avg negatives per pair: 6.99

=== Puzzle-Based Split ===
Total puzzles: 197
Train puzzles: 157
Val puzzles: 40
Created train split: 2896 pairs from 157 puzzles
Created val split: 593 pairs from 40 puzzles
```

#### Models
| Model | Val Acc. | Pos/Neg | Val Loss | Train Loss | MAX Epoch | best EPOCH | batch size |
|:------|:--------:|:-------:|:--------:|:----------:|:---------:|:----------:|:----------:|
| RGB*   |   0.442  | 0.527/0.505 | 11.21  | 13.79      |  10    | 1 | 16 |
| RGB   |   0.327  | 0.0/0.0 | 16.29  | 20.65      |  10    | 1 | 16 |
| RGB+Geom   |   0.639  | 0.658/0.168 | 4.54  | 0.06     |  10   | 7 | 16 |
| RGB+Geom+DINO** | 0.751 |  0.727/0.308  |   1.53  |  0.30  | 10 | 7 | 8 

*The baseline RGB model went worse after the first epoch, either something is off or the task without geometry is too hard, as there is no "space" between the pieces. It could also be that geometry is *needed* to understand the alignment scoring task at all

**Because we actually run the training few hours later, the MultiModal model was trained on ~100 puzzles more which have been created while the other two models were training. However, there not too much of a difference, so we keep it here. the reduced `batch_size` is due to memory constraints (24GB VRAM)








# RePAIR 

## DATASET ISSUES
Wrong rendered pieces (position is off) for the following groups:
```
puzzle_0000031_RP_group_30
puzzle_0000032_RP_group_31
puzzle_0000059_RP_group_58
puzzle_0000062_RP_group_61
```

# FIRST RUN
``` bash
> python train_v2.py
Using device: cuda
Loaded 656 positive samples
Loaded 984 negative samples
Loaded 338 hard negative samples
Dataset split: 524 train, 132 val

============================================================
TRAINING VERSION 1: Baseline (RGB only)
============================================================
Some weights of ViTModel were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Training baseline for 3 epochs
Train samples: 524 groups
Val samples: 132 groups
Device: cuda
------------------------------------------------------------
Epoch 1: 100%|█████████████████████████████████████████████████████████████| 33/33 [00:43<00:00,  1.30s/it, loss=0.1619]
Epoch   1/3 | Train Loss: 0.2921 | Val Loss: 0.1465 | Val Acc: 0.530 | Pos/Neg: 0.687/0.411
  → Saved best model (acc: 0.530)
Epoch 2: 100%|█████████████████████████████████████████████████████████████| 33/33 [00:43<00:00,  1.32s/it, loss=0.0595]
Epoch   2/3 | Train Loss: 0.0962 | Val Loss: 0.1162 | Val Acc: 0.667 | Pos/Neg: 0.678/0.250
  → Saved best model (acc: 0.667)
Epoch 3: 100%|█████████████████████████████████████████████████████████████| 33/33 [00:43<00:00,  1.33s/it, loss=0.0031]
Epoch   3/3 | Train Loss: 0.0310 | Val Loss: 0.0938 | Val Acc: 0.712 | Pos/Neg: 0.702/0.183
  → Saved best model (acc: 0.712)
------------------------------------------------------------
Training complete! Best validation accuracy: 0.712
Saved training history plot to checkpoints/baseline_history.png

============================================================
TRAINING VERSION 2: RGB + Geometry
============================================================
Some weights of ViTModel were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Training geometric for 3 epochs
Train samples: 524 groups
Val samples: 132 groups
Device: cuda
------------------------------------------------------------
Epoch 1: 100%|█████████████████████████████████████████████████████████████| 33/33 [00:43<00:00,  1.31s/it, loss=0.2048]
Epoch   1/3 | Train Loss: 0.2696 | Val Loss: 0.1241 | Val Acc: 0.568 | Pos/Neg: 0.628/0.324
  → Saved best model (acc: 0.568)
Epoch 2: 100%|█████████████████████████████████████████████████████████████| 33/33 [00:43<00:00,  1.33s/it, loss=0.0313]
Epoch   2/3 | Train Loss: 0.0607 | Val Loss: 0.0715 | Val Acc: 0.765 | Pos/Neg: 0.598/0.175
  → Saved best model (acc: 0.765)
Epoch 3: 100%|█████████████████████████████████████████████████████████████| 33/33 [00:43<00:00,  1.33s/it, loss=0.0077]
Epoch   3/3 | Train Loss: 0.0120 | Val Loss: 0.0331 | Val Acc: 0.886 | Pos/Neg: 0.751/0.151
  → Saved best model (acc: 0.886)
------------------------------------------------------------
Training complete! Best validation accuracy: 0.886
Saved training history plot to checkpoints/geometric_history.png

============================================================
FINAL COMPARISON
============================================================
Baseline (RGB only):      0.712
+ Geometry:               0.886
```

# SECOND RUN
``` bash
> python train_v2.py
Using device: cuda
Loaded 772 positive samples
Loaded 1158 negative samples
Loaded 420 hard negative samples
Dataset split: 617 train, 155 val

============================================================
FINAL COMPARISON
============================================================
Baseline (RGB only):      0.987
+ Geometry:               0.955
```

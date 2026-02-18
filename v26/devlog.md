# DATASET ISSUES
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

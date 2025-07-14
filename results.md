# Results

Keeping track of the results of the quick attempts (using AI-enabled auto completion for boilerplate code, so most likely full of terrible stuff, be aware) for **binary classification** (correct alignment / incorrect alignment) of pairwise alignments. After I get something *almost working*, will try to fix the code and make it usable and see how to get it into the solver.

## Dataset (v1)
9977 images (80/20 train/val split)
- Training:
    - 3991 correct
    - 3990 wrong
- Validation:
    - 998 correct
    - 998 wrong

## Version 2 (14/07):
- Using Hugging Face library (wandb for logging)
- 15 epochs, `training_batch_size=64`
- Use `AdamW` as optimizer (`pytorch` implementation)
- Data Augmentation using [Albumentationsx](https://github.com/albumentations-team/AlbumentationsX)
- Attention Map visualization

**Scripts:**
- `train_with_transformers.py` (trains using the `transformers` library from hugging face)
- `show_attention_map.py` (visualizes the attention map)
- `evaluate_with_transformers.py` (for the metrics)

#### Performance
The wandb logger says:
| Model | Accuracy | Loss | Runtime | Samples per second |
|:----|:--------:|:----:|:-------:|:------------------:|
| ViTForImageClassification |  0.93136 | 0.22247| 30.105 | 66.301 |

*Note: it is trained starting from "google/vit-base-patch16-224-in21k"*

#### Issues:
- I am doing double preprocessing (albumentations + AutoImageProcessor)
- evaluation code has some problem when loading the data which I do not understand (visualization works)

#### Attention Maps
| Correct Alignments | Wrong Alignments |
|:--------:|:--------:|
|![correct](repo_imgs/att_maps/correct.jpg)|![wrong](repo_imgs/att_maps/wrong.jpg)|
|![correct](repo_imgs/att_maps/correct_2.jpg)|![wrong](repo_imgs/att_maps/wrong_2.jpg)|


## Version 1 (11/07):
- basic initialization (no tricks, not studied, no optimization, no masks, no spatial attention)
- 25 epochs each (with early stopping, ResNet did not stop, HF stop after 19, ST after 12)
- Transforms to perform on-the-fly data augmentation (random crop, flip, rotation)

**Scripts:**
- `train.py` (even the HF transformer model is trained using standard pytorch training loop)
- `evaluate.py` (all can be evaluated using the same code)
- `visualize.py` (show the results)

Clear winner: **Transformer using HuggingFace Pretrained ViT**

*Note: It is a transformer-based model using a pretrained feature extractor (from Google) with a classifier on top (sigmoid output layer).*

Examples: 
| Random 8 Images (Validation Set) | Random 8 Images (Validation Set) |
|:--------:|:--------:|
|![thf results](repo_imgs/predictions_1.png)|![thf results](repo_imgs/predictions_2.png)|

### Performance with `evaluate.py`

| Model          |      Accuracy |  AUC   |  
|:--------------:|:-------------:|:------:|
| ResNet50              | 0.6889 | 0.7570 |
| SmallTransformer      | 0.6042 | 0.6413 |
| **Transformer (HF)**  | **0.9158** | **0.9684** |

**Confusion Matrix:**

| ResNet  |  T |  F | SmallTransformer | T | F | HuggingFace | T | F |  
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P  | 743| 255| P  | 581| 417| P  | 911|  87| 
| N  | 366| 632| N  | 373| 625| N  |  81| 917|

**Note:** I was starting to evaluate on more metrics (to get false positive and so on), but I might need to fix precision/recall/f1 code (auc is from scikit-learn), so for now accuracy and AUC seems to me the only reliable ones. 

| Model          |      Accuracy | Precision | Recall |  F1    |  AUC   |  
|:--------------:|:-------------:|:---------:|:------:|:------:|:------:|
| ResNet50              | 0.6889 | 0.6889    | 0.3111 | 0.4287 | 0.7570 |
| SmallTransformer      | 0.6042 | 0.6042    | 0.3958 | 0.4783 | 0.6413 |
| **Transformer (HF)**  | **0.9158** | **0.9158** | **0.0842** | **0.1542** | **0.9684** |


### Visualization

#### ResNet50
| ResNet50 (Training Set) | ResNet50 (Validation Set) |
|:--------:|:--------:|
|![resnet results](repo_imgs/RN_predictions_train_20epochs.png)|![resnet results](repo_imgs/RN_predictions_val_20epochs.png)|

#### SmallTransformer (PT)
| SmallTransformer (PT) (Training Set) | SmallTransformer (PT) (Validation Set) |
|:--------:|:--------:|
|![thf results](repo_imgs/ST_predictions_train_10epochs.png)|![thf results](repo_imgs/ST_predictions_val_10epochs.png)|


#### Transformer (HF)
| Transformer (HF) (Training Set) | Transformer (HF) (Validation Set) |
|:--------:|:--------:|
|![thf results](repo_imgs/HF_predictions_train_15epochs.png)|![thf results](repo_imgs/HF_predictions_val_15epochs.png)|

**Note:** the validation set was not shuffled, so all the *correct* images are at the front (that's why you see only label 1 in the visualization, you can use `visualize.py` to get random results)

#### Performance during Training

|   Model   | Train Acc. | Val Acc. | Train Loss | Val Loss |
|:---------:|:----------:|:--------:|:----------:|:--------:|
| ResNet50 (Pytorch)            | 67% | 69% | 0.61 | 0.59 |
| SmallTransformer (Pytorch)    | 61% | 59% | 0.66 | 0.67 |
| **Transformer (HuggingFace)** | **92%** | **90%** | **0.18** | **0.24** |


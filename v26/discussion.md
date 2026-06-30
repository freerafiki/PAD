# Key points

## 1. **Representation of the input**
while the image of the two aligned pieces is an obvious choice, we need the optimal way for the model to understand the situation and use the geometric shape of the pieces as an additional input. One alternative idea would be to have a siamese network and feed each piece as separate input (still they would be *transformed* as if they were being aligned). This could also be extended to use cross-attention between the two RGB images. 

Another idea is to use two separate networks with two purposes, one ViT for RGB images, and one CNN for geometric features, then merge them and check the cross-attention between them.

### How it works now
![image](/v26/md_imgs/input.jpg)
We have RGB image, and the possibiilty to include geometric features (3 channel as well, the first two are truncated SDF (proximity field if we want), the third one is the combination and have higher value close to the matching borders of the two pieces). 

### Code
In the code there is a [`RGBScorer`](/v26/single_image/models.py#L17) (ViT plus a small head for the classification) and [`GeometricScorer`](/v26/single_image/models.py#L69) which uses a projection to shrink the 6 layers (rgb + the geometric layers) into 3  - maybe even the projection is a bad idea.


## 2. **Dataloader**
Including context (in terms of multiple alignment of the same pair of pieces) is interesting, but it seems easier and more effective to build the network with one input (one image) and one output (the score) - but the batch should still contains (mostly or exclusively?) images of the same pair of pieces.

### How it works now
![batch](https://codeberg.org/rafiki/PAD/src/branch/main/v26/md_imgs/batch.jpg)
I started out with the single image, moved to the ranking (set of images, you can see here above) and now moved back to the single image (it convinces me more). The *images of the same pair* are found using regex on the filenames (which usually is something like `..puzzle__something___piece_A_vs_piece_B_x_y_theta_...`) and the idea is shared from both dataloaders. 

### Code
In the code we had a [`PrecomposedAlignmentDataset`](/v26/ranking/dataset_ranking.py#L286) (which is used for ranking) which was used to be able to *rank* a set of images, now we are switching to a [`SingleImageDataset`](/v26/single_image/dataset_single.py#L175) which allows us to *score* each image. One trick pointed out would be to build batches so that they have the same piece pairs.

## 3. **Loss** 
BCE is for sure effective and widely used, do we need other contributions to guide the learning (correlation with the attention map, gradient continuity on the edges, ..)

### How it works now
We have multiple losses which can be combined. I saw the difference when changing the weights between BCE (which enforces the `0`/`1` labeling) and RankingLoss (which penalizes ranking the *correct* alignments far from the top 3 alignments of the same pieces). A third loss on the Boundaries (which penalizes continuity on the edges) was built in a hurry and did not really help much. We are currently switching to the simple BCE for easier/faster experimenting, we might use a second one later.

### Code
The custom losses were implemented in the [`loss_ranking` file](/v26/ranking/loss_ranking.py). Latest experiments used a combination of BCE (`nn.BCEWithLogitsLoss` from Pytorch with weights) + [`AdaptiveTopNRankingLoss`](/v26/ranking/loss_ranking.py#L567) + [`BoundaryPairwiseCorrelationLoss`](/v26/ranking/loss_ranking.py#L323)


## 4. **Backbone** 
I tried ViT for strength (and the idea of visualizing attention), but smaller network (ResNet, Inception, MobileNet) could be better for smaller task such as these (and CNN-based network may be faster to learn spatial reasoning tasks). Do we need the network to be rotation-invariant? (e.g. if we rotate both piece and still align them correctly, we want the network to score 1 as well! - if we have one single input, it means that rotation does not change the score - if we have two inputs, rotation does change the score)

### How it works now
Now I have multiple models, mainly three:
1. Baseline / RGB scorer: just takes the RGB image and scores, it was supposed to be the baseline to *beat*. The input is the RGB image (3, 224, 224)
2. Geometric scorer: it includes geometric information about the shape of the pieces, adding the three channels described above in the `representation` section. The input is then RGB + geom (6, 224, 224)
3. Multimodal scorer (not really fond of the name, claude is the culprit, but at least it is different): it includes the features extracted from DINO (initially v2, last experiment with v3) on the RGB image. The input is then RGB + geom + DINO feats (9, 224, 224). I expected this to be the *best* model. Maybe it is too much.

### Code
The backbone is always the standard ViT (`self.vit = ViTModel.from_pretrained`) with the default pretrained model from Google --> `google/vit-base-patch16-224`. An example can be found [here](/v26/single_image/models.py#L26)


## 5. **Evaluation** 
The leap between just scoring a batch of images and solving a puzzle can be huge (false positive, non-adjacent pairs, inbalanced data, ..). We need a quick way to check and evaluate trained model. One way is to run the solver (we have the relaxation-labeling available), another one is to compute the error between the *ground truth* compatibility matrix (obtained from the solved puzzle) and the compatibility matrix obtained scoring each single pairwise alignment with the model.

### How it works now
``` bash
===================================================================================
                Validation Metrics                
===================================================================================
  Accuracy:           0.460         --> total accuracy
  Positive Acc:       0.294         --> accuracy on correct samples
  Negative Acc:       0.410         --> accuracy on wrong samples
  Avg Pos Score:      0.004         --> average score on the correct ones
  Avg Neg Score:      0.002         --> average score on the wrong ones
  Ranking Accuracy:   0.412 (40/97) --> score a point with correct ones ranked top
===================================================================================
```
It evaluates with the *classic* metrics (train accuracy, validation accuracy, false positive). The numbers above are random, but they show the lates metrics used from the evaluation routine. 

### Code
In the [`evaluate_single_image`](/v26/single_image/train_option1.py) function it evauates per-image metrics (I let big pickle add a live `rich`-based visualizaton on the terminal) such as accuracy for positive image, accuracy for negative images, accuracy in general, false positive and ranking (which we might not use anymore)
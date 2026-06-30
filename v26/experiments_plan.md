# Plan

## Dataset
Now we have sets of images of the same pair with different alignments (strongly **imbalanced** dataset: ~93% are wrong alignment, ~7% are correct ones). 
We discussed three main possibilities to handle image/sample/batch creation:
1. Each image is treated as a separate entity independent from th other (baseline, old approach)
2. Each set of images (same pair of pieces) is a sample (which is used for ranking loss, while BCE still goes image by image) - here a batch has more sample (so more images) - this different approach was designed to make the relative relation between alignments of the same pair explicit (ranking)
3. Each set of images (same pair of pieces) is a batch - the BCE loss goes image by image, the "relative-ness" is implicit but not explicit (TO BE IMPLEMENTED)

## Option 1: RGB + ViT + BCE
We train the ViT using RGB only (3 channel) images of A and B aligned, and as loss we use BCE. 
The relative script is [`train_option1.py`](/v26/single_image_vit/train_option1.py)

## Option 2: RGB + Geom + ViT + BCE
We train the ViT using RGB + Geom (6 channels: R, G, B, binary mask A, binary mask B, surface contact between A and B) and as loss we use BCE

## Option 3: RGB + Non-ViT Network (ResNet, Inception, MobileNetv3, ..) + BCE
We train the network using RGB only but without the ViT, with a convolution-based network.
Check the [`single_image_cnn` folder](/v26/single_image_cnn/README.md). Experimenting also with using the contact map as guidance for the network.

## Evaluation:
Evaluate the compatibility at matrix level. Not only image-wise (against target label), but also how far we are from the `ground truth` compatibility matrix (which we create from the ground truth puzzle before creating the pieces). 

## Bonus track: Split Network
We pass the RGB through a ViT and the Geom channels through a CNN and we merge them later (+ MLP) to then score (0,1). We use cross-attention to *match* the attention of the geometric and color information of the two channels.

## Alternative use of RGB
Instead of using the image of A and B aligned as RGB, we can *separate* the two images as RGB_A and RGB_B and pass them to a siamese(-like) network (similar to the bonus track, we can always add some geometric / mask feature-layers) and then merge them and use cross-attention to see if this helps.

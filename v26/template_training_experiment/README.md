# Template Training Experiment

Template folder to build upon for experiments on training a network to score the compatibility given an image (and the geometric features).

## Code Structure

### Parameters
All params goes in a configuration file, for the template we provide as example the `config.py` file

### Model
The model itself is usually saved in a `models.py` or similar, you can take inspiration from the others: `single_image_cnn/resnet_models.py` or `single_image_vit/models.py`.

### Training
The `train_experiment.py` file is an example (taken from the CNN-based network, with some code to show the ViT model loading too) for the training, can and should be customized for the specific experiments

### Visualization
The visualization scripts are usually aligned with the model (they extract attention maps for the ViT, gate maps for the ResNet-based GuidanceGated approach, ..) so there is no template, but you can copy / take some parts from them (in the `single_image_vit/cnn` folders)

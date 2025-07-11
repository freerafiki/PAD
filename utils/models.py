from utils.vit_hf import HuggingFaceTransformer
from utils.vit_pt import SmallTransformer
from utils.resnet import ResNetBinaryClassifier
import os 
import torch 

def load_model(model_type: str, trained_model_path:str = None):
    
    if model_type == 'HF' or model_type == 'transformer_hf' or model_type == 'vit_hf':
        model_name = 'HuggingFaceTransformer'
        # Create and move model to device
        model = HuggingFaceTransformer(
            model_name="google/vit-base-patch16-224-in21k",
            num_classes=2,
            dropout=0.1
        )
    elif model_type == 'resnet':
        model_name = 'ResNet50'
         # Create and move model to device
        model = ResNetBinaryClassifier(
            input_size=(3, 224, 224),
            freeze_base=True,
            dropout=0.2
        )
    elif model_type == 'transformer_pt' or model_type == 'vit_pt':
        model_name = 'SmallTransformer'
        # Create and move model to device
        model = SmallTransformer(
            patch_size=16,
            embed_dim=128,
            depth=4,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.1
        )
    else:
        raise NotImplementedError("Unknown Model!")

    print("Chosen", model_name)
    # CONTINUE TRAINING
    if trained_model_path is not None:
        if trained_model_path == 'best_model.pth':
            trained_model_path = os.path.join('checkpoints', model_name, trained_model_path)
        model.load_state_dict(torch.load(trained_model_path))
        print(f"loading weights from {trained_model_path}..")
    else:
        print("Did not load any weights!")
        
    return model
from transformers import ViTForImageClassification, AutoImageProcessor
import argparse 
import os 
from PIL import Image 
import time 
import torch 
import matplotlib.pyplot as plt 

def main(args):

    # LOAD MODEL
    model = ViTForImageClassification.from_pretrained(args.path)
    processor = AutoImageProcessor.from_pretrained(
        os.path.join(args.path, "config.json"),
        use_fast=True,
        trust_remote_code=True  # Required for local models
    )
    
    image = Image.open(args.image)
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    inputs = processor(images=image, return_tensors="pt")
    # Forward pass with attention output
    with torch.no_grad():
        time1 = time.time()
        outputs = model(**inputs, output_attentions=args.show_attn)
        time2 = time.time()
        print(f"Took {time2-time1} milliseconds")
        if args.show_attn == True:
            attentions = outputs.attentions[-1]  # Last layer attention
    pred_score = outputs['logits']
    pred_class = torch.argmax(pred_score).item()
    pred_class_labels = ['wrong', 'correct']

    # num_plots = 1 if args.show_attn == False else 2
    # plt.figure()
    # plt.suptitle("Inference")
    # plt.subplot(1, num_plots, 1)
    plt.imshow(image)
    plt.title(f"The Model predicts that this is a {pred_class_labels[pred_class]} alignment\nwrong: {pred_score[0][0].item()}\ncorrect: {pred_score[0][1].item()}")
    print(f"The Model predicts that this is a {pred_class_labels[pred_class]} alignment\nwrong: {pred_score[0][0].item()}\ncorrect: {pred_score[0][1].item()}")
    plt.show()
    # plt.subplot(1, num_plots, 2)
    # # Process attention map
    # attention_map = attentions.mean(dim=1)[:, 0, 1:]  # Average across heads, exclude CLS token
    # attention_map = attention_map.reshape(-1, num_of_patches, num_of_patches)
    # # Upsample attention map to match original image size
    # attention_map = torch.nn.functional.interpolate(
    #     attention_map.unsqueeze(0),
    #     scale_factor=(patch_size, patch_size),
    #     mode='bicubic',
    #     align_corners=False
    # )[0]
    # # Normalize attention map
    # if attention_map.min() < 0:
    #     attention_map = (attention_map - attention_map.min())
    # attention_map = attention_map / attention_map.max()
    # # Convert to PIL image
    # attention_map = Image.fromarray((np.squeeze(attention_map.numpy()) * 255).astype('uint8'))
    # reshaped_attention_map = inputs['pixel_values'].squeeze(0).permute(1, 2, 0)
    # plt.title(f"Attention Map Overlay (classified as {pred_class})")
    # plt.imshow(image)
    # plt.imshow(np.array(reshaped_attention_map), cmap='jet', alpha=0.35)
    # plt.axis('off')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show/Save the attention maps from a trained model')
    parser.add_argument('--path', type=str, default="./results_vit_v3_fast", help='output folder of the trained model')  
    parser.add_argument('--image', type=str, default='', help='path for saving the images (folder) - if none, it will show them')  
    parser.add_argument('--show_attn', action="store_true", default=False, help='show relative attention map')  
    args = parser.parse_args()
    main(args)

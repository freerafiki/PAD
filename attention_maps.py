import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoImageProcessor, ViTForImageClassification, AutoModelForImageClassification
import os, random
import argparse 

def visualize_attention_map(model, processor, image_path):
    # Load and preprocess image
    image = Image.open(image_path)
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')


    inputs = processor(images=image, return_tensors="pt")
    
    # Get patch size
    patch_size = processor.patch_size
    num_of_patches = int(processor.image_size // processor.patch_size)
    
    # breakpoint()

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1]  # Last layer attention
    pred_score = outputs['logits']
    pred_class = torch.argmax(pred_score).item()

    # Process attention map
    attention_map = attentions.mean(dim=1)[:, 0, 1:]  # Average across heads, exclude CLS token
    attention_map = attention_map.reshape(-1, num_of_patches, num_of_patches)
    
    # Upsample attention map to match original image size
    attention_map = torch.nn.functional.interpolate(
        attention_map.unsqueeze(0),
        scale_factor=(patch_size, patch_size),
        mode='bicubic',
        align_corners=False
    )[0]
    
    # breakpoint()

    # Normalize attention map
    if attention_map.min() < 0:
        attention_map = (attention_map - attention_map.min())
    attention_map = attention_map / attention_map.max()
    
    # Convert to PIL image
    attention_map = Image.fromarray((np.squeeze(attention_map.numpy()) * 255).astype('uint8'))
    
    reshaped = inputs['pixel_values'].squeeze(0).permute(1, 2, 0)
    
    return reshaped, attention_map, pred_class, pred_score

 
def main(args):
    pred_class_labels = ['wrong', 'correct']

    # Usage
    results_to_show_from = args.path
    model = ViTForImageClassification.from_pretrained(results_to_show_from)
    processor = AutoImageProcessor.from_pretrained(
        os.path.join(results_to_show_from, "config.json"),
        do_center_crop=True, 
        crop_size={"height": args.size, "width": args.size},
        use_fast = True,
        trust_remote_code=True  # Required for local models
    )
    folder_name = '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/validation/'

    correct_folder = os.path.join(folder_name, 'correct')
    num_imgs = 4
    plt.figure(figsize=(32, 24))
    plt.suptitle('Correct Examples')
    imgs_names = random.sample(os.listdir(correct_folder), num_imgs)
    for j, img_name in enumerate(imgs_names):
        preprocessed_image, scaled_attention_map, pred_class, pred_score = visualize_attention_map(model, processor, \
                                                            os.path.join(correct_folder, img_name))
        plt.subplot(3, num_imgs, j+1)
        plt.title(f"Preprocessed Image (classified as {pred_class_labels[pred_class]})\nwrong: {pred_score[0][0].item()}\ncorrect: {pred_score[0][1].item()}")
        plt.imshow(preprocessed_image)
        plt.axis('off')

        plt.subplot(3, num_imgs, j+5)
        plt.title("Attention Map")
        plt.imshow(np.array(scaled_attention_map), cmap='jet', alpha=1)
        plt.axis('off')

        plt.subplot(3, num_imgs, j+9)
        plt.title(f"Attention Map Overlay (classified as {pred_class})")
        plt.imshow(preprocessed_image)
        plt.imshow(np.array(scaled_attention_map), cmap='jet', alpha=0.35)
        plt.axis('off')

    plt.axis('off')
    # plt.tight_layout()
    if args.save == "":
        # plt.show()
        print("will show both images")
    else:
        plt.savefig(os.path.join(args.save, 'correct_attn_maps.jpg'))
    
    wrong_folder = os.path.join(folder_name, 'wrong')
    plt.figure(figsize=(32, 24))
    plt.suptitle('Wrong Examples')
    imgs_names = random.sample(os.listdir(wrong_folder), num_imgs)
    for j, img_name in enumerate(imgs_names):
        preprocessed_image, scaled_attention_map, pred_class, pred_score = visualize_attention_map(model, processor, \
                                                            os.path.join(wrong_folder, img_name))
        plt.subplot(3, num_imgs, j+1)
        plt.title(f"Preprocessed Image (classified as {pred_class_labels[pred_class]})\nwrong: {pred_score[0][0].item()}\ncorrect: {pred_score[0][1].item()}")
        plt.imshow(preprocessed_image)
        plt.axis('off')

        plt.subplot(3, num_imgs, j+5)
        plt.title("Attention Map")
        plt.imshow(np.array(scaled_attention_map), cmap='jet', alpha=1)
        plt.axis('off')

        plt.subplot(3, num_imgs, j+9)
        plt.title(f"Attention Map Overlay (classified as {pred_class})")
        plt.imshow(preprocessed_image)
        plt.imshow(np.array(scaled_attention_map), cmap='jet', alpha=0.35)
        plt.axis('off')

    plt.axis('off')
    # plt.tight_layout()
    if args.save == "":
        plt.show()
    else:
        plt.savefig(os.path.join(args.save, 'wrong_attn_maps.jpg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show/Save the attention maps from a trained model')
    parser.add_argument('--path', type=str, default="./results_vit_v3_fast", help='output folder of the trained model')  
    parser.add_argument('--save', type=str, default='', help='path for saving the images (folder) - if none, it will show them')  
    parser.add_argument('--size', type=int, default=224, help='center_crop_size')
    args = parser.parse_args()
    main(args)

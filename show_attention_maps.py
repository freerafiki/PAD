import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoImageProcessor, ViTForImageClassification
import os, random

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
    
    pred_class = torch.argmax(outputs['logits']).item()

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
    
    # Normalize attention map
    attention_map = attention_map / attention_map.max()
    
    # Convert to PIL image
    attention_map = Image.fromarray((np.squeeze(attention_map.numpy()) * 255).astype('uint8'))
    
    reshaped = inputs['pixel_values'].squeeze(0).permute(1, 2, 0)
    
    return reshaped, attention_map, pred_class

 

# Usage
model = ViTForImageClassification.from_pretrained("./results_aug/checkpoint-1800")
#     "results/checkpoint-2500/config.json",
#     trust_remote_code=True  # Required for local models
# )
# processor = AutoImageProcessor.from_pretrained(model_name)
preprocessing_steps = {
    'resize': {'method': 'center_crop', 'size': 224},
    'do_normalize': False,
}
processor = AutoImageProcessor.from_pretrained(
    "./results_aug/checkpoint-1800/config.json",
    preprocessing_steps=preprocessing_steps,
    trust_remote_code=True  # Required for local models
)
folder_name = '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/train/correct/'

num_imgs = 4
plt.figure(figsize=(32, int(8*num_imgs)))
    
imgs_names = random.sample(os.listdir(folder_name), num_imgs)
for j, img_name in enumerate(imgs_names):
    preprocessed_image, scaled_attention_map, pred_class = visualize_attention_map(model, processor, \
                                                        os.path.join(folder_name, img_name))
    plt.subplot(3, num_imgs, j+1)
    plt.title(f"Preprocessed Image (classified as {pred_class})")
    plt.imshow(preprocessed_image)

    plt.subplot(3, num_imgs, j+5)
    plt.title("Attention Map")
    plt.imshow(np.array(scaled_attention_map), cmap='jet', alpha=1)

    plt.subplot(3, num_imgs, j+9)
    plt.title(f"Attention Map Overlay (classified as {pred_class})")
    plt.imshow(preprocessed_image)
    plt.imshow(np.array(scaled_attention_map), cmap='jet', alpha=0.35)

plt.axis('off')
# plt.tight_layout()
plt.show()

# img_name = 'puzzle_0000002_RP_group_2_vis_5_RPf_00016_6_RPf_00017_1_14_0_gt.png'
# # 'puzzle_0000002_RP_group_2_vis_2_RPf_00013_0_RPf_00011_7_25_0_gt.png'
# img_path = os.path.join(folder_name, img_name)
# visualize_attention(model, processor, img_path)



   # Create visualization
















# import torch
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from transformers import AutoImageProcessor, ViTForImageClassification

# def visualize_attention(image_path, model_name="google/vit-base-patch16-224"):
#     # Load model and processor
#     model = ViTForImageClassification.from_pretrained(model_name)
#     # processor = AutoImageProcessor.from_pretrained(model_name)
#     processor = AutoImageProcessor.from_pretrained(
#         "results/checkpoint-2500/config.json",
#         trust_remote_code=True  # Required for local models
#     )
    
#     # Load and preprocess image
#     image = Image.open(image_path)
#     # Convert RGBA to RGB if necessary
#     if image.mode == 'RGBA':
#         image = image.convert('RGB')

#     inputs = processor(images=image, return_tensors="pt")
    
#     # Get attention weights
#     with torch.no_grad():
#         outputs = model(**inputs, output_attentions=True)
#         attentions = outputs.attentions[-1]  # Last layer attention
    
#     # Average across heads
#     avg_attention = torch.mean(attentions[0], dim=1)
    
#     # Resize attention map to match original image size
#     patch_size = processor.model_input_names[0] + "_size"
#     attention_map = torch.nn.functional.interpolate(
#         avg_attention.unsqueeze(0),
#         scale_factor=(patch_size, patch_size),
#         mode='bicubic',
#         align_corners=False
#     )[0]
    
#     # Normalize attention map
#     attention_map = attention_map - attention_map.min()
#     attention_map = attention_map / attention_map.max()
    
#     # Convert to numpy array
#     attention_map = attention_map.numpy()
    
#     # Create visualization
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Plot original image
#     ax1 = plt.subplot(121)
#     ax1.imshow(np.array(image))
#     ax1.axis('off')
#     ax1.set_title('Original Image')
    
#     # Plot attention heatmap
#     ax2 = plt.subplot(122)
#     img = ax2.imshow(attention_map.squeeze(), cmap='hot', alpha=0.5)
#     ax2.axis('off')
#     ax2.set_title('Attention Map')
#     plt.colorbar(img)
    
#     plt.tight_layout()
#     return fig

# fig = visualize_attention('/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/train/correct/puzzle_0000002_RP_group_2_vis_2_RPf_00013_0_RPf_00011_7_25_0_gt.png')
# plt.show()
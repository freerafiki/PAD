from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import albumentations as A
import numpy as np
import cv2

# Define a custom dataset class
class ImageDataset(Dataset):
    def __init__(self, images, labels, processor, transform=None):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=np.array(image))
            image = augmented['image'].to(torch.float)

        inputs = self.processor(images=image, return_tensors="pt")
        # inputs['pixel_values'].to(torch.float)
        inputs['labels'] = torch.tensor(self.labels[idx])
        return inputs

# Define data collator
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'].squeeze() for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

training_transform_384 = A.Compose([
    A.Resize(height=384, width=384, interpolation=cv2.INTER_AREA, area_for_downscale="image", p=1.0),  # Use INTER_AREA when downscaling images),
    A.RandomCrop(width=384, height=384),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=180, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
    A.RandomBrightnessContrast(p=0.1),
    # A.Blur(blur_limit=(3, 7), p=0.1),
    A.ImageCompression(quality_range=(50, 90), compression_type='jpeg', p=0.25),
    A.ToTensorV2(),
])

training_transform_256 = A.Compose([
    A.Resize(height=256, width=256, interpolation=cv2.INTER_AREA, area_for_downscale="image", p=1.0),  # Use INTER_AREA when downscaling images),
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=180, p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.05),
    A.RandomBrightnessContrast(p=0.05),
    A.ImageCompression(quality_range=(70, 90), compression_type='jpeg', p=0.05),
    A.ToTensorV2(),
])

training_transform_224 = A.Compose([
    A.Resize(height=224, width=224, interpolation=cv2.INTER_AREA, area_for_downscale="image", p=1.0),  # Use INTER_AREA when downscaling images),
    A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=180, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
    A.RandomBrightnessContrast(p=0.1),
    # A.Blur(blur_limit=(3, 7), p=0.1),
    A.ImageCompression(quality_range=(50, 90), compression_type='jpeg', p=0.25),
    A.ToTensorV2(),
])

def training_transforms(examples, size=224):
    if size == 224:
        images = [training_transform_224(image=np.array(image))["image"] for image in examples["image"]]
    elif size == 384:
        images = [training_transform_384(image=np.array(image))["image"] for image in examples["image"]]
    else:
        raise NotImplementedError()
    return {"pixel_values": images}

validation_transform_384 = A.Compose([
    A.Resize(height=384, width=384, interpolation=cv2.INTER_AREA, area_for_downscale="image", p=1.0),  # Use INTER_AREA when downscaling images),
    A.CenterCrop(width=384, height=384),
    A.ToTensorV2(),
])

validation_transform_224 = A.Compose([
    A.Resize(height=224, width=224, interpolation=cv2.INTER_AREA, area_for_downscale="image", p=1.0),  # Use INTER_AREA when downscaling images),
    A.CenterCrop(width=224, height=224),
    A.ToTensorV2(),
])

def validation_transforms(examples):
    if size == 224:
        images = [validation_transform_224(image=np.array(image))["image"] for image in examples["image"]]
    elif size == 384:
        images = [validation_transform_384(image=np.array(image))["image"] for image in examples["image"]]
    else:
        raise NotImplementedError()
    return {"pixel_values": images}

def format_images_labels_list(data_directory:str):
    
    train_folder = os.path.join(data_directory, "train")

    # Create paths to your images
    train_correct_images = [f"{data_directory}/train/correct/{file}" for file in os.listdir(f"{data_directory}/train/correct")]
    train_wrong_images = [f"{data_directory}/train/wrong/{file}" for file in os.listdir(f"{data_directory}/train/wrong")]
    train_images = train_correct_images + train_wrong_images
    train_labels = [1] * len(train_correct_images) + [0] * len(train_wrong_images)

    # Create paths to your images
    val_correct_images = [f"{data_directory}/validation/correct/{file}" for file in os.listdir(f"{data_directory}/validation/correct")]
    val_wrong_images = [f"{data_directory}/validation/wrong/{file}" for file in os.listdir(f"{data_directory}/validation/wrong")]
    val_images = val_correct_images + val_wrong_images
    val_labels = [1] * len(val_correct_images) + [0] * len(val_wrong_images)

    return train_images, train_labels, val_images, val_labels
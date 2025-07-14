from transformers import Trainer, TrainingArguments, ViTForImageClassification, AutoImageProcessor
from utils.dataset_hf import format_images_labels_list, ImageDataset, validation_transform

data_directory = '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/'
train_images, train_labels, val_images, val_labels = format_images_labels_list(data_directory)

preprocessing_steps = {
    'resize': {'method': 'center_crop', 'size': 224},
    'do_normalize': False,
}
processor = AutoImageProcessor.from_pretrained(
    "./results_aug/checkpoint-1800/config.json",
    preprocessing_steps=preprocessing_steps,
    trust_remote_code=True  # Required for local models
)

# Create datasets
# train_dataset = ImageDataset(train_images, train_labels, processor, transform=training_transform)
eval_dataset = ImageDataset(val_images, val_labels, processor)#, transform=validation_transform)

# Create minimal training arguments (you can customize these)
eval_args = TrainingArguments(
    output_dir="./evaluations",
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
)

model = ViTForImageClassification.from_pretrained("./results_aug/checkpoint-1800")
# Initialize trainer with your model and dataset
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=eval_dataset
)

print("Dataset shape:", next(iter(eval_dataset))['pixel_values'].shape)
print("Model expected input shape:", model.config.image_size)

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)
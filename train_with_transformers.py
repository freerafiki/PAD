from transformers import ViTForImageClassification, AutoImageProcessor
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer
from utils.dataset_hf import ImageDataset, format_images_labels_list, collate_fn, \
                            training_transform, validation_transform

# Initialize the model and processor
model_name = "google/vit-base-patch16-384" #"google/vit-base-patch16-224-in21k"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Configure the model for your specific number of classes
num_classes = 2  # Replace with your actual number of classes
model.classifier = nn.Linear(model.config.hidden_size, num_classes)

epochs = 15
# Training arguments
training_args = TrainingArguments(
    output_dir="./results_384",
    num_train_epochs=epochs,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=100,
    # learning_rate=2e-4,
    save_steps=100,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    no_cuda=False if torch.cuda.is_available() else True,
    save_on_each_node=True
)

data_directory = '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/'
train_images, train_labels, val_images, val_labels = format_images_labels_list(data_directory)

# Create datasets
train_dataset = ImageDataset(train_images, train_labels, processor, transform=training_transform)
eval_dataset = ImageDataset(val_images, val_labels, processor, transform=validation_transform)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda pred: {"accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1))},
    data_collator=collate_fn,
    optimizers=(torch.optim.AdamW(model.parameters(), lr=1e-4), None)
)

# Train the model
trainer.train()

# Save the model
# trainer.save_pretrained("./vit_hf_{epochs}epochs")
trainer.save_model()
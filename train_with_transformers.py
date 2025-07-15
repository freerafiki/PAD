from transformers import ViTForImageClassification, AutoImageProcessor
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer
from utils.dataset_hf import ImageDataset, format_images_labels_list, collate_fn, \
    training_transform_384, validation_transform_384
import argparse

def main(args):
    # Initialize the model and processor
    model_name = args.model_name #"google/vit-base-patch16-224-in21k" #"google/vit-base-patch16-384" #
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = ViTForImageClassification.from_pretrained(model_name)

    # Configure the model for your specific number of classes
    num_classes = 2  # Replace with your actual number of classes
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)

    epochs = args.epochs
    run_name = args.run_name if args.run_name != '' else args.output_dir
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir, #"./results_vit_v3_fast_dataset_v2",
        run_name=run_name, #'ViT_fast_v3_15_epochs_dataset_v2',
        num_train_epochs=epochs,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        eval_strategy="steps",
        eval_steps=100,
        # learning_rate=2e-4,
        save_steps=100,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=4,
        no_cuda=False if torch.cuda.is_available() else True,
        save_on_each_node=True
    )

    data_directory = '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/'
    train_images, train_labels, val_images, val_labels = format_images_labels_list(data_directory)

    # Create datasets
    train_dataset = ImageDataset(train_images, train_labels, processor, transform=training_transform_384)
    eval_dataset = ImageDataset(val_images, val_labels, processor, transform=validation_transform_384)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda pred: {"accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1))},
        data_collator=collate_fn,
        optimizers=(torch.optim.AdamW(model.parameters(), lr=1e-4), None),
    )

    # Train the model
    trainer.train() #resume_from_checkpoint="./results_vit_v3_fast/checkpoint-315")

    # Save the model
    # trainer.save_pretrained("./vit_hf_{epochs}epochs")
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a ViTForImageClassification model for binary classification')
    parser.add_argument('--model_name', type=str, default="google/vit-base-patch16-224-in21k", help='ViT model name')  
    parser.add_argument('--output_dir', type=str, default='./results_vit_base_224', help='output directory')  
    parser.add_argument('--run_name', type=str, default='', help='naming the run for logging (otherwise takes output dir)')  
    parser.add_argument('--resume_from', type=str, default='', help='resume from checkpoint')  
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs for training')  
    args = parser.parse_args()
    main(args)
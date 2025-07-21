from transformers import Trainer, TrainingArguments, ViTForImageClassification, AutoImageProcessor
from utils.dataset_hf import format_images_labels_list, ImageDataset, collate_fn
import os 
import torch 
from transformers import Trainer, EvalPrediction
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse 

def compute_metrics_sk(p: EvalPrediction):
    predictions = p.predictions.argmax(-1)
    labels = p.label_ids
    
    # Basic accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Additional metrics
    report = classification_report(labels, predictions, output_dict=True)
    matrix = confusion_matrix(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

def main(args):
    data_directory = '/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset/'
    train_images, train_labels, val_images, val_labels = format_images_labels_list(data_directory)

    processor = AutoImageProcessor.from_pretrained(os.path.join(results_to_show_from, "config.json"), do_center_crop=True, crop_size={"height": args.size, "width": args.size}, use_fast=True)
    model = ViTForImageClassification.from_pretrained(results_to_show_from)

    # Usage
    results_to_show_from = args.path #"./results_vit_v3_fast"
    model = ViTForImageClassification.from_pretrained(results_to_show_from)
    processor = AutoImageProcessor.from_pretrained(
        os.path.join(results_to_show_from, "config.json"),
        do_center_crop=True, 
        crop_size={"height": args.size, "width": args.size},
        use_fast = args.use_fast,
        trust_remote_code=True  # Required for local models
    )

    # Create datasets
    # train_dataset = ImageDataset(train_images, train_labels, processor, transform=training_transform)
    eval_dataset = ImageDataset(val_images, val_labels, processor)#, transform=validation_transform)

    # Create minimal training arguments (you can customize these)
    eval_args = TrainingArguments(
        output_dir="./evaluations",
        per_device_eval_batch_size=64,
        eval_strategy="steps",
    )

    # Initialize trainer with your model and dataset
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_sk,
    )

    # Evaluate the model
    metrics = trainer.evaluate()
    print("#" * 50)
    for metric in metrics.keys():
        print(f"- {metric}: {metrics[metric]:.04f}")
    print("#" * 50)

    if args.md == True:
        metric_names = "| Model | "
        for metric in metrics.keys():
            metric_names += f"{metric} | "
        print(metric_names)
        md_row = "|:----"
        for metric in metrics.keys():
            md_row += "|:---:"
        md_row += "|"
        print(md_row)
        metric_vals = f"| {args.path.split('/')[-1]} | "
        for metric in metrics.keys():
            metric_vals += f"{metrics[metric]:.03f} | "
        print(metric_vals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show/Save the attention maps from a trained model')
    parser.add_argument('--path', type=str, default="./results_vit_v3_fast", help='output folder of the trained model')  
    parser.add_argument('--md', action="store_true", default=False, help='print metrics in markdown table style')  
    parser.add_argument('--use_fast', action="store_true", default=False, help='use_fast processor')  
    args = parser.parse_args()
    main(args)
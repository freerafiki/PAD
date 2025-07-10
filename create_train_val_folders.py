import os
import shutil
import random
from typing import List, Tuple

def list_files(directory: str) -> List[str]:
    """Lists all files in the given directory recursively."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def quick_list_files(directory: str) -> List[str]:
    files = []
    for filename in os.listdir(directory):
        files.append(os.path.join(directory, filename))
    return files

def split_and_copy_files(
    source_dir: str,
    train_dir: str,
    val_dir: str,
    split_ratio: float = 0.8
) -> None:
    """Splits files into training and validation sets and copies them."""
    # Get all files from correct and wrong folders
    correct_files = quick_list_files(os.path.join(source_dir, "correct"))
    wrong_files = quick_list_files(os.path.join(source_dir, "wrong"))
    
    # Combine and shuffle files
    all_files = correct_files + wrong_files
    random.shuffle(all_files)
    
    # Calculate split point
    split_point = int(len(all_files) * split_ratio)
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "wrong"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "wrong"), exist_ok=True)
    
    # Copy files to training directory
    print(f"Copying {split_point} files to training directory...")
    for i, file_path in enumerate(all_files[:split_point]):
        if i % 100 == 0:
            print(f"Progress: {i}/{split_point}")
        
        # Determine destination folder based on source folder
        if "correct" in file_path:
            dest_folder = os.path.join(train_dir, "correct")
        else:
            dest_folder = os.path.join(train_dir, "wrong")
        
        try:
            shutil.copy2(file_path, dest_folder)
        except Exception as e:
            print(f"Error copying {file_path}: {str(e)}")
    
    # Copy remaining files to validation directory
    print(f"Copying {len(all_files) - split_point} files to validation directory...")
    for i, file_path in enumerate(all_files[split_point:], split_point):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(all_files)}")
        
        # Determine destination folder based on source folder
        if "correct" in file_path:
            dest_folder = os.path.join(val_dir, "correct")
        else:
            dest_folder = os.path.join(val_dir, "wrong")
        
        try:
            shutil.copy2(file_path, dest_folder)
        except Exception as e:
            print(f"Error copying {file_path}: {str(e)}")

# Example usage
if __name__ == "__main__":
    source_directory = "/home/ssd/datasets/RePAIR_ReLab_luca/PAD"
    train_directory = "/home/ssd/datasets/RePAIR_ReLab_luca/PAD/dataset/train"
    val_directory = "/home/ssd/datasets/RePAIR_ReLab_luca/PAD/dataset/validation"
    
    print("Starting file splitting and copying...")
    split_and_copy_files(source_directory, train_directory, val_directory)
    print("Done!")
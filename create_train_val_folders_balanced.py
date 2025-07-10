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
    split_ratio: float = 0.8,
    balance_tolerance: float = 0.1  # 10% tolerance in class balance
) -> None:
    """Splits files into training and validation sets while maintaining class balance."""
    # Get all files from correct and wrong folders
    correct_files = quick_list_files(os.path.join(source_dir, "correct"))
    wrong_files = quick_list_files(os.path.join(source_dir, "wrong"))
    
    # Calculate target counts for each set
    total_correct = len(correct_files)
    total_wrong = len(wrong_files)
    train_correct_target = int(total_correct * split_ratio)
    train_wrong_target = int(total_wrong * split_ratio)
    
    # Initialize counters
    train_correct_count = 0
    train_wrong_count = 0
    val_correct_count = 0
    val_wrong_count = 0
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "wrong"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "wrong"), exist_ok=True)
    
    # Shuffle files within each class
    random.shuffle(correct_files)
    random.shuffle(wrong_files)
    
    # Copy files while maintaining balance
    print(f"Copying files while maintaining class balance...")
    for file_path in correct_files:
        if train_correct_count < train_correct_target:
            dest_folder = os.path.join(train_dir, "correct")
            train_correct_count += 1
        else:
            dest_folder = os.path.join(val_dir, "correct")
            val_correct_count += 1
        
        try:
            shutil.copy2(file_path, dest_folder)
        except Exception as e:
            print(f"Error copying {file_path}: {str(e)}")
    
    for file_path in wrong_files:
        if train_wrong_count < train_wrong_target:
            dest_folder = os.path.join(train_dir, "wrong")
            train_wrong_count += 1
        else:
            dest_folder = os.path.join(val_dir, "wrong")
            val_wrong_count += 1
        
        try:
            shutil.copy2(file_path, dest_folder)
        except Exception as e:
            print(f"Error copying {file_path}: {str(e)}")
    
    # Print distribution statistics
    print("\nFinal distribution:")
    print(f"Training set - Correct: {train_correct_count}, Wrong: {train_wrong_count}")
    print(f"Validation set - Correct: {val_correct_count}, Wrong: {val_wrong_count}")
    
    # Check balance
    train_total = train_correct_count + train_wrong_count
    val_total = val_correct_count + val_wrong_count
    
    train_correct_ratio = train_correct_count / train_total if train_total > 0 else 0
    val_correct_ratio = val_correct_count / val_total if val_total > 0 else 0
    
    print(f"\nClass balance ratios:")
    print(f"Training set: {train_correct_ratio:.2%} correct, {1-train_correct_ratio:.2%} wrong")
    print(f"Validation set: {val_correct_ratio:.2%} correct, {1-val_correct_ratio:.2%} wrong")
    
    # Verify balance is within tolerance
    if not (0.5 - balance_tolerance <= train_correct_ratio <= 0.5 + balance_tolerance):
        print("\nWarning: Training set balance is outside tolerance!")
    if not (0.5 - balance_tolerance <= val_correct_ratio <= 0.5 + balance_tolerance):
        print("\nWarning: Validation set balance is outside tolerance!")

# Example usage
if __name__ == "__main__":
    source_directory = "/home/ssd/datasets/RePAIR_ReLab_luca/PAD"
    train_directory = "/home/ssd/datasets/RePAIR_ReLab_luca/PAD/dataset/train"
    val_directory = "/home/ssd/datasets/RePAIR_ReLab_luca/PAD/dataset/validation"
    
    print("Starting file splitting and copying...")
    split_and_copy_files(
        source_directory,
        train_directory,
        val_directory,
        split_ratio=0.8,
        balance_tolerance=0.1
    )
    print("Done!")
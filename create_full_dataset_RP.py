import os 
import shutil 
import random 
import numpy as np 

source_directory = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v2"
target_directory = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v2/as_dataset"
train_target_folder = os.path.join(target_directory, 'train')
val_target_folder = os.path.join(target_directory, 'validation')
test_target_folder = os.path.join(target_directory, 'test')
os.makedirs(train_target_folder, exist_ok=True)
os.makedirs(val_target_folder, exist_ok=True)
os.makedirs(test_target_folder, exist_ok=True)

categories = ['correct', 'wrong']

for cat in categories:
    for folder in [train_target_folder, val_target_folder, test_target_folder]:
        os.makedirs(os.path.join(folder, cat), exist_ok=True)

print("Created folders")
breakpoint()
num_imgs = {}

for cat in categories:
    cat_folder = os.path.join(source_directory, cat)
    images = os.listdir(cat_folder)
    num_imgs[cat] = len(images)

num_imgs['total'] = num_imgs[categories[0]] + num_imgs[categories[1]]

for k,v in num_imgs.items():
    print(f'{k}:{v}')

print("Counted images")
breakpoint()

# test_set_puzzles = np.loadtxt("/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD_v2/test.txt", dtype=str)
# test_set_size = 0

# for cat in categories:
#     cat_folder = os.path.join(source_directory, cat)
#     images = os.listdir(cat_folder)
#     num_imgs[cat] = len(images)
#     # remove "test set" images
#     for img_name in images:
#         puzzle_name = "_".join(img_name.split('_')[:5])
#         if puzzle_name in test_set_puzzles:
#             src=os.path.join(cat_folder, img_name)
#             dst=os.path.join(test_target_folder, cat, img_name)
#             # print(f"would move {src} to {dst}")
#             shutil.move(src, dst)    
#             test_set_size += 1

# print(f"Test set size: {test_set_size}")

print("Created test set")
breakpoint()

for cat in categories:
    cat_folder = os.path.join(source_directory, cat)
    images = os.listdir(cat_folder)
    num_imgs[cat] = len(images)

num_imgs['total'] = num_imgs[categories[0]] + num_imgs[categories[1]]

for k,v in num_imgs.items():
    print(f'{k}:{v}')

print("Re-counted images")
breakpoint()

train_size = 0
val_size = 0

split_ratio = 0.9
for cat in categories:
    print(cat)
    cat_folder = os.path.join(source_directory, cat)
    images = os.listdir(cat_folder)
    random.shuffle(images)
    train_img_num = len(images) * 0.9
    for j, img_name in enumerate(images):
        if j <= train_img_num:
            src=os.path.join(cat_folder, img_name)
            dst=os.path.join(train_target_folder, cat, img_name)
            # print(f"would move {src} to {dst}")
            shutil.move(src, dst)    
            train_size += 1
        else:
            src=os.path.join(cat_folder, img_name)
            dst=os.path.join(val_target_folder, cat, img_name)
            # print(f"would move {src} to {dst}")
            shutil.move(src, dst)    
            val_size += 1

print(f"Train set size: {train_size}")
print(f"Validation set size: {val_size}")    


print("Done")
breakpoint()
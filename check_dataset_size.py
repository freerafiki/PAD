import os 

source_directory = "/run/user/1000/gvfs/sftp:host=gpu1.dsi.unive.it,user=luca.palmieri/home/ssd/datasets/RePAIR_ReLab_luca/PAD/as_dataset"
train_directory = os.path.join(source_directory,"train")
val_directory = os.path.join(source_directory,"validation")
test_directory = os.path.join(source_directory,"test")

print("#" * 30)
print("# TRAINING")
train_images = 0
for category in ['correct', 'wrong']:
    data_dir = os.path.join(train_directory, category)
    num_cat = len(os.listdir(data_dir))
    print(f"# {category}: {num_cat}")
    train_images += num_cat

print("#" * 30)
print("# VALIDATION")
val_images = 0
for category in ['correct', 'wrong']:
    data_dir = os.path.join(val_directory, category)
    num_cat = len(os.listdir(data_dir))
    print(f"# {category}: {num_cat}")
    val_images += num_cat

print("#" * 30)
print("# TEST")
test_images = 0
for category in ['correct', 'wrong']:
    data_dir = os.path.join(test_directory, category)
    num_cat = len(os.listdir(data_dir))
    print(f"# {category}: {num_cat}")
    test_images += num_cat


tot_imgs = train_images + val_images + test_images
train_imgs_perc = train_images / tot_imgs
val_imgs_perc = val_images / tot_imgs
test_imgs_perc = test_images / tot_imgs
print("#" * 30)
print("# SPLIT")
print(f"# total: {tot_imgs}")
print(f"# train: {train_images} ({train_imgs_perc*100:.02f}%)")
print(f"# validation: {val_images} ({val_imgs_perc*100:.02f}%)")
print(f"# test: {test_images} ({test_imgs_perc*100:.02f}%)")
print("#" * 30)
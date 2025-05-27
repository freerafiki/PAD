import os 
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 

dafne_root = '/media/lucap/big_data/datasets/DS_5_Dafne'
images = os.listdir(dafne_root)

output_folder = '/media/lucap/big_data/datasets/dafne_border_pairs_1000'
correct_folder = os.path.join(output_folder, 'correct')
wrong_same_img_folder = os.path.join(output_folder, 'wrong_same_img')
transf_folder = os.path.join(output_folder, 'transformed')
other_img_folder = os.path.join(output_folder, 'other')
os.makedirs(correct_folder, exist_ok=True)
os.makedirs(wrong_same_img_folder, exist_ok=True)
os.makedirs(transf_folder, exist_ok=True)
os.makedirs(other_img_folder, exist_ok=True)

# image size
border_len = 128
hbl = border_len // 2

# samples of the dataset
num_samples = 1000 # whole dataset contains 20 pairs (20 x 4 in fact)
samples_per_img = num_samples // len(images)

# prnt
print("#" * 60)
print("Border Pairs Dataset Creation")
print("-" * 60)
print(f"Using {dafne_root.split('/')[-1]} dataset")
print(f"Num pairs in the dataset: {num_samples}")
print(f"Num samples per image: {samples_per_img}")
print("#" * 60)

max_attempts = 10

if samples_per_img < 1:
    print('too few samples, too many images - it will not be a real dataset')

for j, image in enumerate(images):

    img_full_path = os.path.join(dafne_root, image)
    img = Image.open(img_full_path)
    w, h = img.size

    for sample in range(samples_per_img):

        random_x = np.random.uniform(border_len, w - border_len)
        random_y = np.random.uniform(hbl, h - hbl)

        # rectangle around it
        correct_pair = img.crop((random_x-border_len, random_y-hbl, random_x+border_len, random_y+hbl))
        correct_pair.save(os.path.join(correct_folder, f'img_{j}_sample_{sample}_correct.jpg'))

        # patches
        left_patch = img.crop((random_x-border_len, random_y-hbl, random_x, random_y+hbl))
        right_patch = img.crop((random_x, random_y-hbl, random_x+border_len, random_y+hbl))

        # # noisy
        # gaussian_noise = (np.random.random((border_len, border_len, 3)) - 0.5) * noise_factor
        # noisy_np_patch = np.round(np.array(right_patch) + gaussian_noise).astype(np.uint8)
        # noisy_r_patch = Image.fromarray(noisy_np_patch)
        # noisy_pair = Image.new('RGB', (left_patch.width * 2, left_patch.height))
        # noisy_pair.paste(left_patch, (0, 0))
        # noisy_pair.paste(noisy_r_patch, (left_patch.width, 0))

        # transformed
        random_rot_choices = np.round(np.random.uniform(0, 3))
        if random_rot_choices == 0:
            trf_ang = 90
        elif random_rot_choices == 1:
            trf_ang = 180
        elif random_rot_choices == 2:
            trf_ang = 270
        else:
            trf_ang = 180
        transformed = right_patch.rotate(trf_ang)
        transf_pair = Image.new('RGB', (left_patch.width * 2, left_patch.height))
        transf_pair.paste(left_patch, (0, 0))
        transf_pair.paste(transformed, (left_patch.width, 0))
        transf_pair.save(os.path.join(transf_folder, f'img_{j}_sample_{sample}_transformed.jpg'))

        # other part of image
        
        found_diff_point = False
        tried_enough = False
        counter_tries = 0
        while not found_diff_point or not tried_enough:
            random_x2 = np.random.uniform(border_len, w - border_len)
            random_y2 = np.random.uniform(hbl, h - hbl)
            if np.linalg.norm(np.array([random_x2, random_y2])-np.array([random_x, random_y])) > border_len:
                found_diff_point = True 
            else:
                counter_tries += 1
                if counter_tries > max_attempts:
                    tried_enough = True
        right_same_img_patch = img.crop((random_x2, random_y2-hbl, random_x2+border_len, random_y2+hbl))  
        random_rot_choices = np.round(np.random.uniform(0, 3))
        if random_rot_choices == 0:
            right_same_img_patch = right_same_img_patch.rotate(90)
        elif random_rot_choices == 1:
            right_same_img_patch = right_same_img_patch.rotate(270)  
        else:
            # nothing
            right_same_img_patch = right_same_img_patch
        wrong_same_img_pair = Image.new('RGB', (left_patch.width * 2, left_patch.height))
        wrong_same_img_pair.paste(left_patch, (0, 0))
        wrong_same_img_pair.paste(right_same_img_patch, (left_patch.width, 0))
        wrong_same_img_pair.save(os.path.join(wrong_same_img_folder, f'img_{j}_sample_{sample}_wrong_same_img.jpg'))        

        # other image
        possible_other_images = list(np.arange(1, len(images), 1))
        possible_other_images.pop(j-1)
        chosen_idx = np.random.choice(possible_other_images)
        second_img_full_path = os.path.join(dafne_root, images[chosen_idx])
        second_img = Image.open(second_img_full_path)
        w2, h2 = second_img.size
        random_x3 = np.random.uniform(border_len, w2 - border_len)
        random_y3 = np.random.uniform(hbl, h2 - hbl)
        right_patch_other = second_img.crop((random_x3, random_y3-hbl, random_x3+border_len, random_y3+hbl))
        other_image_pair = Image.new('RGB', (left_patch.width * 2, left_patch.height))
        other_image_pair.paste(left_patch, (0, 0))
        other_image_pair.paste(right_patch_other, (left_patch.width, 0))
        other_image_pair.save(os.path.join(other_img_folder, f'img_{j}_sample_{sample}_wrong_other_img.jpg'))        

        progress = (j * len(images) + sample)//1
        perc = progress / num_samples * 100
        print(f"Done with sample {progress:d} / {num_samples:d} ({perc:.02f} %)", end='\r')
        # plt.subplot(141); plt.title("Correct Pair")
        # plt.imshow(np.array(correct_pair))
        # plt.subplot(143); plt.title("Same Image, Other Position Pair")
        # plt.imshow(np.array(wrong_same_img_pair))
        # plt.subplot(142); plt.title(f"Transformed Pair (rotated {trf_ang})")
        # plt.imshow(np.array(transf_pair))
        # plt.subplot(144); plt.title("Other Image Pair")
        # plt.imshow(np.array(other_image_pair))
        # plt.show()
        # breakpoint()



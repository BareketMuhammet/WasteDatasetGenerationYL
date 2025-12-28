
from pathlib import Path

import cv2
import numpy as np
from dataset.scripts.background_aug_functions import *


oil_effect_probability = 0.7
blisters_effect_probability = 0.7
dust_effect_probability = 0.7

noisy_mask_scale=[5, 100]
oil_strength_range=[0.4, 0.9]
blisters_count_range=[5, 25]
blisters_amount_range=[1, 5]
dust_amount_range=[0.1, 2]
rotation_angle_range = [-30, 30]

image_width = 640
image_height = 640

debug = True

background_dir = Path("background_textures")


valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
background_candidates = [
    path for path in background_dir.glob("*") if path.suffix.lower() in valid_extensions
]
if not background_candidates:
    raise FileNotFoundError(f"No background images found in {background_dir}")

# Select and load random background texture image
selected_background = np.random.choice(background_candidates)
if debug:
    print(f"Using background: {selected_background}")
img = cv2.imread(str(selected_background))
if img is None:
    raise FileNotFoundError(f"Failed to read background image: {selected_background}")

# Crop
orig_height, orig_width = img.shape[:2]
target_ratio = image_width / image_height
current_ratio = orig_width / orig_height

if current_ratio > target_ratio:
    # Crop width to match target aspect ratio
    new_width = max(1, int(round(orig_height * target_ratio)))
    x_offset = (orig_width - new_width) // 2
    img = img[:, x_offset : x_offset + new_width]
elif current_ratio < target_ratio:
    # Crop height to match target aspect ratio
    new_height = max(1, int(round(orig_width / target_ratio)))
    y_offset = (orig_height - new_height) // 2
    img = img[y_offset : y_offset + new_height, :]

# random rotation
angle = np.random.uniform(rotation_angle_range[0], rotation_angle_range[1])
if debug:
    print(f"Rotation angle: {angle:.2f} degrees")
img = rotate_and_center_crop(img, angle)
img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_AREA)
out = img.copy()

# oil effect
if np.random.rand() < oil_effect_probability:
    if debug:
        print("oil effect")
    oil_strength = np.random.uniform(oil_strength_range[0], oil_strength_range[1])
    out = oil_effect(out, strength=oil_strength, noisy_mask_scale=noisy_mask_scale)

# blisters effect
if np.random.rand() < blisters_effect_probability:
    if debug:
        print("blisters effect")
    blisters_count = np.random.randint(blisters_count_range[0], blisters_count_range[1])
    blisters_amount = np.random.uniform(blisters_amount_range[0], blisters_amount_range[1])
    out = blisters_effect(out, amount=blisters_amount, count=blisters_count)

# dust effect
if np.random.rand() < dust_effect_probability:
    if debug:
        print("dust effect")
    dust_amount = np.random.uniform(dust_amount_range[0], dust_amount_range[1])
    out = dust_effect(out, amount=dust_amount)

cv2.imshow('out', out)
cv2.waitKey(0)

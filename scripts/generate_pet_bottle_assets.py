

import os
import sys


# Ensure local modules remain importable when Blender launches this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from dataset.scripts.deformation_functions_pet_bottle_blend_file import *
from dataset.scripts.utilities import load_settings


input_dir = "/home/tem/Waste-Dataset-Generation/fbx_standard/pet"
output_dir = "/home/tem/Waste-Dataset-Generation/res_fbx_objects/pet"
label_image_dir = "/home/tem/Waste-Dataset-Generation/label_images"
prefix = "pet"
start_index = 1
num_needed = 10000 
seed=None
flatten_probability = 0.7
press_band_inward_twice_probability = 0.3
two_peaks_vertical_squash_probability = 0.3
label_probability = 0.5
blue_color_probability = 0.4
green_color_probability = 0.1
# white_color_probability = 1 - blue_color_probability - green_color_probability = 0.5
max_attempts_per_file = 3

blend_mode = 'BLEND'
roughness = 0.2
transmission = 1.0
alpha = 0.3
color_code = (1.0, 1.0, 1.0, 1.0)

default_roughness_range = (roughness, roughness)
default_transmission_range = (transmission, transmission)
default_alpha_range = (alpha, alpha)
default_color_ranges = {
    "r": (color_code[0], color_code[0]),
    "g": (color_code[1], color_code[1]),
    "b": (color_code[2], color_code[2]),
}

if seed is not None:
    np.random.seed(seed)

settings = load_settings(os.path.join(SCRIPT_DIR, "settings.json"))
pet_transparency = {}
if settings:
    dataset_settings = settings.get("DatasetSettings", {})
    categories = dataset_settings.get("categories", [])
    for category in categories:
        if category.get("name") == "pet":
            pet_transparency = category.get("transparentcy", {}) or {}
            break


def _parse_range(value, default_range):
    try:
        if isinstance(value, (list, tuple)):
            low = float(value[0])
            high = float(value[1] if len(value) > 1 else value[0])
            if high < low:
                low, high = high, low
            return (low, high)
    except (TypeError, ValueError):
        pass
    return default_range


def _sample_from_range(value_range):
    low, high = value_range
    if low == high:
        return low
    return np.random.uniform(low, high)


roughness_range = _parse_range(pet_transparency.get("roughness"), default_roughness_range)
transmission_range = _parse_range(pet_transparency.get("transmission"), default_transmission_range)
alpha_range = _parse_range(pet_transparency.get("alpha"), default_alpha_range)
color_settings = pet_transparency.get("color", {}) or {}
color_r_range = _parse_range(color_settings.get("r"), default_color_ranges["r"])
color_g_range = _parse_range(color_settings.get("g"), default_color_ranges["g"])
color_b_range = _parse_range(color_settings.get("b"), default_color_ranges["b"])

ensure_dir(output_dir)
pool = list_fbx_files(input_dir)
label_image_paths = list_png_files(label_image_dir)
log(f"Found {len(pool)} source FBX files in '{input_dir}'.")
log(f"Found {len(label_image_paths)} label images in '{label_image_dir}'.")

for i in range(num_needed):
    idx = start_index + i
    out_name = make_name(prefix, idx)
    out_path = os.path.join(output_dir, out_name + ".blend")
    attempt = 0
    while attempt < max_attempts_per_file:
        attempt += 1
        # choose random source
        src = pool[np.random.randint(0, len(pool))]
        label_image_path = label_image_paths[np.random.randint(0, len(label_image_paths))]
        roughness_value = _sample_from_range(roughness_range)
        transmission_value = _sample_from_range(transmission_range)
        alpha_value = _sample_from_range(alpha_range)
        color_code_value = (
            _sample_from_range(color_r_range),
            _sample_from_range(color_g_range),
            _sample_from_range(color_b_range),
            1.0,
        )

        # override color
        color_roll = np.random.random()
        if color_roll < blue_color_probability:
            print("blue pet")
            color_r = np.random.uniform(0.6, 0.68)
            color_g = np.random.uniform(0.88, 0.96)
            color_b = np.random.uniform(0.98, 1.0)        
            color_override = (color_r, color_g, color_b, 1.0)
        elif color_roll < blue_color_probability + green_color_probability:
            print("green pet")
            color_r = np.random.uniform(0.75, 0.83)
            color_g = np.random.uniform(0.9, 0.98)
            color_b = np.random.uniform(0.75, 0.85)
            color_override = (color_r, color_g, color_b, 1.0)
        else:
            print("white pet")
            color_p = np.random.uniform(0.95, 1.0)
            color_override = (color_p, color_p, color_p, 1.0)
        color_code_value = color_override

        log(f"[{i+1}/{num_needed}] Attempt {attempt}/{max_attempts_per_file} Source: {src} -> {out_path}")
        try:
            deform_save_pet_bottle(src,
                                   label_image_path,
                                   out_path,
                                   out_name,
                                   blend_mode, roughness_value, transmission_value, alpha_value, color_code_value,
                                   flatten_probability=flatten_probability,
                                   press_band_inward_twice_probability=press_band_inward_twice_probability,
                                   two_peaks_vertical_squash_probability=two_peaks_vertical_squash_probability,
                                   label_probability = label_probability)
            if os.path.exists(out_path):
                break
            log(f"Generation finished but file not found: {out_path}")
        except Exception as e:
            log(f"ERROR while generating {out_name} (attempt {attempt}): {e}")
    else:
        log(f"FAILED to generate {out_name} after {max_attempts_per_file} attempts")

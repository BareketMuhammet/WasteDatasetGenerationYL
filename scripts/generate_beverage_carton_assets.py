

import os
import sys


# Ensure local modules remain importable when Blender launches this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from dataset.scripts.deformation_functions_beverage_carton import *


input_dir = "/home/tem/Waste-Dataset-Generation/fbx_standard/beverage_carton"
output_dir = "/home/tem/Waste-Dataset-Generation/res_fbx_objects/beverage_carton"
prefix = "beverage_carton"
start_index = 1
num_needed = 10000
seed=None
flatten_probability = 0.7
press_band_inward_twice_probability = 0.3
two_peaks_vertical_squash_probability = 0.3
max_attempts_per_file = 3

if seed is not None:
        np.random.seed(seed)

ensure_dir(output_dir)
pool = list_fbx_files(input_dir)
log(f"Found {len(pool)} source FBX files in '{input_dir}'.")

for i in range(num_needed):
    idx = start_index + i
    out_name = make_name(prefix, idx)
    out_path = os.path.join(output_dir, out_name)
    attempt = 0
    while attempt < max_attempts_per_file:
        attempt += 1
        # choose random source
        src = pool[np.random.randint(0, len(pool))]
        log(f"[{i+1}/{num_needed}] Attempt {attempt}/{max_attempts_per_file} Source: {src} -> {out_path}")
        try:
            deform_export_beverage_carton(
                src,
                out_path,
                flatten_probability=flatten_probability,
                press_band_inward_twice_probability=press_band_inward_twice_probability,
                two_peaks_vertical_squash_probability=two_peaks_vertical_squash_probability,
            )
            if os.path.exists(out_path):
                break
            log(f"Generation finished but file not found: {out_path}")
        except Exception as e:
            log(f"ERROR while generating {out_name} (attempt {attempt}): {e}")
    else:
        log(f"FAILED to generate {out_name} after {max_attempts_per_file} attempts")

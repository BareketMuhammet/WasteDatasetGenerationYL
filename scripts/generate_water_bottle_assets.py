

import os
import sys


# Ensure local modules remain importable when Blender launches this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from dataset.scripts.deformation_functions_water_bottle import *


input_dir = "/home/tem/Waste-Dataset-Generation/fbx_standard/pet"
output_dir = "/home/tem/Waste-Dataset-Generation/res_fbx_objects/pet"
prefix = "pet"
start_index = 1
num_needed = 100 
seed=None
flatten_probability = 0.7
press_band_inward_twice_probability = 0.1
two_peaks_vertical_squash_probability = 0.1

if seed is not None:
        np.random.seed(seed)

ensure_dir(output_dir)
pool = list_fbx_files(input_dir)
log(f"Found {len(pool)} source FBX files in '{input_dir}'.")

for i in range(num_needed):
    idx = start_index + i
    out_name = make_name(prefix, idx)
    out_path = os.path.join(output_dir, out_name)
    # choose random source
    src = pool[np.random.randint(0, len(pool))]
    log(f"[{i+1}/{num_needed}] Source: {src} -> {out_path}")
    try:
        deform_export_water_bottle(src, out_path, flatten_probability=flatten_probability, press_band_inward_twice_probability=press_band_inward_twice_probability, two_peaks_vertical_squash_probability=two_peaks_vertical_squash_probability)
    except Exception as e:
        log(f"ERROR while generating {out_name}: {e}")
        # continue with the next sample
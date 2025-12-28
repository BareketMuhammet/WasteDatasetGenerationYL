#!/usr/bin/env python3
"""Clean up dataset by removing images and masks without corresponding depth/label files."""

import os
from pathlib import Path
import shutil


def cleanup_dataset(dataset_dir: str = "dataset"):
    """Remove image and mask files that don't have corresponding depth and label files."""
    
    dataset_path = Path(dataset_dir)
    
    # Define directories
    depth_dir = dataset_path / "depth"
    labels_dir = dataset_path / "labels"
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    
    # Check if directories exist
    for dir_path in [depth_dir, labels_dir, images_dir, masks_dir]:
        if not dir_path.exists():
            print(f"Warning: {dir_path} does not exist")
            return
    
    # Get base names of files that have BOTH depth and label files
    depth_files = {f.stem for f in depth_dir.glob("*.npz")}
    label_files = {f.stem for f in labels_dir.glob("*.json")}
    
    # Files must have both depth AND label
    valid_files = depth_files & label_files
    
    print(f"Found {len(depth_files)} depth files")
    print(f"Found {len(label_files)} label files")
    print(f"Valid files (with both depth and label): {len(valid_files)}")
    
    # Clean up images directory
    print("\n=== Cleaning images directory ===")
    images_removed = 0
    for image_file in images_dir.glob("*.png"):
        if image_file.stem not in valid_files:
            print(f"Removing image: {image_file.name}")
            image_file.unlink()
            images_removed += 1
    
    print(f"Removed {images_removed} orphaned image files")
    
    # Clean up masks directory
    print("\n=== Cleaning masks directory ===")
    mask_dirs_removed = 0
    if masks_dir.exists():
        for mask_subdir in masks_dir.iterdir():
            if mask_subdir.is_dir():
                # Extract the base name from the mask directory name
                # Mask directories are named like "000001", "000002", etc.
                dir_name = mask_subdir.name
                if dir_name not in valid_files:
                    print(f"Removing mask directory: {mask_subdir.name}")
                    shutil.rmtree(mask_subdir)
                    mask_dirs_removed += 1
    
    print(f"Removed {mask_dirs_removed} orphaned mask directories")
    
    # Summary
    print("\n=== Cleanup Summary ===")
    print(f"Valid files: {len(valid_files)}")
    print(f"Removed {images_removed} image files")
    print(f"Removed {mask_dirs_removed} mask directories")
    print(f"\nRemaining files:")
    print(f"  - Images: {len(list(images_dir.glob('*.png')))}")
    print(f"  - Depth: {len(list(depth_dir.glob('*.npz')))}")
    print(f"  - Labels: {len(list(labels_dir.glob('*.json')))}")
    print(f"  - Mask dirs: {len(list(masks_dir.iterdir()))}")


if __name__ == "__main__":
    cleanup_dataset()

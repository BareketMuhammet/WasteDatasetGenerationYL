
import os

def find_empty_folders(base_folder="masks"):
    empty_folders = []

    # Walk through all subdirectories of the base folder
    for root, dirs, files in os.walk(base_folder):
        # Only consider *direct* folders inside "masks", not nested deeper
        if root != base_folder:
            # Check if this folder has any .npz files
            npz_files = [f for f in files if f.lower().endswith('.npz')]
            if not npz_files:
                empty_folders.append(root)

    return empty_folders


if __name__ == "__main__":
    # dir = "/home/tem/Waste-Dataset-Generation/validate_dataset/masks"
    dir ="/home/tem/Waste-Dataset-Generation/train_dataset/masks"
    empty = find_empty_folders(dir)
    if empty:
        print("Folders with no .npz files:")
        for folder in empty:
            print(folder)
    else:
        print("No empty folders found.")

"""Entry point for bulk dataset generation.

Prepares dataset directories, snapshots configuration, generates ``categories.json``
from the active settings, and then launches Blender runs via
``generate_one_sample_data.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from dataset.scripts.utilities import copy_file, ensure_directories_exist, load_settings


def write_categories_file(settings: dict[str, Any], destination_dir: str) -> None:
    """Persist categories from ``settings`` into ``categories.json``."""
    dataset_settings = settings.get("DatasetSettings", {})
    categories = dataset_settings.get("categories")
    if not categories:
        print("No categories found in settings; skipping categories.json generation.")
        return

    # Keep only the category fields that downstream consumers expect.
    allowed_keys = ("id", "name", "supercategory", "probability")
    filtered_categories = [
        {key: category[key] for key in allowed_keys if key in category}
        for category in categories
    ]

    payload = {
        "version": dataset_settings.get("Version", "1.0"),
        "categories": filtered_categories,
    }

    categories_path = os.path.join(destination_dir, "categories.json")
    with open(categories_path, "w", encoding="utf-8") as categories_file:
        json.dump(payload, categories_file, indent=4)
    print(f"categories.json written to {categories_path}")


def main() -> None:
    settings = load_settings("settings.json")
    if settings is None:
        print("Settings could not be loaded correctly.")
        raise SystemExit(1)

    current_directory = os.getcwd()
    dataset_directory = settings["DatasetSettings"]["DatasetDirectory"]

    dataset_directory_path = os.path.join(current_directory, dataset_directory)
    temp_directory_path = os.path.join(current_directory, "temp")

    images_folder_path = os.path.join(dataset_directory_path, "images")
    depth_folder_path = os.path.join(dataset_directory_path, "depth")
    masks_folder_path = os.path.join(dataset_directory_path, "masks")
    labels_folder_path = os.path.join(dataset_directory_path, "labels")

    # Ensure output directories exist before Blender runs.
    ensure_directories_exist(
        dataset_directory_path,
        temp_directory_path,
        labels_folder_path,
        images_folder_path,
        depth_folder_path,
        masks_folder_path,
    )

    # Preserve the configuration used for this batch inside the dataset folder.
    settings_copy_path = os.path.join(dataset_directory_path, "settings.json")
    if os.path.abspath("settings.json") != os.path.abspath(settings_copy_path):
        copy_file("settings.json", dataset_directory_path, "settings.json")

    # Persist categories.json derived from the active settings.
    write_categories_file(settings, dataset_directory_path)

    # Launch parameters
    start_index = settings["DatasetSettings"]["StartIndex"]
    number_of_samples = settings["DatasetSettings"]["NumberOfSamples"]

    # Launch Blender for each sample in the requested index range.
    index = start_index
    while index < number_of_samples + start_index:
        print(f"Generating image, depth and label {index}")
        try:
            subprocess.run(
                [
                    "blender",
                    "--background",
                    "--python",
                    "generate_one_sample_data.py",
                    "--",
                    str(index),
                ],
                timeout=60,
                check=True,
            )
            index += 1
        except subprocess.TimeoutExpired:
            print(f"Image {index} timed out. Retrying...")
        except subprocess.CalledProcessError as exc:
            print(f"Image {index} crashed (exit code {exc.returncode}). Retrying...")


if __name__ == "__main__":
    main()

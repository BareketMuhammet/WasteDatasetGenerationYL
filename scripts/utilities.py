"""Shared utility helpers for dataset generation scripts.

The functions are grouped by domain so they are easier to find and reuse.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from typing import Any, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# JSON and settings helpers
# ---------------------------------------------------------------------------

def load_json(
    file_path: str,
    *,
    success_message: Optional[str] = None,
    empty_warning: Optional[str] = None,
) -> Optional[Any]:
    """Load JSON data from ``file_path`` and return it, or ``None`` on failure."""
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from the file '{file_path}'.")
        return None
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return None

    print(success_message or f"{file_path} loaded successfully.")
    if data is None:
        warning = empty_warning or f"{file_path} could not be loaded correctly."
        print(warning)
    return data


def load_settings(file_path: str = "settings.json") -> Optional[Any]:
    """Convenience wrapper to load the project settings JSON."""
    return load_json(
        file_path,
        success_message="Settings loaded successfully.",
        empty_warning="Settings could not be loaded correctly.",
    )


def save_settings(settings: Any, file_path: str = "settings.json") -> bool:
    """Persist ``settings`` as nicely formatted JSON."""
    try:
        with open(file_path, "w", encoding="utf-8") as settings_file:
            json.dump(settings, settings_file, indent=4)
    except TypeError as exc:
        print(f"Failed to serialize settings: {exc}")
        return False
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return False

    print("Settings saved successfully.")
    return True


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def create_directory(directory: str) -> None:
    """Create ``directory`` if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"{directory} directory created.")


def ensure_directories_exist(*directories: str) -> None:
    """Ensure each directory in ``directories`` exists."""
    for directory in directories:
        create_directory(directory)


def copy_file(file_path: str, destination_folder: str, new_name: str) -> None:
    """Copy ``file_path`` into ``destination_folder`` with ``new_name``."""
    new_path = os.path.join(destination_folder, new_name)
    shutil.copy(file_path, new_path)
    print(f"Copied {file_path} to {new_path}")


def list_files(folder_path: str, extension: str = "") -> list[str]:
    """Return all files under ``folder_path`` filtered by file ``extension``."""
    files: list[str] = []
    for root, _dirs, file_list in os.walk(folder_path):
        for file_name in file_list:
            if not extension or file_name.endswith(extension):
                files.append(os.path.join(root, file_name))
    return files


def replace_keywords_in_paths(file_list: list[str], old_keyword: str, new_keyword: str) -> list[str]:
    """Replace ``old_keyword`` with ``new_keyword`` in each path from ``file_list``."""
    replaced_files: list[str] = []
    for file_path in file_list:
        replaced_files.append(file_path.replace(old_keyword, new_keyword))
    return replaced_files


def empty_directory(directory: str) -> None:
    """Remove all files and sub-directories contained in ``directory``."""
    if not os.path.exists(directory):
        return

    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def move_file(source_path: str, destination_path: str) -> None:
    """Move ``source_path`` to ``destination_path``, creating parent folders if needed."""
    destination_dir = os.path.dirname(destination_path)
    if destination_dir and not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    shutil.move(source_path, destination_path)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def add_current_directory_to_sys_path() -> None:
    """Add the current directory to ``sys.path`` if it is not already there."""
    current_directory = os.getcwd()
    print("Current directory:", current_directory)
    if current_directory not in sys.path:
        sys.path.append(current_directory)
        print("Current directory added to sys.path")


# ---------------------------------------------------------------------------
# Imaging helpers
# ---------------------------------------------------------------------------

def color_temperature_to_rgb(kelvin: int) -> tuple[int, int, int]:
    """Convert a color temperature in Kelvin to an sRGB tuple."""
    kelvin = max(1000, min(kelvin, 40000))
    temp = kelvin / 100.0

    if temp <= 66:
        red = 255
    else:
        red = temp - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = min(255, max(0, red))

    if temp <= 66:
        green = temp
        green = 99.4708025861 * math.log(green) - 161.1195681661
        green = min(255, max(0, green))
    else:
        green = temp - 60
        green = 288.1221695283 * (green ** -0.0755148492)
        green = min(255, max(0, green))

    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        blue = temp - 10
        blue = 138.5177312231 * math.log(blue) - 305.0447927307
        blue = min(255, max(0, blue))

    return int(red), int(green), int(blue)


def get_sensor_size(pixel_size: float, image_width: int, image_height: int) -> tuple[float, float]:
    """Return the physical sensor dimensions in millimeters."""
    sensor_width_mm = image_width * pixel_size
    sensor_height_mm = image_height * pixel_size
    return sensor_width_mm, sensor_height_mm


def get_fov(sensor_width: float, sensor_height: float, focal_length_mm: float) -> tuple[float, float]:
    """Compute horizontal and vertical field-of-view in degrees."""
    hfov = 2 * math.atan(sensor_width / (2 * focal_length_mm))
    vfov = 2 * math.atan(sensor_height / (2 * focal_length_mm))
    return math.degrees(hfov), math.degrees(vfov)


def convert_jpg_mask_to_binary(mask: np.ndarray, threshold: int) -> np.ndarray:
    """Convert a grayscale JPG ``mask`` to a binary mask (uint8 0/1)."""
    _, binary_mask_255 = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    binary_mask = np.where(binary_mask_255 >= threshold, 1, 0).astype(np.uint8)
    return binary_mask

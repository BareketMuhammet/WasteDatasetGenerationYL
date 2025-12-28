"""Display all instance masks for a given dataset index sequentially."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from dataset.scripts.utilities import load_settings


# ---------------------------------------------------------------------------
# Configuration
# Adjust these constants to point to the dataset and index you want to inspect.
# Set DATASET_ROOT to None to fall back to the dataset directory in settings.json.
# ---------------------------------------------------------------------------
DATA_INDEX = 1
DATASET_ROOT: Path | None = Path("dataset")
SETTINGS_PATH = Path("settings.json")


def resolve_dataset_root(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    settings = load_settings(str(SETTINGS_PATH))
    if not settings:
        raise SystemExit("Unable to load settings.json to locate dataset directory.")

    dataset_settings = settings.get("DatasetSettings", {})
    dataset_dir = dataset_settings.get("DatasetDirectory")
    if not dataset_dir:
        raise SystemExit("DatasetDirectory is not defined in settings.json.")

    return Path(dataset_dir)


def iter_mask_files(mask_dir: Path) -> Iterable[Path]:
    for entry in sorted(mask_dir.glob("*.npz")):
        if entry.is_file():
            yield entry


def load_mask(mask_path: Path) -> np.ndarray:
    with np.load(mask_path) as data:
        if not data.files:
            raise ValueError(f"{mask_path} does not contain any arrays.")
        array = data[data.files[0]]

    if array.dtype != np.uint8:
        array = array.astype(np.uint8)

    if array.ndim != 2:
        raise ValueError(f"{mask_path} has unexpected shape {array.shape}.")

    return array


def show_masks(mask_paths: Iterable[Path], window_name: str) -> None:
    for mask_path in mask_paths:
        try:
            mask = load_mask(mask_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load {mask_path}: {exc}")
            continue

        display_mask = (mask * 255).clip(0, 255).astype(np.uint8)

        cv2.imshow(window_name, display_mask)
        key = cv2.waitKey(0) & 0xFF  # Wait for user input before advancing
        if key in (27, ord("q")):  # ESC or 'q'
            break


def main() -> None:
    dataset_root = resolve_dataset_root(DATASET_ROOT)
    masks_root = dataset_root / "masks"

    if not masks_root.exists():
        raise SystemExit(f"Masks directory not found: {masks_root}")

    index_str = f"{DATA_INDEX:06d}"
    mask_dir = masks_root / index_str
    if not mask_dir.exists():
        raise SystemExit(f"No masks found for index {index_str} in {masks_root}")

    mask_files = list(iter_mask_files(mask_dir))
    if not mask_files:
        raise SystemExit(f"No .npz mask files found in {mask_dir}")

    print(
        f"Displaying {len(mask_files)} mask(s) for index {index_str}. "
        "Press any key to advance, 'q' or ESC to quit."
    )
    show_masks(mask_files, f"Masks {index_str}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

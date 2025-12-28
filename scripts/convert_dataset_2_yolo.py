from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


# ----------------------- Configuration ------------------------------------- #
SOURCE_DATASET_DIR = Path("/home/tem/Waste-Sorting/Waste-Detection-Deep-Learning/validate_dataset")
IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"
MASKS_SUBDIR = "masks"
METADATA_FILENAME = "categories.json"

OUTPUT_DATASET_DIR = Path("yolo_dataset")
OVERWRITE_OUTPUT = True

MIN_CONTOUR_AREA = 5.0
APPROX_EPSILON_RATIO = 0.002

MAX_POLYGON_POINTS = None  # Optional hard limit; set to None to disable


# ----------------------- Data containers ----------------------------------- #
@dataclass(frozen=True)
class ImageRecord:
    stem: str
    image_path: Path
    width: int
    height: int
    label_lines: List[str]


# ----------------------- Helpers ------------------------------------------- #
def ensure_paths() -> tuple[Path, Path, Path, Path]:
    source_images = SOURCE_DATASET_DIR / IMAGES_SUBDIR
    source_labels = SOURCE_DATASET_DIR / LABELS_SUBDIR
    source_masks = SOURCE_DATASET_DIR / MASKS_SUBDIR
    categories_path = SOURCE_DATASET_DIR / METADATA_FILENAME

    for path in (SOURCE_DATASET_DIR, source_images, source_labels, source_masks):
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    if not categories_path.exists():
        raise FileNotFoundError(f"Missing categories file: {categories_path}")

    return source_images, source_labels, source_masks, categories_path


def load_categories(categories_path: Path) -> list[str]:
    with categories_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    categories = payload.get("categories")
    if not categories:
        raise ValueError(f"No categories found in {categories_path}")

    categories_by_id = sorted(categories, key=lambda item: item["id"])
    return [item["name"] for item in categories_by_id]


def load_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    data = np.load(mask_path)
    array = data[data.files[0]]
    if array.ndim != 2:
        raise ValueError(f"Unexpected mask shape at {mask_path}: {array.shape}")
    return np.ascontiguousarray((array > 0).astype(np.uint8) * 255)


def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return None

    epsilon = APPROX_EPSILON_RATIO * cv2.arcLength(contour, True)
    if epsilon > 0:
        contour = cv2.approxPolyDP(contour, epsilon, True)

    contour = contour.reshape(-1, 2)
    if contour.shape[0] < 3:
        return None

    if np.array_equal(contour[0], contour[-1]) and contour.shape[0] > 3:
        contour = contour[:-1]

    if MAX_POLYGON_POINTS is not None and contour.shape[0] > MAX_POLYGON_POINTS:
        indices = np.linspace(0, contour.shape[0] - 1, MAX_POLYGON_POINTS, dtype=int)
        contour = contour[indices]

    return contour.astype(np.float32)


def polygon_to_yolo_line(
    contour: np.ndarray, class_id: int, width: int, height: int
) -> str:
    norm = contour.copy()
    norm[:, 0] = norm[:, 0] / float(width)
    norm[:, 1] = norm[:, 1] / float(height)

    points: Iterable[str] = (
        f"{value:.6f}" for point in norm for value in point.tolist()
    )
    return f"{class_id} " + " ".join(points)


def gather_records(
    source_images: Path,
    source_labels: Path,
    source_masks: Path,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    for label_path in sorted(source_labels.glob("*.json")):
        with label_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        image_meta = payload["image"]
        file_name = image_meta["file_name"]
        stem = Path(file_name).stem
        width = int(image_meta["width"])
        height = int(image_meta["height"])

        image_path = source_images / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mask_dir = source_masks / stem
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory missing for {stem}: {mask_dir}")

        label_lines: list[str] = []

        for instance in payload.get("instances", []):
            instance_id = int(instance["instance_id"])
            class_id = int(instance["category_id"])
            mask_path = mask_dir / f"{stem}_{instance_id:04d}.npz"

            try:
                mask = load_mask(mask_path)
            except FileNotFoundError:
                print(f"[WARN] Mask missing for {mask_path}, skipping instance.")
                continue

            contour = mask_to_polygon(mask)
            if contour is None:
                print(
                    f"[WARN] Skipping instance {stem}#{instance_id} due to invalid contour."
                )
                continue

            label_line = polygon_to_yolo_line(contour, class_id, width, height)
            label_lines.append(label_line)

        records.append(
            ImageRecord(
                stem=stem,
                image_path=image_path,
                width=width,
                height=height,
                label_lines=label_lines,
            )
        )

    return records


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        if OVERWRITE_OUTPUT:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"Output directory {path} already exists. Set OVERWRITE_OUTPUT=True to replace."
            )
    path.mkdir(parents=True, exist_ok=True)


def export_records(records: list[ImageRecord]) -> None:
    image_out_dir = OUTPUT_DATASET_DIR / "images"
    label_out_dir = OUTPUT_DATASET_DIR / "labels"

    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        destination_image = image_out_dir / record.image_path.name
        shutil.copy2(record.image_path, destination_image)

        label_path = label_out_dir / f"{record.stem}.txt"
        with label_path.open("w", encoding="utf-8") as lf:
            if record.label_lines:
                lf.write("\n".join(record.label_lines))


def write_data_yaml(class_names: list[str], splits: dict[str, list[ImageRecord]]) -> None:
    yaml_lines = [
        f"path: {OUTPUT_DATASET_DIR.resolve()}",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
        "train: images",
        "val: images",
    ]

    if splits.get("test"):
        yaml_lines.append("test: images")

    yaml_content = "\n".join(yaml_lines) + "\n"
    data_path = OUTPUT_DATASET_DIR / "data.yaml"
    with data_path.open("w", encoding="utf-8") as f:
        f.write(yaml_content)


def main() -> None:
    print("Preparing YOLO-format dataset...")
    (
        source_images,
        source_labels,
        source_masks,
        categories_path,
    ) = ensure_paths()

    class_names = load_categories(categories_path)
    print(f"Loaded {len(class_names)} categories.")

    records = gather_records(source_images, source_labels, source_masks)
    print(f"Discovered {len(records)} labeled images.")
    if not records:
        raise ValueError("No annotated images found; aborting export.")

    prepare_output_dir(OUTPUT_DATASET_DIR)

    export_records(records)

    write_data_yaml(class_names, {"full": records})
    print(f"YOLO dataset written to {OUTPUT_DATASET_DIR.resolve()}")


if __name__ == "__main__":
    main()

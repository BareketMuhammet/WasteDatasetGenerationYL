from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml


# ----------------------- Configuration ------------------------------------- #
YOLO_DATASET_DIR = Path("/home/tem/Waste-Sorting-YOLO/datasets/train")
# YOLO_DATASET_DIR = Path("/home/tem/Waste-Sorting-YOLO/datasets/validate")
IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"
DATA_YAML_FILENAME = "data.yaml"

SHUFFLE_IMAGES = False
RANDOM_SEED = 42
MAX_IMAGES_TO_SHOW: Optional[int] = None  # Set to an integer to limit preview count
SHOW_CLASS_ID_OVERLAYS = True
CUSTOM_CLASS_COLORS: Optional[List[Tuple[int, int, int]]] = None  # Override per-class BGR colors
POLYGON_FILL_ALPHA = 0.4
LABEL_BACKGROUND = (0, 0, 0)  # BGR
LABEL_TEXT_COLOR = (255, 255, 255)  # BGR
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 1
LABEL_BOX_PADDING = 4


# ----------------------- Data containers ----------------------------------- #
@dataclass(frozen=True)
class SegmentationInstance:
    class_id: int
    polygon_xy: np.ndarray  # shape (N, 2) with pixel coordinates


@dataclass(frozen=True)
class ImageExample:
    image_path: Path
    label_path: Path
    width: int
    height: int
    instances: List[SegmentationInstance]


# ----------------------- Helpers ------------------------------------------- #
def ensure_dataset_paths() -> Tuple[Path, Path, Path]:
    images_dir = YOLO_DATASET_DIR / IMAGES_SUBDIR
    labels_dir = YOLO_DATASET_DIR / LABELS_SUBDIR
    data_yaml_path = YOLO_DATASET_DIR / DATA_YAML_FILENAME

    for path in (YOLO_DATASET_DIR, images_dir, labels_dir, data_yaml_path):
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")

    return images_dir, labels_dir, data_yaml_path


def load_class_names(data_yaml_path: Path) -> List[str]:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    names = payload.get("names")
    if names is None:
        raise ValueError(f"'names' field not found in {data_yaml_path}")

    if isinstance(names, dict):
        # Support mapping style {id: name}
        ordered = [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
        return ordered

    if isinstance(names, list):
        return [str(name) for name in names]

    raise TypeError(f"Unsupported 'names' format ({type(names)}) in {data_yaml_path}")


def parse_yolo_segmentation_line(
    line: str, width: int, height: int
) -> SegmentationInstance:
    parts = line.strip().split()
    if len(parts) < 7:
        raise ValueError(f"Invalid segmentation line (too few values): '{line}'")
    if len(parts[1:]) % 2 != 0:
        raise ValueError(f"Odd number of coordinate values in line: '{line}'")

    class_id = int(parts[0])
    coords = np.array([float(value) for value in parts[1:]], dtype=np.float32)
    polygon = coords.reshape(-1, 2)
    polygon[:, 0] *= width
    polygon[:, 1] *= height

    return SegmentationInstance(class_id=class_id, polygon_xy=polygon)


def load_example(image_path: Path, labels_dir: Path) -> ImageExample:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]
    label_path = labels_dir / f"{image_path.stem}.txt"

    instances: List[SegmentationInstance] = []
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as lf:
            for line_number, line in enumerate(lf, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    instance = parse_yolo_segmentation_line(stripped, width, height)
                    instances.append(instance)
                except Exception as exc:
                    raise ValueError(
                        f"Error parsing {label_path}:{line_number}: {exc}"
                    ) from exc

    return ImageExample(
        image_path=image_path,
        label_path=label_path,
        width=width,
        height=height,
        instances=instances,
    )


def iterate_examples(images_dir: Path, labels_dir: Path) -> Iterable[ImageExample]:
    image_paths = sorted(images_dir.glob("*"))
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = [path for path in image_paths if path.suffix.lower() in valid_extensions]

    if SHUFFLE_IMAGES:
        random.seed(RANDOM_SEED)
        random.shuffle(image_paths)

    if MAX_IMAGES_TO_SHOW is not None:
        image_paths = image_paths[:MAX_IMAGES_TO_SHOW]

    for image_path in image_paths:
        yield load_example(image_path, labels_dir)


def build_class_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    if CUSTOM_CLASS_COLORS:
        if len(CUSTOM_CLASS_COLORS) < num_classes:
            raise ValueError(
                f"Not enough custom colors ({len(CUSTOM_CLASS_COLORS)}) for {num_classes} classes."
            )
        return list(CUSTOM_CLASS_COLORS[:num_classes])

    if num_classes <= 0:
        return []

    colors: List[Tuple[int, int, int]] = []
    for idx in range(num_classes):
        hue = int(180 * idx / num_classes)
        hsv_color = np.uint8([[[hue, 200, 255]]])  # type: ignore[arg-type]
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(channel) for channel in bgr_color))

    return colors


def draw_label(image: np.ndarray, text: str, position: Tuple[int, int]) -> None:
    text_size, baseline = cv2.getTextSize(
        text, LABEL_FONT, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
    )
    x, y = position
    x = max(x, 0)
    y = max(y, 0)

    box_width = text_size[0] + 2 * LABEL_BOX_PADDING
    box_height = text_size[1] + 2 * LABEL_BOX_PADDING

    top_left = (int(x - box_width / 2), int(y - box_height / 2))
    bottom_right = (top_left[0] + box_width, top_left[1] + box_height)

    cv2.rectangle(image, top_left, bottom_right, LABEL_BACKGROUND, thickness=-1)

    text_origin = (top_left[0] + LABEL_BOX_PADDING, bottom_right[1] - LABEL_BOX_PADDING - baseline)
    cv2.putText(
        image,
        text,
        text_origin,
        LABEL_FONT,
        LABEL_FONT_SCALE,
        LABEL_TEXT_COLOR,
        LABEL_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def visualize_example(
    example: ImageExample, class_names: List[str], class_colors: List[Tuple[int, int, int]]
) -> np.ndarray:
    image = cv2.imread(str(example.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image for visualization: {example.image_path}")

    overlay = image.copy()

    for instance in example.instances:
        polygon = instance.polygon_xy.astype(np.int32)
        polygon = polygon.reshape(-1, 1, 2)

        color = class_colors[instance.class_id % len(class_colors)] if class_colors else (0, 0, 255)

        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)

        if SHOW_CLASS_ID_OVERLAYS:
            class_name = (
                class_names[instance.class_id]
                if 0 <= instance.class_id < len(class_names)
                else f"class_{instance.class_id}"
            )
            centroid = np.mean(polygon.reshape(-1, 2), axis=0)
            draw_label(overlay, class_name, (int(centroid[0]), int(centroid[1])))

    blended = cv2.addWeighted(overlay, POLYGON_FILL_ALPHA, image, 1 - POLYGON_FILL_ALPHA, 0)
    cv2.putText(
        blended,
        example.image_path.name,
        (10, 25),
        LABEL_FONT,
        LABEL_FONT_SCALE,
        (255, 255, 255),
        LABEL_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    return blended


# ----------------------- Main routine -------------------------------------- #
def main() -> None:
    images_dir, labels_dir, data_yaml_path = ensure_dataset_paths()
    class_names = load_class_names(data_yaml_path)
    class_colors = build_class_colors(len(class_names))

    examples = list(iterate_examples(images_dir, labels_dir))
    if not examples:
        print("No images found to display.")
        return

    print(f"Loaded {len(examples)} image(s) for inspection.")

    cv2.namedWindow("YOLO Dataset Preview", cv2.WINDOW_NORMAL)

    for idx, example in enumerate(examples, start=1):
        display = visualize_example(example, class_names, class_colors)
        cv2.imshow("YOLO Dataset Preview", display)
        print(f"[{idx}/{len(examples)}] Showing {example.image_path.name}")

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):  # ESC
            print("Exiting viewer.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

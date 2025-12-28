from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DatasetMetrics:
    num_images: int
    total_annotations: int
    objects_per_image: List[int]
    class_counts_per_image: Dict[str, List[int]]
    bbox_widths: List[float]
    bbox_heights: List[float]
    bbox_areas: List[float]
    bbox_centers_x: List[float]
    bbox_centers_y: List[float]
    image_dimension_counts: Dict[Tuple[int, int], int]
    class_names: List[str]
    bbox_areas_by_class: Dict[str, List[float]]
    mask_areas: List[float]
    mask_perimeters: List[float]
    mask_area_ratios: List[float]
    mask_compactness: List[float]
    mask_density_map: Optional[np.ndarray]


def load_categories(categories_path: Path) -> Dict[int, str]:
    """Return a mapping from category id to category name."""
    with categories_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    categories = data.get("categories", [])
    return {category["id"]: category["name"] for category in categories}


def _compute_mask_perimeter(mask: np.ndarray) -> int:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return 0

    up = np.zeros_like(mask_bool)
    up[1:] = mask_bool[:-1]

    down = np.zeros_like(mask_bool)
    down[:-1] = mask_bool[1:]

    left = np.zeros_like(mask_bool)
    left[:, 1:] = mask_bool[:, :-1]

    right = np.zeros_like(mask_bool)
    right[:, :-1] = mask_bool[:, 1:]

    perimeter_edges = (
        (mask_bool & ~up)
        + (mask_bool & ~down)
        + (mask_bool & ~left)
        + (mask_bool & ~right)
    )
    return int(perimeter_edges.sum())


def collect_dataset_metrics(labels_dir: Path, categories: Dict[int, str]) -> DatasetMetrics:
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        raise ValueError(f"No label files found in {labels_dir}")

    class_names = [categories[idx] for idx in sorted(categories.keys())]
    class_counts_per_image: Dict[str, List[int]] = {name: [] for name in class_names}
    bbox_areas_by_class: Dict[str, List[float]] = {name: [] for name in class_names}
    masks_root = labels_dir.parent / "masks"

    objects_per_image: List[int] = []
    bbox_widths: List[float] = []
    bbox_heights: List[float] = []
    bbox_areas: List[float] = []
    bbox_centers_x: List[float] = []
    bbox_centers_y: List[float] = []
    dimension_counts: Counter[Tuple[int, int]] = Counter()
    total_annotations = 0
    mask_areas: List[float] = []
    mask_perimeters: List[float] = []
    mask_area_ratios: List[float] = []
    mask_compactness: List[float] = []
    mask_density_accumulators: Dict[Tuple[int, int], np.ndarray] = {}
    mask_counts_by_shape: Counter[Tuple[int, int]] = Counter()

    for file_path in label_files:
        with file_path.open("r", encoding="utf-8") as handle:
            label_data = json.load(handle)

        image_info = label_data.get("image", {})
        instances = label_data.get("instances", [])

        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        dimension_counts[(width, height)] += 1

        objects_per_image.append(len(instances))
        total_annotations += len(instances)

        per_image_class_counter: Dict[str, int] = defaultdict(int)
        for instance in instances:
            category_id = instance.get("category_id")
            class_name = categories.get(category_id, f"category_{category_id}")
            per_image_class_counter[class_name] += 1

            bbox_x, bbox_y, bbox_w, bbox_h = instance.get("bbox_xywh", [0, 0, 0, 0])
            bbox_widths.append(float(bbox_w))
            bbox_heights.append(float(bbox_h))
            area = float(bbox_w) * float(bbox_h)
            bbox_areas.append(area)
            bbox_centers_x.append(float(bbox_x) + float(bbox_w) / 2.0)
            bbox_centers_y.append(float(bbox_y) + float(bbox_h) / 2.0)
            bbox_areas_by_class.setdefault(class_name, []).append(area)

        for class_name in class_counts_per_image.keys():
            class_counts_per_image[class_name].append(per_image_class_counter.get(class_name, 0))

        for class_name in per_image_class_counter.keys():
            if class_name not in class_counts_per_image:
                # Add new classes on the fly if they were not in categories.json
                existing_length = len(objects_per_image) - 1
                class_counts_per_image[class_name] = [0] * existing_length + [per_image_class_counter[class_name]]
                bbox_areas_by_class.setdefault(class_name, [])

        mask_dir = masks_root / file_path.stem
        if mask_dir.exists():
            for mask_file in sorted(mask_dir.glob("*.npz")):
                try:
                    with np.load(mask_file) as mask_data:
                        if not mask_data.files:
                            continue
                        mask_array = mask_data[mask_data.files[0]]
                except Exception:  # noqa: BLE001
                    continue

                mask_bool = mask_array.astype(bool)
                area = float(mask_bool.sum())
                perimeter = float(_compute_mask_perimeter(mask_bool))
                area_ratio = area / mask_bool.size if mask_bool.size else 0.0
                compactness = area / (perimeter ** 2) if perimeter else 0.0

                mask_areas.append(area)
                mask_perimeters.append(perimeter)
                mask_area_ratios.append(area_ratio)
                mask_compactness.append(compactness)

                shape = mask_bool.shape
                accumulator = mask_density_accumulators.setdefault(
                    shape, np.zeros(shape, dtype=np.float64)
                )
                accumulator += mask_bool.astype(np.float64)
                mask_counts_by_shape[shape] += 1

    num_images = len(objects_per_image)
    class_names = list(class_counts_per_image.keys())

    mask_density_map: Optional[np.ndarray] = None
    if mask_counts_by_shape:
        target_shape, shape_count = max(mask_counts_by_shape.items(), key=lambda item: item[1])
        accumulator = mask_density_accumulators[target_shape]
        mask_density_map = accumulator / float(shape_count) if shape_count else accumulator

    return DatasetMetrics(
        num_images=num_images,
        total_annotations=total_annotations,
        objects_per_image=objects_per_image,
        class_counts_per_image=class_counts_per_image,
        bbox_widths=bbox_widths,
        bbox_heights=bbox_heights,
        bbox_areas=bbox_areas,
        bbox_centers_x=bbox_centers_x,
        bbox_centers_y=bbox_centers_y,
        image_dimension_counts=dict(dimension_counts),
        class_names=class_names,
        bbox_areas_by_class=bbox_areas_by_class,
        mask_areas=mask_areas,
        mask_perimeters=mask_perimeters,
        mask_area_ratios=mask_area_ratios,
        mask_compactness=mask_compactness,
        mask_density_map=mask_density_map,
    )


def compute_summary_statistics(metrics: DatasetMetrics) -> Dict[str, object]:
    if metrics.num_images == 0:
        raise ValueError("Dataset has no images to summarize")

    avg_objects = metrics.total_annotations / metrics.num_images if metrics.num_images else 0.0
    avg_per_class = {
        class_name: (sum(counts) / metrics.num_images if metrics.num_images else 0.0)
        for class_name, counts in metrics.class_counts_per_image.items()
    }
    total_per_class = {class_name: int(sum(counts)) for class_name, counts in metrics.class_counts_per_image.items()}

    image_dimensions = [
        {"width": width, "height": height, "count": count}
        for (width, height), count in sorted(metrics.image_dimension_counts.items())
    ]

    summary = {
        "number_of_images": metrics.num_images,
        "total_number_of_annotations": metrics.total_annotations,
        "average_number_of_objects_per_image": round(avg_objects, 4),
        "image_dimensions": image_dimensions,
        "object_classes": metrics.class_names,
        "average_number_of_each_class_per_image": {
            class_name: round(value, 4) for class_name, value in avg_per_class.items()
        },
        "total_instances_per_class": total_per_class,
    }

    if metrics.mask_areas:
        mask_count = len(metrics.mask_areas)
        avg_mask_area = sum(metrics.mask_areas) / mask_count if mask_count else 0.0
        avg_mask_perimeter = sum(metrics.mask_perimeters) / mask_count if mask_count else 0.0
        avg_mask_ratio = sum(metrics.mask_area_ratios) / mask_count if mask_count else 0.0
        avg_mask_compactness = sum(metrics.mask_compactness) / mask_count if mask_count else 0.0
        summary["mask_statistics"] = {
            "number_of_masks": mask_count,
            "average_mask_area_pixels": round(avg_mask_area, 4),
            "average_mask_perimeter_pixels": round(avg_mask_perimeter, 4),
            "average_mask_area_ratio": round(avg_mask_ratio, 6),
            "average_mask_compactness": round(avg_mask_compactness, 6),
        }

    return summary


def save_statistics_json(statistics: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=4)


def generate_visualizations(metrics: DatasetMetrics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_objects_per_image_histogram(metrics.objects_per_image, output_dir / "hist_objects_per_image.png")
    _plot_class_counts_histogram(metrics.class_counts_per_image, output_dir / "hist_class_counts_per_image.png")
    _plot_bbox_scatter(metrics.bbox_widths, metrics.bbox_heights, output_dir / "scatter_bbox_width_vs_height.png")
    _plot_histogram(metrics.bbox_areas, "Bounding Box Areas", "Area (px^2)", output_dir / "hist_bbox_area.png")
    _plot_histogram(metrics.bbox_widths, "Bounding Box Widths", "Width (px)", output_dir / "hist_bbox_width.png")
    _plot_histogram(metrics.bbox_heights, "Bounding Box Heights", "Height (px)", output_dir / "hist_bbox_height.png")
    _plot_boxplot(metrics.bbox_areas_by_class, output_dir / "boxplot_bbox_area_per_class.png")
    _plot_heatmap(metrics.bbox_centers_x, metrics.bbox_centers_y, metrics.image_dimension_counts, output_dir / "heatmap_bbox_centers.png")
    _plot_mask_area_vs_perimeter_scatter(
        metrics.mask_areas,
        metrics.mask_perimeters,
        output_dir / "scatter_mask_area_vs_perimeter.png",
    )
    _plot_histogram(
        metrics.mask_area_ratios,
        "Mask Area to Image Area Ratio",
        "Area Ratio",
        output_dir / "hist_mask_area_ratio.png",
    )
    _plot_histogram(
        metrics.mask_compactness,
        "Mask Compactness (Area / Perimeter^2)",
        "Compactness",
        output_dir / "hist_mask_compactness.png",
    )
    _plot_mask_density_map(metrics.mask_density_map, output_dir / "density_mask_frequency.png")


def _plot_objects_per_image_histogram(objects_per_image: List[int], output_path: Path) -> None:
    if not objects_per_image:
        return

    bins = range(0, max(objects_per_image) + 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(objects_per_image, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_title("Histogram of Objects per Image")
    ax.set_xlabel("Number of objects")
    ax.set_ylabel("Frequency")
    ax.set_xticks(list(bins))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_class_counts_histogram(class_counts: Dict[str, List[int]], output_path: Path) -> None:
    if not class_counts:
        return

    max_count = max((max(counts) for counts in class_counts.values() if counts), default=0)
    bins = range(0, max_count + 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    for class_name, counts in class_counts.items():
        if not counts:
            continue
        ax.hist(counts, bins=bins, alpha=0.5, label=class_name, edgecolor="black")

    ax.set_title("Histogram of Class Counts per Image")
    ax.set_xlabel("Objects per image")
    ax.set_ylabel("Frequency")
    ax.set_xticks(list(bins))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_bbox_scatter(widths: List[float], heights: List[float], output_path: Path) -> None:
    if not widths or not heights:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(widths, heights, alpha=0.6, edgecolors="none")
    ax.set_title("Bounding Box Width vs Height")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_histogram(data: List[float], title: str, xlabel: str, output_path: Path) -> None:
    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data, bins="auto", edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_boxplot(data_by_class: Dict[str, List[float]], output_path: Path) -> None:
    valid_items = [(class_name, areas) for class_name, areas in data_by_class.items() if areas]
    if not valid_items:
        return

    labels, data = zip(*valid_items)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_title("Bounding Box Area Distribution per Class")
    ax.set_ylabel("Area (px^2)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_heatmap(
    centers_x: List[float],
    centers_y: List[float],
    image_dimensions: Dict[Tuple[int, int], int],
    output_path: Path,
) -> None:
    if not centers_x or not centers_y:
        return

    max_width = max((dims[0] for dims in image_dimensions.keys()), default=0)
    max_height = max((dims[1] for dims in image_dimensions.keys()), default=0)
    if max_width == 0 or max_height == 0:
        return

    aspect_ratio = max_height / max_width if max_width else 1.0
    fig_width = 8
    fig_height = max(3.0, fig_width * aspect_ratio)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    heatmap, xedges, yedges, img = ax.hist2d(
        centers_x,
        centers_y,
        bins=50,
        range=[[0, max_width], [0, max_height]],
        cmap="viridis",
    )
    ax.set_title("Heatmap of Bounding Box Centers")
    ax.set_xlabel("X center (px)")
    ax.set_ylabel("Y center (px)")
    ax.invert_yaxis()
    ax.set_aspect('equal')
    fig.colorbar(img, ax=ax, label="Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_mask_area_vs_perimeter_scatter(
    mask_areas: List[float], mask_perimeters: List[float], output_path: Path
) -> None:
    if not mask_areas or not mask_perimeters:
        return

    data_length = min(len(mask_areas), len(mask_perimeters))
    if data_length == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mask_perimeters[:data_length], mask_areas[:data_length], alpha=0.6, edgecolors="none")
    ax.set_title("Mask Area vs Perimeter")
    ax.set_xlabel("Perimeter (px)")
    ax.set_ylabel("Area (px^2)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_mask_density_map(mask_density_map: Optional[np.ndarray], output_path: Path) -> None:
    if mask_density_map is None:
        return

    if mask_density_map.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(mask_density_map, cmap="inferno", origin="upper")
    ax.set_title("Mask Frequency Density Map")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    fig.colorbar(img, ax=ax, label="Average mask coverage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

from __future__ import annotations

import sys
from pathlib import Path

from dataset.scripts.analyze_dataset_functions import (
    collect_dataset_metrics,
    compute_summary_statistics,
    generate_visualizations,
    load_categories,
    save_statistics_json,
)

# Configuration parameters for the analysis run.
DATASET_DIR = Path("dataset")
DATASET_DIR = Path("/home/tem/Waste-Sorting/Waste-Detection-Deep-Learning/dataset")
ANALYSIS_SUBDIR = "analysis_results"
STATS_FILENAME = "dataset_statistics.json"
VISUALIZATIONS_SUBDIR = "visualizations"


def main() -> int:
    dataset_dir = DATASET_DIR
    labels_dir = dataset_dir / "labels"
    categories_path = dataset_dir / "categories.json"

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}", file=sys.stderr)
        return 1

    if not categories_path.exists():
        print(f"categories.json not found at: {categories_path}", file=sys.stderr)
        return 1

    try:
        categories = load_categories(categories_path)
        metrics = collect_dataset_metrics(labels_dir, categories)
        summary = compute_summary_statistics(metrics)
    except Exception as error:  # noqa: BLE001
        print(f"Failed to analyze dataset: {error}", file=sys.stderr)
        return 1

    output_dir = dataset_dir / ANALYSIS_SUBDIR
    stats_path = output_dir / STATS_FILENAME
    visuals_dir = output_dir / VISUALIZATIONS_SUBDIR

    try:
        save_statistics_json(summary, stats_path)
        generate_visualizations(metrics, visuals_dir)
    except Exception as error:  # noqa: BLE001
        print(f"Failed to save analysis outputs: {error}", file=sys.stderr)
        return 1

    print(f"Dataset statistics saved to: {stats_path}")
    print(f"Visualizations saved under: {visuals_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

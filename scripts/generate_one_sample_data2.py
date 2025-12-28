from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import bpy  # type: ignore
import cv2 as cv
import numpy as np

# Ensure local modules remain importable when Blender launches this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from dataset.scripts.blender_pipeline_functions import (  # noqa: E402
    add_light,
    add_object_from_fbx_file,
    add_output_slot,
    add_rigid_body,
    add_working_area,
    clear_cache,
    create_camera,
    create_depth_output_node,
    create_id_musk_node,
    create_output_node,
    create_render_node,
    delete_all,
    delete_all_nodes,
    link_nodes,
    make_object_transparent,
    remove_transparency,
    set_object_active,
    set_render_device,
    set_scene_settings,
    set_color_management,
    restore_color_management,
    append_single_object,
)
from dataset.scripts.background_aug_functions import (  # noqa: E402
    blisters_effect,
    dust_effect,
    oil_effect,
    rotate_and_center_crop,
)
from dataset.scripts.utilities import (  # noqa: E402
    add_current_directory_to_sys_path,
    color_temperature_to_rgb,
    convert_jpg_mask_to_binary,
    create_directory,
    ensure_directories_exist,
    empty_directory,
    load_json,
    move_file,
)


def _parse_axis_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return default
    return default


def _ensure_vec3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return (scalar, scalar, scalar)
    if isinstance(value, (list, tuple)):
        if len(value) == 3:
            try:
                return tuple(float(component) for component in value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default
        if len(value) == 1:
            scalar = float(value[0])
            return (scalar, scalar, scalar)
    return default


def _parse_float_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return (scalar, scalar)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            try:
                scalar = float(value[0])
                return (scalar, scalar)
            except (TypeError, ValueError):
                return default
        if len(value) >= 2:
            try:
                low = float(value[0])
                high = float(value[1])
                if high < low:
                    low, high = high, low
                return (low, high)
            except (TypeError, ValueError):
                return default
    return default


def _parse_int_range(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            try:
                scalar = int(round(float(value[0])))
                return (scalar, scalar)
            except (TypeError, ValueError):
                return default
        if len(value) >= 2:
            try:
                low = int(round(float(value[0])))
                high = int(round(float(value[1])))
                if high < low:
                    low, high = high, low
                return (low, high)
            except (TypeError, ValueError):
                return default
    return default


def _parse_color_ranges(
    value: Any,
    default: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if isinstance(value, dict):
        r = _parse_float_range(value.get("r"), default[0])
        g = _parse_float_range(value.get("g"), default[1])
        b = _parse_float_range(value.get("b"), default[2])
        return (r, g, b)

    if isinstance(value, (list, tuple)):
        channels = []
        for idx in range(3):
            try:
                channel_value = value[idx]
            except IndexError:
                channel_value = default[idx]
            channels.append(_parse_float_range(channel_value, default[idx]))
        return tuple(channels)  # type: ignore[return-value]

    return default


def _sample_from_range(value_range: tuple[float, float]) -> float:
    low, high = value_range
    if low == high:
        return low
    return random.uniform(low, high)


def _has_unclassified_assets(unclassified_assets: dict[str, list[str]]) -> bool:
    return any(unclassified_assets.get(key) for key in ("transparent", "untransparent"))


def _take_unclassified_asset(
    unclassified_assets: dict[str, list[str]],
    allow_repeat: bool,
) -> Optional[tuple[str, bool]]:
    available_types = [key for key, assets in unclassified_assets.items() if assets]
    if not available_types:
        return None

    if len(available_types) == 2:
        chosen_type = "transparent" if random.random() < 0.5 else "untransparent"
    else:
        chosen_type = available_types[0]

    assets = unclassified_assets[chosen_type]
    asset_path = random.choice(assets)
    if not allow_repeat:
        assets.remove(asset_path)
    apply_transparency = chosen_type == "transparent"
    return asset_path, apply_transparency


@dataclass
class TransparencyConfig:
    enabled: bool
    roughness_range: tuple[float, float]
    transmission_range: tuple[float, float]
    alpha_range: tuple[float, float]
    color_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    @classmethod
    def from_dict(
        cls,
        data: Optional[dict[str, Any]],
        *,
        default_enabled: bool = False,
    ) -> TransparencyConfig:
        data = data or {}
        enabled_value = data.get("transparent", default_enabled)
        enabled = bool(enabled_value)
        roughness = _parse_float_range(data.get("roughness"), (0.0, 0.0))
        transmission = _parse_float_range(data.get("transmission"), (0.0, 0.0))
        alpha = _parse_float_range(data.get("alpha"), (1.0, 1.0))
        color_ranges = _parse_color_ranges(
            data.get("color"),
            ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        )
        return cls(enabled, roughness, transmission, alpha, color_ranges)

    @staticmethod
    def default(enabled: bool = False) -> TransparencyConfig:
        return TransparencyConfig(
            enabled,
            (0.0, 0.0),
            (0.0, 0.0),
            (1.0, 1.0),
            ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        )

    def sample_parameters(self) -> tuple[float, float, float, tuple[float, float, float, float]]:
        roughness = _sample_from_range(self.roughness_range)
        transmission = _sample_from_range(self.transmission_range)
        alpha = _sample_from_range(self.alpha_range)
        r = _sample_from_range(self.color_ranges[0])
        g = _sample_from_range(self.color_ranges[1])
        b = _sample_from_range(self.color_ranges[2])
        color = (r, g, b, 1.0)
        return roughness, transmission, alpha, color


@dataclass
class CategoryConfig:
    id: int
    name: str
    supercategory: str
    probability: float
    transparency: TransparencyConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoryConfig:
        if "id" not in data or "name" not in data:
            raise ValueError("Each category must define 'id' and 'name'.")
        transparency = TransparencyConfig.from_dict(data.get("transparentcy"))
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            supercategory=str(data.get("supercategory", "")),
            probability=float(data.get("probability", 1.0)),
            transparency=transparency,
        )


@dataclass
class DatasetConfig:
    version: str
    directory: str
    number_of_samples: int
    start_index: int
    categories: list[CategoryConfig]
    unclassified_probability: float
    unclassified_transparency: TransparencyConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        if not data:
            raise ValueError("DatasetSettings are missing.")

        categories_data = data.get("categories", [])
        if not categories_data:
            raise ValueError("DatasetSettings.categories is empty.")
        categories = [CategoryConfig.from_dict(cat) for cat in categories_data]

        unclassified_source = (
            data.get("UnclassifiedSettings")
            or data.get("Unclassified")
            or data.get("unclassified")
        )
        if isinstance(unclassified_source, dict):
            unclassified_probability = float(unclassified_source.get("probability", data.get("UnclassifiedProbability", 0.0)))
        else:
            unclassified_probability = float(data.get("UnclassifiedProbability", 0.0))
        transparency_data: Optional[dict[str, Any]] = None
        default_enabled = False
        if isinstance(unclassified_source, dict):
            transparency_data = unclassified_source.get("transparentcy") or unclassified_source
            default_enabled = bool(unclassified_source.get("transparent", False))

        unclassified_transparency = TransparencyConfig.from_dict(
            transparency_data,
            default_enabled=default_enabled,
        )

        return cls(
            version=str(data.get("Version", "")),
            directory=str(data["DatasetDirectory"]),
            number_of_samples=int(data.get("NumberOfSamples", 0)),
            start_index=int(data.get("StartIndex", 1)),
            categories=categories,
            unclassified_probability=unclassified_probability,
            unclassified_transparency=unclassified_transparency,
        )


@dataclass
class RenderConfig:
    engine: str
    samples: int
    device: str
    start_frame: int
    end_frame: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderConfig:
        if not data:
            raise ValueError("RenderSettings are missing.")
        return cls(
            engine=str(data.get("RenderEngine", "CYCLES")),
            samples=int(data.get("CyclesSamples", 16)),
            device=str(data.get("RenderDevice", "GPU")),
            start_frame=int(data.get("StartFrame", 1)),
            end_frame=int(data.get("EndFrame", 250)),
        )


@dataclass
class CameraConfig:
    sensor_width: float
    sensor_height: float
    focal_length: float
    image_width: int
    image_height: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
        if not data:
            raise ValueError("CameraSettings are missing.")
        return cls(
            sensor_width=float(data.get("SensorWidth", 36.0)),
            sensor_height=float(data.get("SensorHeight", 24.0)),
            focal_length=float(data.get("FocalLength", 35.0)),
            image_width=int(data.get("ImageWidth", 640)),
            image_height=int(data.get("ImageHeight", 480)),
        )


@dataclass
class LightingConfig:
    min_sources: int
    max_sources: int
    radius: float
    power_min: float
    power_max: float
    temp_min: float
    temp_max: float
    location_limits: tuple[float, float, float, float, float, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LightingConfig:
        if not data:
            raise ValueError("LightingSettings are missing.")
        limits = data.get("LocationLimits", [-0.4, 0.4, -0.4, 0.4, 0.0, 0.1])
        if not isinstance(limits, (list, tuple)) or len(limits) != 6:
            limits = [-0.4, 0.4, -0.4, 0.4, 0.0, 0.1]
        limits_tuple = tuple(float(value) for value in limits)  # type: ignore[misc]
        return cls(
            min_sources=int(data.get("MinNumberOfSources", 1)),
            max_sources=int(data.get("MaxNumberOfSources", 3)),
            radius=float(data.get("Radius", 0.1)),
            power_min=float(data.get("MinPower", 10.0)),
            power_max=float(data.get("MaxPower", 50.0)),
            temp_min=float(data.get("MinColorTemperature", 5000)),
            temp_max=float(data.get("MaxColorTemperature", 7000)),
            location_limits=limits_tuple,  # type: ignore[arg-type]
        )


@dataclass
class SpawnBounds:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @classmethod
    def from_settings(cls, data: dict[str, Any]) -> SpawnBounds:
        default_x = (-0.4, 0.4)
        default_y = (-0.25, 0.25)
        default_z = (0.0, 0.8)
        bounds = data.get("SpawnBounds")
        if isinstance(bounds, dict):
            x = _parse_axis_range(bounds.get("X"), default_x)
            y = _parse_axis_range(bounds.get("Y"), default_y)
            z = _parse_axis_range(bounds.get("Z"), default_z)
        elif isinstance(bounds, (list, tuple)) and len(bounds) == 6:
            x = (float(bounds[0]), float(bounds[1]))
            y = (float(bounds[2]), float(bounds[3]))
            z = (float(bounds[4]), float(bounds[5]))
        else:
            x, y, z = default_x, default_y, default_z
        return cls(x, y, z)


@dataclass
class RotationBounds:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @classmethod
    def from_settings(cls, data: dict[str, Any]) -> RotationBounds:
        default = (0.0, 360.0)
        rotation = data.get("RotationRange")
        if isinstance(rotation, dict):
            x = _parse_axis_range(rotation.get("X"), default)
            y = _parse_axis_range(rotation.get("Y"), default)
            z = _parse_axis_range(rotation.get("Z"), default)
        elif isinstance(rotation, (list, tuple)) and len(rotation) == 3:
            x = _parse_axis_range(rotation[0], default)
            y = _parse_axis_range(rotation[1], default)
            z = _parse_axis_range(rotation[2], default)
        else:
            x = y = z = default
        return cls(x, y, z)

    def random_euler(self) -> tuple[float, float, float]:
        return tuple(
            math.radians(random.uniform(*axis_range))
            for axis_range in (self.x, self.y, self.z)
        )


@dataclass
class ObjectsConfig:
    directory: str
    min_instances: int
    max_instances: int
    repeat: bool
    spawn_bounds: SpawnBounds
    rotation_bounds: RotationBounds

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectsConfig:
        if not data:
            raise ValueError("ObjectsSettings are missing.")
        min_instances = int(data.get("MinNumberOfInstances", 1))
        max_instances = int(data.get("MaxNumberOfInstances", max(min_instances, 1)))
        if max_instances < min_instances:
            max_instances = min_instances
        return cls(
            directory=str(data["Directory"]),
            min_instances=min_instances,
            max_instances=max_instances,
            repeat=bool(data.get("Repeat", True)),
            spawn_bounds=SpawnBounds.from_settings(data),
            rotation_bounds=RotationBounds.from_settings(data),
        )


@dataclass
class BackgroundAugmentationConfig:
    rotation_angle_range: tuple[float, float]
    oil_probability: float
    oil_strength_range: tuple[float, float]
    noisy_mask_scale: tuple[int, int]
    blisters_probability: float
    blisters_count_range: tuple[int, int]
    blisters_amount_range: tuple[float, float]
    dust_probability: float
    dust_amount_range: tuple[float, float]
    debug: bool

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> BackgroundAugmentationConfig:
        data = data or {}
        rotation_range = _parse_float_range(data.get("RotationAngleRange"), (-30.0, 30.0))

        oil = data.get("OilEffect", {})
        oil_probability = float(oil.get("Probability", 0.7))
        oil_probability = max(0.0, min(1.0, oil_probability))
        oil_strength_range = _parse_float_range(oil.get("StrengthRange"), (0.4, 0.9))
        noisy_mask_scale = _parse_int_range(oil.get("NoisyMaskScale"), (5, 100))

        blisters = data.get("BlistersEffect", {})
        blisters_probability = float(blisters.get("Probability", 0.7))
        blisters_probability = max(0.0, min(1.0, blisters_probability))
        blisters_count_range = _parse_int_range(blisters.get("CountRange"), (5, 25))
        blisters_amount_range = _parse_float_range(blisters.get("AmountRange"), (1.0, 5.0))

        dust = data.get("DustEffect", {})
        dust_probability = float(dust.get("Probability", 0.7))
        dust_probability = max(0.0, min(1.0, dust_probability))
        dust_amount_range = _parse_float_range(dust.get("AmountRange"), (0.1, 2.0))

        debug = bool(data.get("Debug", False))

        return cls(
            rotation_angle_range=rotation_range,
            oil_probability=oil_probability,
            oil_strength_range=oil_strength_range,
            noisy_mask_scale=noisy_mask_scale,
            blisters_probability=blisters_probability,
            blisters_count_range=blisters_count_range,
            blisters_amount_range=blisters_amount_range,
            dust_probability=dust_probability,
            dust_amount_range=dust_amount_range,
            debug=debug,
        )


@dataclass
class WorkAreaConfig:
    texture_path: str
    scale: tuple[float, float, float]
    conveyor_distance_range: tuple[float, float]
    augmentation: BackgroundAugmentationConfig

    @classmethod
    def from_settings(cls, data: dict[str, Any]) -> WorkAreaConfig:
        background_settings = data.get("BackgroundSettings", {})
        work_area = background_settings.get("WorkArea", data.get("WorkAreaSettings", {}))
        texture_path = str(
            work_area.get(
                "TextureDirectory",
                work_area.get("TexturePath", "background_textures"),
            )
        )
        scale = _ensure_vec3(work_area.get("Scale", (5.0, 5.0, 5.0)), (5.0, 5.0, 5.0))
        conveyor_distance_range = _parse_float_range(
            work_area.get("ConveyorBeltDistance", (1.2, 1.2)),
            (1.2, 1.2),
        )
        augmentation = BackgroundAugmentationConfig.from_dict(background_settings.get("Augmentation"))
        return cls(texture_path, scale, conveyor_distance_range, augmentation)

    def sample_conveyor_distance(self) -> float:
        return _sample_from_range(self.conveyor_distance_range)


@dataclass
class PhysicsConfig:
    active_margin: float
    active_mass: float
    active_friction: float
    passive_margin: float
    passive_mass: float
    passive_friction: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhysicsConfig:
        physics = data.get("PhysicsSettings", {})
        return cls(
            active_margin=float(physics.get("ActiveCollisionMargin", 0.0)),
            active_mass=float(physics.get("ActiveMass", 0.2)),
            active_friction=float(physics.get("ActiveFriction", 0.9)),
            passive_margin=float(physics.get("PassiveCollisionMargin", 0.0)),
            passive_mass=float(physics.get("PassiveMass", 0.4)),
            passive_friction=float(physics.get("PassiveFriction", 0.9)),
        )


@dataclass
class SettingsBundle:
    render: RenderConfig
    camera: CameraConfig
    dataset: DatasetConfig
    lighting: LightingConfig
    objects: ObjectsConfig
    work_area: WorkAreaConfig
    physics: PhysicsConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SettingsBundle:
        return cls(
            render=RenderConfig.from_dict(data.get("RenderSettings", {})),
            camera=CameraConfig.from_dict(data.get("CameraSettings", {})),
            dataset=DatasetConfig.from_dict(data.get("DatasetSettings", {})),
            lighting=LightingConfig.from_dict(data.get("LightingSettings", {})),
            objects=ObjectsConfig.from_dict(data.get("ObjectsSettings", {})),
            work_area=WorkAreaConfig.from_settings(data),
            physics=PhysicsConfig.from_dict(data),
        )


@dataclass
class OutputPaths:
    dataset_root: str
    temp_dir: str
    objects_root: str
    images_dir: str
    depth_dir: str
    masks_dir: str
    labels_dir: str

    @classmethod
    def from_settings(cls, base_dir: str, dataset_dir: str, objects_dir: str) -> OutputPaths:
        dataset_root = os.path.join(base_dir, dataset_dir)
        temp_dir = os.path.join(base_dir, "temp")
        return cls(
            dataset_root=dataset_root,
            temp_dir=temp_dir,
            objects_root=os.path.join(base_dir, objects_dir),
            images_dir=os.path.join(dataset_root, "images"),
            depth_dir=os.path.join(dataset_root, "depth"),
            masks_dir=os.path.join(dataset_root, "masks"),
            labels_dir=os.path.join(dataset_root, "labels"),
        )

    def mask_directory(self, image_stem: str) -> str:
        return os.path.join(self.masks_dir, image_stem)


@dataclass
class CategoryAssetPool:
    config: CategoryConfig
    assets: list[str]

    def take(self, allow_repeat: bool) -> Optional[str]:
        if not self.assets:
            return None
        choice = random.choice(self.assets)
        if not allow_repeat:
            self.assets.remove(choice)
        return choice

    def has_assets(self) -> bool:
        return bool(self.assets)


@dataclass
class MaskEntry:
    instance_index: int
    category_id: int
    slot_name: str


@dataclass
class CompositorSetup:
    nodes: Any
    links: Any
    render_node: Any
    png_output: Any
    depth_output: Any
    rgb_slot_name: str

    def add_mask_slot(self, pass_index: int, category_id: int) -> str:
        slot_name = f"{pass_index:04}-{category_id}-"
        location_y = -80 * (pass_index + 1)
        mask_node = create_id_musk_node(
            self.nodes,
            slot_name,
            pass_index,
            location_x=300,
            location_y=location_y,
        )
        add_output_slot(self.png_output, slot_name)
        link_nodes(self.links, self.render_node, "IndexOB", mask_node, "ID value")
        link_nodes(self.links, mask_node, "Alpha", self.png_output, slot_name)
        return slot_name


def reset_file_output_node(node: Any) -> None:
    while node.inputs:
        node.inputs.remove(node.inputs[0])
    while node.file_slots:
        node.file_slots.remove(node.file_slots[0])


def parse_image_id(argv: Sequence[str]) -> int:
    try:
        return int(argv[-1])
    except (IndexError, ValueError) as exc:
        raise SystemExit("Image index argument is required.") from exc


def convert_depth_exr_to_npz(source_path: str, target_path: str) -> None:
    if not os.path.exists(source_path):
        raise FileNotFoundError(source_path)

    depth_image = bpy.data.images.load(filepath=source_path, check_existing=False)
    try:
        width, height = depth_image.size
        channels = max(depth_image.channels, 1)
        pixels = np.array(depth_image.pixels[:], dtype=np.float32)

        if width == 0 or height == 0 or pixels.size == 0:
            raise RuntimeError(f"Depth EXR {source_path} loaded with empty data.")

        if pixels.size != width * height * channels:
            raise RuntimeError(
                f"Unexpected pixel buffer size for depth image: {source_path}"
            )

        depth_values = pixels[::channels]
        depth_map = depth_values.reshape(height, width)[::-1, :]
        np.savez_compressed(target_path, depth_map)
    finally:
        bpy.data.images.remove(depth_image)


def load_settings_bundle(settings_path: str = "settings.json") -> SettingsBundle:
    settings_data = load_json(settings_path)
    if settings_data is None:
        raise SystemExit("Failed to load settings.")
    return SettingsBundle.from_dict(settings_data)


def prepare_directories(paths: OutputPaths) -> None:
    ensure_directories_exist(
        paths.dataset_root,
        paths.temp_dir,
        paths.images_dir,
        paths.depth_dir,
        paths.masks_dir,
        paths.labels_dir,
    )


def prepare_work_area_texture(
    work_area: WorkAreaConfig,
    image_width: int,
    image_height: int,
    temp_dir: str,
) -> str:
    texture_source = Path(work_area.texture_path)
    if not texture_source.is_absolute():
        texture_source = Path(os.getcwd()) / texture_source

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if texture_source.is_dir():
        candidates = [
            path for path in texture_source.iterdir() if path.suffix.lower() in valid_extensions
        ]
    elif texture_source.is_file():
        candidates = [texture_source]
    else:
        raise FileNotFoundError(f"Background texture source '{texture_source}' not found.")

    if not candidates:
        raise FileNotFoundError(f"No background images found in '{texture_source}'.")

    selected_background = random.choice(candidates)
    aug_config = work_area.augmentation
    if aug_config.debug:
        print(f"Using background: {selected_background}")

    image = cv.imread(str(selected_background))
    if image is None:
        raise FileNotFoundError(f"Failed to read background image: {selected_background}")

    # we add this to compensate for the fact that the original image is too small
    factor = 2.4
    image_width = int(factor * float(image_width)) 
    image_height = int(factor * float(image_height))

    orig_height, orig_width = image.shape[:2]
    target_ratio = image_width / image_height
    current_ratio = orig_width / orig_height

    if current_ratio > target_ratio:
        new_width = max(1, int(round(orig_height * target_ratio)))
        x_offset = max(0, (orig_width - new_width) // 2)
        image = image[:, x_offset : x_offset + new_width]
    elif current_ratio < target_ratio:
        new_height = max(1, int(round(orig_width / target_ratio)))
        y_offset = max(0, (orig_height - new_height) // 2)
        image = image[y_offset : y_offset + new_height, :]

    angle = np.random.uniform(*aug_config.rotation_angle_range)
    if aug_config.debug:
        print(f"Rotation angle: {angle:.2f} degrees")
    image = rotate_and_center_crop(image, angle)
    image = cv.resize(image, (image_width, image_height), interpolation=cv.INTER_AREA)
    augmented = image.copy()

    if np.random.rand() < aug_config.oil_probability:
        oil_strength = np.random.uniform(*aug_config.oil_strength_range)
        if aug_config.debug:
            print(f"Applying oil effect (strength={oil_strength:.3f})")
        augmented = oil_effect(augmented, strength=oil_strength, noisy_mask_scale=list(aug_config.noisy_mask_scale))

    if np.random.rand() < aug_config.blisters_probability:
        low, high = aug_config.blisters_count_range
        if high <= low:
            blisters_count = low
        else:
            blisters_count = int(np.random.randint(low, high))
        blisters_amount = np.random.uniform(*aug_config.blisters_amount_range)
        if aug_config.debug:
            print(f"Applying blisters effect (count={blisters_count}, amount={blisters_amount:.3f})")
        augmented = blisters_effect(augmented, amount=blisters_amount, count=max(1, blisters_count))

    if np.random.rand() < aug_config.dust_probability:
        dust_amount = np.random.uniform(*aug_config.dust_amount_range)
        if aug_config.debug:
            print(f"Applying dust effect (amount={dust_amount:.3f})")
        augmented = dust_effect(augmented, amount=dust_amount)

    os.makedirs(temp_dir, exist_ok=True)
    output_path = Path(temp_dir) / f"background_{uuid.uuid4().hex}.png"
    success = cv.imwrite(str(output_path), augmented)
    if not success:
        raise RuntimeError(f"Failed to write augmented background to '{output_path}'.")

    return str(output_path)


def setup_scene(
    scene: bpy.types.Scene,
    settings: SettingsBundle,
    temp_dir: str,
) -> bpy.types.Object:
    delete_all()
    clear_cache()

    scene.frame_start = settings.render.start_frame
    scene.frame_end = settings.render.end_frame

    set_scene_settings(
        scene,
        settings.render.engine,
        settings.render.samples,
        settings.camera.image_width,
        settings.camera.image_height,
    )
    set_render_device(scene, settings.render.device)
    delete_all_nodes(scene)

    camera = create_camera(
        settings.camera.sensor_width,
        settings.camera.sensor_height,
        settings.camera.focal_length,
    )
    scene.camera = camera

    texture_path = prepare_work_area_texture(
        settings.work_area,
        settings.camera.image_width,
        settings.camera.image_height,
        temp_dir,
    )
    conveyor_distance = settings.work_area.sample_conveyor_distance()

    add_working_area(
        camera,
        settings.camera.image_width,
        settings.camera.image_height,
        conveyor_distance,
        texture_path,
        settings.work_area.scale,
    )

    bpy.context.view_layer.update()
    return camera


def setup_compositor(
    scene: bpy.types.Scene,
    temp_dir: str,
    image_id: int,
) -> CompositorSetup:
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    render_node = create_render_node(nodes)
    png_output = create_output_node(nodes, "PNGOutput")
    depth_output = create_depth_output_node(nodes, "DepthOutput")

    reset_file_output_node(png_output)
    reset_file_output_node(depth_output)

    png_output.base_path = temp_dir
    depth_output.base_path = os.path.join(temp_dir, f"depth-{image_id}-")

    rgb_slot_name = f"rgb_image-{image_id}-"

    add_output_slot(png_output, rgb_slot_name)
    add_output_slot(depth_output, "depth")

    link_nodes(links, render_node, "Image", png_output, rgb_slot_name)
    link_nodes(links, render_node, "Depth", depth_output, "depth")

    return CompositorSetup(nodes, links, render_node, png_output, depth_output, rgb_slot_name)


def setup_lighting(lighting: LightingConfig) -> None:
    max_sources = max(lighting.min_sources, lighting.max_sources)
    number_of_sources = random.randint(lighting.min_sources, max_sources)
    x_min, x_max, y_min, y_max, z_min, z_max = lighting.location_limits

    for index in range(number_of_sources):
        location = (
            random.uniform(x_min, x_max),
            random.uniform(y_min, y_max),
            random.uniform(z_min, z_max),
        )
        power = random.uniform(lighting.power_min, lighting.power_max)
        temperature = random.uniform(lighting.temp_min, lighting.temp_max)
        color_rgb = color_temperature_to_rgb(int(temperature))
        color = tuple(component / 255.0 for component in color_rgb)

        add_light(
            name=f"light_{index}",
            location=location,
            radius=lighting.radius,
            color=color,
            power=power,
        )


def build_asset_pools(
    objects_root: str,
    categories: Sequence[CategoryConfig],
) -> tuple[list[CategoryAssetPool], dict[str, list[str]]]:
    if not os.path.isdir(objects_root):
        raise FileNotFoundError(f"Objects directory '{objects_root}' not found.")

    base_dir = os.path.dirname(objects_root)
    category_dirs = {
        os.path.basename(path).lower(): path
        for path in glob.glob(os.path.join(objects_root, "*"))
        if os.path.isdir(path)
    }

    unclassified_assets: dict[str, list[str]] = {"transparent": [], "untransparent": []}
    unclassified_dir = category_dirs.get("unclassified")
    if unclassified_dir:
        transparent_dir = os.path.join(unclassified_dir, "transparent")
        if os.path.isdir(transparent_dir):
            unclassified_assets["transparent"] = sorted(glob.glob(os.path.join(transparent_dir, "*.fbx")))
        untransparent_dir = os.path.join(unclassified_dir, "untransparent")
        if os.path.isdir(untransparent_dir):
            unclassified_assets["untransparent"] = sorted(glob.glob(os.path.join(untransparent_dir, "*.fbx")))
        if not _has_unclassified_assets(unclassified_assets):
            unclassified_assets["untransparent"] = sorted(glob.glob(os.path.join(unclassified_dir, "*.fbx")))

    pools: list[CategoryAssetPool] = []
    for category in categories:
        category_dir = category_dirs.get(category.name.lower())
        if not category_dir:
            print(f"No directory found for category '{category.name}'.")
            continue

        assets: list[str] = []
        if category.name.lower() == "pet":
            pet_blend_dir = os.path.join(base_dir, "res_fbx_objects", "pet_blend")
            search_dirs = [pet_blend_dir, category_dir]
            for search_dir in search_dirs:
                if not os.path.isdir(search_dir):
                    continue
                assets = sorted(glob.glob(os.path.join(search_dir, "*.blend")))
                if assets:
                    break
        else:
            assets = sorted(glob.glob(os.path.join(category_dir, "*.fbx")))
        if not assets:
            print(f"No assets found for category '{category.name}'.")
            continue

        pools.append(CategoryAssetPool(category, assets))

    return pools, unclassified_assets


def choose_asset(
    pools: list[CategoryAssetPool],
    unclassified_assets: dict[str, list[str]],
    dataset_config: DatasetConfig,
    allow_repeat: bool,
) -> tuple[str, Optional[CategoryConfig], bool]:
    select_unclassified = random.random() < dataset_config.unclassified_probability

    if select_unclassified:
        unclassified_choice = _take_unclassified_asset(unclassified_assets, allow_repeat)
        if unclassified_choice is not None:
            asset_path, apply_transparency = unclassified_choice
            return asset_path, None, apply_transparency

    available_pools = [pool for pool in pools if pool.has_assets()]
    if not available_pools:
        unclassified_choice = _take_unclassified_asset(unclassified_assets, allow_repeat)
        if unclassified_choice is not None:
            asset_path, apply_transparency = unclassified_choice
            return asset_path, None, apply_transparency
        raise RuntimeError("No objects available to instantiate.")

    while available_pools:
        weights = [max(pool.config.probability, 0.0) for pool in available_pools]
        if all(weight == 0 for weight in weights):
            chosen_pool = random.choice(available_pools)
        else:
            chosen_pool = random.choices(available_pools, weights=weights, k=1)[0]

        asset_path = chosen_pool.take(allow_repeat)
        if asset_path:
            return asset_path, chosen_pool.config, False

        available_pools.remove(chosen_pool)

    unclassified_choice = _take_unclassified_asset(unclassified_assets, allow_repeat)
    if unclassified_choice is not None:
        asset_path, apply_transparency = unclassified_choice
        return asset_path, None, apply_transparency

    raise RuntimeError("Failed to choose an asset for instantiation.")


def append_blend_object(blend_path: str) -> Optional[bpy.types.Object]:
    object_name = Path(blend_path).stem
    existing_names = {obj.name for obj in bpy.data.objects}
    append_single_object(blend_path, object_name)
    new_objects = [obj for obj in bpy.data.objects if obj.name not in existing_names]
    if not new_objects:
        return bpy.data.objects.get(object_name)
    mesh_objects = [obj for obj in new_objects if obj.type == "MESH"]
    return mesh_objects[0] if mesh_objects else new_objects[0]


def apply_transparency(
    obj: bpy.types.Object,
    transparency: TransparencyConfig,
    color_override: Optional[tuple[float, float, float, float]] = None,
) -> None:
    if not transparency.enabled:
        return
    roughness, transmission, alpha, sampled_color = transparency.sample_parameters()
    color = color_override if color_override is not None else sampled_color
    make_object_transparent(
        obj,
        "BLEND",
        roughness,
        transmission,
        alpha,
        color,
    )
    obj["_transparency_applied"] = True


def populate_scene(
    settings: SettingsBundle,
    compositor: CompositorSetup,
    paths: OutputPaths,
) -> list[MaskEntry]:
    pools, unclassified_assets = build_asset_pools(paths.objects_root, settings.dataset.categories)
    object_count = random.randint(settings.objects.min_instances, settings.objects.max_instances)
    mask_entries: list[MaskEntry] = []
    pass_index = 1

    for _ in range(object_count):
        if not pools and not _has_unclassified_assets(unclassified_assets):
            print("No assets left to instantiate.")
            break

        try:
            asset_path, category, apply_unclassified_transparency = choose_asset(
                pools,
                unclassified_assets,
                settings.dataset,
                settings.objects.repeat,
            )
        except RuntimeError as err:
            print(err)
            break

        rotation = settings.objects.rotation_bounds.random_euler()
        position = (
            random.uniform(*settings.objects.spawn_bounds.x),
            random.uniform(*settings.objects.spawn_bounds.y),
            random.uniform(*settings.objects.spawn_bounds.z),
        )

        imported_objects: list[bpy.types.Object] = []
        if asset_path.lower().endswith(".blend"):
            obj = append_blend_object(asset_path)
            if obj is not None:
                obj.parent = None
                obj.constraints.clear()
                obj.location = position
                obj.rotation_euler = rotation
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                bpy.context.view_layer.update()
                imported_objects = [obj]
        else:
            imported_objects = add_object_from_fbx_file(asset_path, position, rotation)
        if not imported_objects:
            print(f"Failed to import object from {asset_path}.")
            continue

        obj = imported_objects[0]
        obj.parent = None
        obj.constraints.clear()

        set_object_active(obj)
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="MEDIAN")
        bpy.context.view_layer.update()

        if category is None:
            if apply_unclassified_transparency:
                apply_transparency(obj, settings.dataset.unclassified_transparency)
        else:
            obj.pass_index = pass_index
            if category.name.lower() != "pet":
                apply_transparency(obj, category.transparency)
            else:
                # PET assets already carry transparency in their authored materials; mark so we can strip it for depth/masks.
                obj["_transparency_applied"] = True
            slot_name = compositor.add_mask_slot(pass_index, category.id)
            mask_entries.append(MaskEntry(pass_index, category.id, slot_name))
            pass_index += 1

    if not mask_entries:
        print("Warning: no labeled instances were added to the scene.")

    bpy.context.view_layer.update()
    return mask_entries


def apply_rigid_body_physics(scene: bpy.types.Scene, physics: PhysicsConfig) -> None:
    for obj in scene.objects:
        if obj.type != "MESH":
            continue

        set_object_active(obj)
        bpy.ops.object.modifier_add(type="COLLISION")

        if obj.name == "working_area":
            add_rigid_body(
                obj,
                "PASSIVE",
                "MESH",
                physics.passive_margin,
                physics.passive_mass,
                physics.passive_friction,
            )
        else:
            add_rigid_body(
                obj,
                "ACTIVE",
                "CONVEX_HULL",
                physics.active_margin,
                physics.active_mass,
                physics.active_friction,
            )


def bake_rigid_body_simulation(scene: bpy.types.Scene, render_config: RenderConfig) -> None:
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()

    rb_world = scene.rigidbody_world
    rb_world.point_cache.frame_start = render_config.start_frame
    rb_world.point_cache.frame_end = render_config.end_frame

    bpy.ops.ptcache.free_bake_all()
    bpy.context.view_layer.update()
    time.sleep(0.1)
    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_set(render_config.end_frame)


def collect_rgb_image(temp_dir: str, slot_prefix: str, target_path: str) -> None:
    pattern = os.path.join(temp_dir, f"{slot_prefix}*.png")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No RGB image rendered for prefix '{slot_prefix}'.")

    latest = candidates[-1]
    move_file(latest, target_path)

    for redundant in candidates[:-1]:
        if os.path.exists(redundant):
            os.remove(redundant)


def clear_temp_png(temp_dir: str) -> None:
    for png_path in glob.glob(os.path.join(temp_dir, "*.png")):
        if os.path.exists(png_path):
            os.remove(png_path)


def remove_scene_transparency(scene: bpy.types.Scene) -> None:
    for obj in scene.objects:
        if not obj.get("_transparency_applied"):
            continue
        remove_transparency(obj)
        obj["_transparency_applied"] = False


def process_masks(
    temp_dir: str,
    mask_output_dir: str,
    image_stem: str,
) -> tuple[list[dict[str, Any]], int]:
    create_directory(mask_output_dir)

    instances: list[dict[str, Any]] = []
    next_instance_id = 1

    for file_name in sorted(os.listdir(temp_dir)):
        full_path = os.path.join(temp_dir, file_name)
        if not os.path.isfile(full_path):
            continue
        if not file_name.lower().endswith(".png"):
            continue
        if file_name.startswith("rgb_image-"):
            os.remove(full_path)
            continue

        name_without_ext = os.path.splitext(file_name)[0]
        parts = name_without_ext.split("-")
        if len(parts) < 3:
            os.remove(full_path)
            continue

        category_id_str = parts[1]
        try:
            category_id = int(category_id_str)
        except ValueError:
            os.remove(full_path)
            continue

        jpg_mask = cv.imread(full_path, cv.IMREAD_GRAYSCALE)
        if jpg_mask is None:
            os.remove(full_path)
            continue

        contours, _ = cv.findContours(jpg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            os.remove(full_path)
            continue

        cnt = contours[0]
        x, y, w, h = cv.boundingRect(cnt)
        area = int(cv.contourArea(cnt))

        binary_mask = convert_jpg_mask_to_binary(jpg_mask, 127)
        mask_name = f"{image_stem}_{next_instance_id:04}.npz"
        np.savez_compressed(os.path.join(mask_output_dir, mask_name), binary_mask)

        instances.append(
            {
                "instance_id": next_instance_id,
                "category_id": category_id,
                "iscrowd": 0,
                "bbox_xywh": [int(x), int(y), int(w), int(h)],
                "area": area,
            }
        )

        next_instance_id += 1
        os.remove(full_path)

    return instances, next_instance_id - 1


def main() -> None:
    add_current_directory_to_sys_path()

    image_id = parse_image_id(sys.argv)
    image_name = f"{image_id:06}.png"
    image_stem, _ = os.path.splitext(image_name)

    settings = load_settings_bundle()
    paths = OutputPaths.from_settings(os.getcwd(), settings.dataset.directory, settings.objects.directory)
    prepare_directories(paths)

    # old_blender_settings = set_color_management(
    #     display_device="sRGB",
    #     view_transform="Standard",
    #     look="None",
    #     exposure=0.0,
    #     gamma=1.0,
    #     use_curve_mapping=False,
    #     s_curve_strength=0.2,   # subtle contrast like many RGB cameras
    #     verbose=True
    # )

    scene = bpy.context.scene
    camera = setup_scene(scene, settings, paths.temp_dir)
    compositor = setup_compositor(scene, paths.temp_dir, image_id)
    setup_lighting(settings.lighting)

    mask_entries = populate_scene(settings, compositor, paths)
    if not mask_entries:
        raise RuntimeError("No labeled objects were added to the scene; aborting render.")

    apply_rigid_body_physics(scene, settings.physics)
    bake_rigid_body_simulation(scene, settings.render)

    scene.camera = camera
    bpy.context.view_layer.update()

    # First pass: render with transparency enabled for relevant objects
    bpy.ops.render.render(write_still=True)
    collect_rgb_image(
        paths.temp_dir,
        compositor.rgb_slot_name,
        os.path.join(paths.images_dir, image_name),
    )

    # Remove any mask artifacts produced by the first pass
    clear_temp_png(paths.temp_dir)

    # Second pass: remove transparency and render masks + depth
    remove_scene_transparency(scene)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)

    mask_output_dir = paths.mask_directory(image_stem)
    instances, mask_count = process_masks(paths.temp_dir, mask_output_dir, image_stem)
    if mask_count == 0:
        raise ValueError(f"No valid masks generated for image {image_name}.")

    labels = {
        "image": {
            "id": image_id,
            "file_name": image_name,
            "width": settings.camera.image_width,
            "height": settings.camera.image_height,
            "num_instances": mask_count,
        },
        "instances": instances,
    }

    depth_source_dir = os.path.join(paths.temp_dir, f"depth-{image_id}-")
    depth_source_exr = os.path.join(depth_source_dir, f"depth{settings.render.end_frame:04d}.exr")
    depth_target_npz = os.path.join(paths.depth_dir, f"{image_stem}.npz")

    convert_depth_exr_to_npz(depth_source_exr, depth_target_npz)

    if os.path.isdir(depth_source_dir):
        empty_directory(depth_source_dir)
        try:
            os.rmdir(depth_source_dir)
        except OSError:
            pass

    with open(os.path.join(paths.labels_dir, f"{image_stem}.json"), "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=4)

    empty_directory(paths.temp_dir)

    # restore_color_management(old_blender_settings)

if __name__ == "__main__":
    main()

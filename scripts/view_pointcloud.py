import os
import json
import numpy as np
import cv2
import open3d as o3d
from PIL import Image

# === Config ===
base_dir = "dataset"
image_index = 1              # change to the frame you want
max_depth_m = 3.5            # clip for viewing
min_depth_m = 0.1            # ignore very near noise
voxel_size = 0.003           # 3 mm downsample for display speed (set to None to skip)

settings_path = os.path.join(base_dir, "settings.json")
if not os.path.exists(settings_path):
    settings_path = "settings.json"

with open(settings_path, "r", encoding="utf-8") as settings_file:
    settings = json.load(settings_file)

camera_settings = settings.get("CameraSettings")
if camera_settings is None:
    raise KeyError("CameraSettings section missing from settings.json.")

try:
    sensor_width_mm = float(camera_settings["SensorWidth"])
    sensor_height_mm = float(camera_settings["SensorHeight"])
    focal_length_mm = float(camera_settings["FocalLength"])
    image_width_px = int(camera_settings["ImageWidth"])
    image_height_px = int(camera_settings["ImageHeight"])
except (KeyError, TypeError, ValueError) as exc:
    raise ValueError("CameraSettings must provide numeric SensorWidth, SensorHeight, FocalLength, ImageWidth, and ImageHeight.") from exc

if not all(v > 0 for v in (sensor_width_mm, sensor_height_mm, focal_length_mm, image_width_px, image_height_px)):
    raise ValueError("CameraSettings values must be positive.")

fx = focal_length_mm * image_width_px / sensor_width_mm
fy = focal_length_mm * image_height_px / sensor_height_mm
cx = image_width_px / 2.0
cy = image_height_px / 2.0

image_path = os.path.join(base_dir, "images", f"{image_index:06}.png")
depth_path = os.path.join(base_dir, "depth", f"{image_index:06}.npz")

if not os.path.exists(image_path):
    raise FileNotFoundError(image_path)
if not os.path.exists(depth_path):
    raise FileNotFoundError(depth_path)

# --- Load data ---
with Image.open(image_path) as pil_image:
    pil_rgb = pil_image.convert("RGB")
color_rgb = np.array(pil_rgb, dtype=np.uint8)
color = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
with np.load(depth_path) as depth_npz:
    depth_m = depth_npz[depth_npz.files[0]].astype(np.float32)
H, W = color_rgb.shape[:2]

if (W, H) != (image_width_px, image_height_px):
    raise ValueError(
        f"Image size {W}x{H} differs from settings {image_width_px}x{image_height_px}."
    )

if depth_m.shape != (H, W):
    raise ValueError(
        f"Depth map shape {depth_m.shape} does not match color image {(H, W)}."
    )

# --- Build pixel grid ---
# Valid depth mask
valid = np.isfinite(depth_m) & (depth_m > min_depth_m) & (depth_m < max_depth_m)

# Pixel coordinates
ys, xs = np.indices((H, W), dtype=np.float32)
zs = depth_m
xs3 = (xs - cx) * zs / fx
ys3 = (ys - cy) * zs / fy
# Camera coords (X right, Y down, Z forward in camera coords)
points = np.stack((xs3[valid], ys3[valid], zs[valid]), axis=1)

# Colors
colors = (color_rgb.reshape(-1, 3))[valid.ravel()].astype(np.float32) / 255.0

# --- Open3D point cloud ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Optional downsample for speed
if voxel_size and voxel_size > 0:
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

# # Save PLY
# ply_out = os.path.join(base_dir, f"pointcloud_{image_index:04}.ply")
# o3d.io.write_point_cloud(ply_out, pcd, write_ascii=False)
# print(f"Saved: {ply_out}")

# Visualize
o3d.visualization.draw_geometries([pcd], window_name=f"PointCloud {image_index:04}")

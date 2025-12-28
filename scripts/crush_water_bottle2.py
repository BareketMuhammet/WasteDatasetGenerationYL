# Blender 3.6.x
# Water Bottle Crushing (vacuum + flatten + fold)
# ------------------------------------------------
# - Clears scene
# - Imports water bottle FBX (aligned to Z)
# - Adds label material to the middle band of the bottle
# - Adds a strong cloth "vacuum" collapse
# - Adds a large crease band
# - Flattens thickness and shortens height with lattices
# - Bends bottle along a vertical curve (S-shaped fold)
# - Makes bottle transparent plastic while keeping the label visible
# - Saves the crushed bottle as a .blend file (to reuse materials / effects)

import bpy  # type: ignore
import math
import numpy as np
from mathutils import Vector, Matrix
import mathutils

# =========================================================
# CONFIG
# =========================================================

FBX_FILEPATH = r"/home/tem/Waste-Dataset-Generation/fbx_standard/pet/pet5.fbx"
BLEND_EXPORT_PATH = r"/home/tem/Waste-Dataset-Generation/res_fbx_objects/bottle_crushed.blend"

# Path to your label image (update to match your environment if needed)
LABEL_IMAGE_PATH = r"/home/tem/Waste-Dataset-Generation/fbx_objects0/w_lable4.png"

# Middle-band where the label should appear, in object-space Z
LABEL_BAND_HALF_HEIGHT = 0.20  # central 40% of the bottle (Z in [-0.2, 0.2])

# Middle band of the bottle (as fraction of height)
LABEL_Z_LOW_FRAC = 0.25   # 30% up from bottom
LABEL_Z_HIGH_FRAC = 0.7 # 65% up from bottom

# =========================================================
# UTILITIES
# =========================================================

def log(msg):
    print(f"[BOTTLE_CRUSH] {msg}")


def clear_scene():
    log("Clearing existing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for datablock_collection in (
        bpy.data.meshes,
        bpy.data.cameras,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.lattices,
    ):
        for datablock in list(datablock_collection):
            if datablock.users == 0:
                datablock_collection.remove(datablock)


# def create_bottle_from_scratch(
#         name="WaterBottle",
#         height=0.22,
#         radius_top=0.015,
#         radius_neck=0.012,
#         radius_shoulder=0.03,
#         radius_body=0.032,
#         radius_base=0.027,
#         wall_thickness=0.0015,
#         screw_steps=24,      # radial resolution (was 64)
#         profile_res=3,       # vertical resolution along profile curve
#         use_subsurf=True,
#         subsurf_levels=1     # was 2
#     ):
#     """
#     Creates a water bottle with much fewer vertices (better for cloth).
#     Key controls:
#       - screw_steps   : radial segments around the bottle
#       - profile_res   : segments along the profile (height)
#       - subsurf_levels: optional smoothing subdivision

#     With defaults above you'll get ~5–10x fewer verts than before.
#     """

#     # ---------------------------------------------------------
#     # 1. Create EMPTY CURVE DATA
#     # ---------------------------------------------------------
#     curve_data = bpy.data.curves.new(f"{name}_ProfileCurve", type='CURVE')
#     curve_data.dimensions = '2D'
#     curve_data.resolution_u = profile_res  # << controls vertical density

#     # Add one Bezier spline
#     spline = curve_data.splines.new('BEZIER')
#     spline.bezier_points.add(4)   # total 5 points (0..4)

#     def set_pt(i, x, z):
#         bp = spline.bezier_points[i]
#         bp.co = Vector((x, 0.0, z))
#         bp.handle_left_type = 'AUTO'
#         bp.handle_right_type = 'AUTO'

#     # 2. Define bottle outline (same shape as before)
#     set_pt(0, radius_base,      0.0)
#     set_pt(1, radius_body,      height * 0.30)
#     set_pt(2, radius_shoulder,  height * 0.65)
#     set_pt(3, radius_neck,      height * 0.90)
#     set_pt(4, radius_top,       height * 1.00)

#     curve_obj = bpy.data.objects.new(f"{name}_Profile", curve_data)
#     bpy.context.scene.collection.objects.link(curve_obj)

#     # ---------------------------------------------------------
#     # 3. Convert curve → mesh
#     # ---------------------------------------------------------
#     bpy.context.view_layer.objects.active = curve_obj
#     curve_obj.select_set(True)
#     bpy.ops.object.convert(target='MESH')
#     mesh_obj = curve_obj   # now a mesh

#     # ---------------------------------------------------------
#     # 4. Screw (revolve) with fewer radial steps
#     # ---------------------------------------------------------
#     screw = mesh_obj.modifiers.new("ScrewGen", "SCREW")
#     screw.axis = 'Z'
#     screw.angle = math.radians(360)
#     screw.steps = screw_steps          # << radial resolution
#     screw.render_steps = screw_steps
#     screw.use_smooth_shade = True
#     screw.screw_offset = 0.0

#     bpy.ops.object.modifier_apply(modifier="ScrewGen")

#     # ---------------------------------------------------------
#     # 5. Optional subsurf (light)
#     # ---------------------------------------------------------
#     if use_subsurf and subsurf_levels > 0:
#         sub = mesh_obj.modifiers.new("Subd", "SUBSURF")
#         sub.levels = subsurf_levels
#         sub.render_levels = subsurf_levels
#         bpy.ops.object.modifier_apply(modifier="Subd")

#     # ---------------------------------------------------------
#     # 6. Optional double wall
#     # ---------------------------------------------------------
#     if wall_thickness > 0.0:
#         solid = mesh_obj.modifiers.new("DoubleWall", "SOLIDIFY")
#         solid.thickness = wall_thickness
#         solid.offset = -1
#         bpy.ops.object.modifier_apply(modifier="DoubleWall")

#     # ---------------------------------------------------------
#     # 7. Smooth shading & origin
#     # ---------------------------------------------------------
#     bpy.ops.object.shade_smooth()
#     bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

#     # Put bottom roughly at Z=0 (nice for later)
#     mesh_obj.location.z += height * 0.5

#     mesh_obj.name = name
#     print("[BottleGen] Low-res bottle created from scratch.")
#     return mesh_obj

# def create_bottle_from_scratch(
#         name="WaterBottle",
#         height=0.22,
#         radius_top=0.015,
#         radius_neck=0.012,
#         radius_shoulder=0.03,
#         radius_body=0.032,
#         radius_base=0.027,
#         wall_thickness=0.0015,
#         screw_steps=24,      # radial resolution (was 64)
#         profile_res=3,       # vertical resolution along profile curve
#         use_subsurf=True,
#         subsurf_levels=1,    # was 2
#         shape_id=0           # 0..9 – choose bottle family
#     ):
#     """
#     Creates a water bottle with much fewer vertices (better for cloth).

#     New:
#       - shape_id in [0..9] selects among 10 preset PET profiles, roughly
#         matching common commercial bottles (tall/skinny, short/fat, waist,
#         small bottles, etc.).

#     Key controls (unchanged):
#       - screw_steps   : radial segments around the bottle
#       - profile_res   : segments along the profile (height)
#       - subsurf_levels: optional smoothing subdivision

#     The radius_* and height parameters are still respected as "base"
#     values and are scaled by the preset so you can fine-tune shapes.
#     """

#     # ---------------------------------------------------------
#     # 0. Shape presets (multiplies input radii & height)
#     #    Fractions are Z positions of control points along height.
#     # ---------------------------------------------------------
#     presets = [
#         # 0 – baseline straight/tall bottle (very close to original)
#         dict(h_scale=1.00, base=1.00, body=1.00, shoulder=1.00, neck=1.00, top=1.00,
#              z_body=0.30, z_shoulder=0.65, z_neck=0.90, z_top=1.00),
#         # 1 – tall & slightly slimmer neck, softer shoulder
#         dict(h_scale=1.10, base=0.95, body=0.98, shoulder=0.95, neck=0.80, top=0.85,
#              z_body=0.32, z_shoulder=0.70, z_neck=0.92, z_top=1.00),
#         # 2 – tall with stronger base & body (like leftmost ribbed bottles)
#         dict(h_scale=1.15, base=1.10, body=1.05, shoulder=1.00, neck=0.85, top=0.85,
#              z_body=0.30, z_shoulder=0.68, z_neck=0.90, z_top=1.00),
#         # 3 – shorter, chunky cylindrical bottle
#         dict(h_scale=0.85, base=1.20, body=1.25, shoulder=1.10, neck=0.90, top=0.90,
#              z_body=0.40, z_shoulder=0.78, z_neck=0.93, z_top=1.00),
#         # 4 – mid-height, strong shoulder, relatively narrow neck
#         dict(h_scale=0.95, base=1.10, body=1.10, shoulder=1.25, neck=0.80, top=0.80,
#              z_body=0.36, z_shoulder=0.72, z_neck=0.92, z_top=1.00),
#         # 5 – “label-band” style: tall body, short neck
#         dict(h_scale=1.00, base=1.05, body=1.15, shoulder=1.05, neck=0.75, top=0.80,
#              z_body=0.38, z_shoulder=0.80, z_neck=0.94, z_top=1.00),
#         # 6 – slim small bottle (right group – taller of the small ones)
#         dict(h_scale=0.75, base=0.95, body=0.90, shoulder=0.95, neck=0.80, top=0.80,
#              z_body=0.32, z_shoulder=0.68, z_neck=0.92, z_top=1.00),
#         # 7 – small “straight” bottle
#         dict(h_scale=0.65, base=1.05, body=1.00, shoulder=1.00, neck=0.85, top=0.85,
#              z_body=0.34, z_shoulder=0.72, z_neck=0.93, z_top=1.00),
#         # 8 – compact bottle with a slight waist (body slimmer than base/shoulder)
#         dict(h_scale=0.60, base=1.15, body=0.85, shoulder=1.10, neck=0.80, top=0.80,
#              z_body=0.30, z_shoulder=0.70, z_neck=0.92, z_top=1.00),
#         # 9 – smallest bottle, pronounced waist / round top
#         dict(h_scale=0.55, base=1.20, body=0.80, shoulder=1.15, neck=0.85, top=0.85,
#              z_body=0.32, z_shoulder=0.74, z_neck=0.94, z_top=1.00),
#     ]

#     # Clamp/normalize shape id
#     idx = int(shape_id) % len(presets)
#     p = presets[idx]

#     # Effective height for this preset
#     bottle_height = height * p["h_scale"]

#     # Effective radii
#     r_base = radius_base * p["base"]
#     r_body = radius_body * p["body"]
#     r_shoulder = radius_shoulder * p["shoulder"]
#     r_neck = radius_neck * p["neck"]
#     r_top = radius_top * p["top"]

#     # ---------------------------------------------------------
#     # 1. Create EMPTY CURVE DATA
#     # ---------------------------------------------------------
#     curve_data = bpy.data.curves.new(f"{name}_ProfileCurve", type='CURVE')
#     curve_data.dimensions = '2D'
#     curve_data.resolution_u = profile_res  # << controls vertical density

#     # Add one Bezier spline
#     spline = curve_data.splines.new('BEZIER')
#     spline.bezier_points.add(4)   # total 5 points (0..4)

#     def set_pt(i, x, z):
#         bp = spline.bezier_points[i]
#         bp.co = Vector((x, 0.0, z))
#         bp.handle_left_type = 'AUTO'
#         bp.handle_right_type = 'AUTO'

#     # ---------------------------------------------------------
#     # 2. Define bottle outline using preset profile
#     #    (same number of layers/control points as original)
#     # ---------------------------------------------------------
#     set_pt(0, r_base,           0.0)
#     set_pt(1, r_body,           bottle_height * p["z_body"])
#     set_pt(2, r_shoulder,       bottle_height * p["z_shoulder"])
#     set_pt(3, r_neck,           bottle_height * p["z_neck"])
#     set_pt(4, r_top,            bottle_height * p["z_top"])

#     curve_obj = bpy.data.objects.new(f"{name}_Profile", curve_data)
#     bpy.context.scene.collection.objects.link(curve_obj)

#     # ---------------------------------------------------------
#     # 3. Convert curve → mesh
#     # ---------------------------------------------------------
#     bpy.context.view_layer.objects.active = curve_obj
#     curve_obj.select_set(True)
#     bpy.ops.object.convert(target='MESH')
#     mesh_obj = curve_obj   # now a mesh

#     # ---------------------------------------------------------
#     # 4. Screw (revolve) with fewer radial steps
#     # ---------------------------------------------------------
#     screw = mesh_obj.modifiers.new("ScrewGen", "SCREW")
#     screw.axis = 'Z'
#     screw.angle = math.radians(360)
#     screw.steps = screw_steps          # << radial resolution
#     screw.render_steps = screw_steps
#     screw.use_smooth_shade = True
#     screw.screw_offset = 0.0

#     bpy.ops.object.modifier_apply(modifier="ScrewGen")

#     # ---------------------------------------------------------
#     # 5. Optional subsurf (light)
#     # ---------------------------------------------------------
#     if use_subsurf and subsurf_levels > 0:
#         sub = mesh_obj.modifiers.new("Subd", "SUBSURF")
#         sub.levels = subsurf_levels
#         sub.render_levels = subsurf_levels
#         bpy.ops.object.modifier_apply(modifier="Subd")

#     # ---------------------------------------------------------
#     # 6. Optional double wall
#     # ---------------------------------------------------------
#     if wall_thickness > 0.0:
#         solid = mesh_obj.modifiers.new("DoubleWall", "SOLIDIFY")
#         solid.thickness = wall_thickness
#         solid.offset = -1
#         bpy.ops.object.modifier_apply(modifier="DoubleWall")

#     # ---------------------------------------------------------
#     # 7. Smooth shading & origin
#     # ---------------------------------------------------------
#     bpy.ops.object.shade_smooth()
#     bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

#     # Put bottom roughly at Z=0 (nice for later)
#     mesh_obj.location.z += bottle_height * 0.5

#     mesh_obj.name = f"{name}_shape{idx}"
#     print(f"[BottleGen] Low-res bottle created from scratch (shape_id={idx}).")
#     return mesh_obj

# def create_bottle_from_scratch(
#         name="WaterBottle",
#         height=0.22,
#         radius_top=0.015,
#         radius_neck=0.012,
#         radius_shoulder=0.03,
#         radius_body=0.032,
#         radius_base=0.027,
#         wall_thickness=0.0015,
#         screw_steps=24,      # radial resolution
#         profile_res=3,       # vertical resolution along profile curve
#         use_subsurf=True,
#         subsurf_levels=1,    # optional smoothing
#         shape_id=0           # 0..9 – choose the bottle from the reference line-up
#     ):
#     """
#     Generate a PET bottle body (no cap) using a Bezier profile + Screw.

#     - The construction pipeline is the same as the original function.
#     - shape_id in [0..9] picks one of 10 preset silhouettes,
#       roughly matching the 10 bottles in the reference image (left→right).
#     - The existing radius_* and height parameters still work: each preset
#       is expressed as multipliers on these base values, so you can tweak.
#     """

#     # -------------------------------
#     # 1. Define 10 preset profiles
#     # -------------------------------
#     # Each preset:
#     #   height_scale: overall height multiplier
#     #   points: list of (z_frac, which_radius, scale)
#     #
#     # z_frac      : 0.0 bottom … 1.0 top (before height scaling)
#     # which_radius: "base", "body", "shoulder", "neck", "top"
#     # scale       : local radius multiplier
#     #
#     # All presets use the same number of profile control points so the
#     # mesh density stays comparable across shapes.
#     base_r = {
#         "base": radius_base,
#         "body": radius_body,
#         "shoulder": radius_shoulder,
#         "neck": radius_neck,
#         "top": radius_top,
#     }

#     # NOTE: ordered to roughly match the 10 bottles from left to right.
#     SHAPES = {
#         # 0 – tall, gently curved (baseline)
#         0: dict(
#             height_scale=1.00,
#             points=[
#                 (0.00, "base",      1.05),
#                 (0.06, "body",      0.95),
#                 (0.27, "body",      1.00),
#                 (0.52, "body",      1.02),
#                 (0.72, "shoulder",  0.98),
#                 (0.90, "neck",      1.00),
#                 (1.00, "top",       1.00),
#             ],
#         ),
#         # 1 – tall with stronger base & shoulder (suggests ribbed body)
#         1: dict(
#             height_scale=1.03,
#             points=[
#                 (0.00, "base",      1.15),
#                 (0.07, "body",      0.90),
#                 (0.22, "body",      1.05),
#                 (0.45, "body",      1.12),
#                 (0.70, "shoulder",  1.00),
#                 (0.90, "neck",      0.95),
#                 (1.00, "top",       0.95),
#             ],
#         ),
#         # 2 – very straight smooth tall bottle
#         2: dict(
#             height_scale=1.00,
#             points=[
#                 (0.00, "base",      1.10),
#                 (0.05, "body",      1.05),
#                 (0.25, "body",      1.03),
#                 (0.55, "body",      1.02),
#                 (0.78, "shoulder",  0.98),
#                 (0.90, "neck",      0.95),
#                 (1.00, "top",       0.95),
#             ],
#         ),
#         # 3 – shorter, chunky bottle with big shoulder
#         3: dict(
#             height_scale=0.90,
#             points=[
#                 (0.00, "base",      1.30),
#                 (0.06, "body",      1.20),
#                 (0.28, "body",      1.25),
#                 (0.55, "body",      1.20),
#                 (0.78, "shoulder",  1.10),
#                 (0.92, "neck",      0.90),
#                 (1.00, "top",       0.90),
#             ],
#         ),
#         # 4 – mid bottle with pronounced middle “label band”
#         4: dict(
#             height_scale=0.95,
#             points=[
#                 (0.00, "base",      1.20),
#                 (0.06, "body",      1.10),
#                 (0.30, "body",      1.30),  # main bulge / label band
#                 (0.50, "body",      1.25),
#                 (0.72, "shoulder",  1.05),
#                 (0.90, "neck",      0.85),
#                 (1.00, "top",       0.85),
#             ],
#         ),
#         # 5 – tall, slimmer bottle with long upper body
#         5: dict(
#             height_scale=1.10,
#             points=[
#                 (0.00, "base",      1.05),
#                 (0.05, "body",      0.95),
#                 (0.25, "body",      0.98),
#                 (0.60, "body",      0.96),
#                 (0.82, "shoulder",  0.92),
#                 (0.93, "neck",      0.85),
#                 (1.00, "top",       0.85),
#             ],
#         ),
#         # 6 – small, straight-ish bottle
#         6: dict(
#             height_scale=0.70,
#             points=[
#                 (0.00, "base",      1.10),
#                 (0.06, "body",      1.00),
#                 (0.30, "body",      1.00),
#                 (0.55, "body",      0.98),
#                 (0.78, "shoulder",  0.96),
#                 (0.92, "neck",      0.88),
#                 (1.00, "top",       0.88),
#             ],
#         ),
#         # 7 – small bottle with slightly rounded body
#         7: dict(
#             height_scale=0.65,
#             points=[
#                 (0.00, "base",      1.20),
#                 (0.06, "body",      1.10),
#                 (0.30, "body",      1.15),
#                 (0.55, "body",      1.12),
#                 (0.78, "shoulder",  1.00),
#                 (0.92, "neck",      0.88),
#                 (1.00, "top",       0.88),
#             ],
#         ),
#         # 8 – compact hourglass (strong waist)
#         8: dict(
#             height_scale=0.70,
#             points=[
#                 (0.00, "base",      1.25),
#                 (0.06, "body",      1.15),
#                 (0.30, "body",      0.80),  # waist in lower body
#                 (0.55, "body",      0.78),  # tightest waist
#                 (0.78, "shoulder",  1.05),
#                 (0.92, "neck",      0.90),
#                 (1.00, "top",       0.90),
#             ],
#         ),
#         # 9 – smallest bottle with very pronounced waist & round top
#         9: dict(
#             height_scale=0.60,
#             points=[
#                 (0.00, "base",      1.30),
#                 (0.06, "body",      1.20),
#                 (0.28, "body",      0.75),  # narrow waist
#                 (0.52, "body",      0.72),
#                 (0.78, "shoulder",  1.10),
#                 (0.92, "neck",      0.95),
#                 (1.00, "top",       0.95),
#             ],
#         ),
#     }

#     # Normalize & pick preset
#     sid = int(shape_id) % 10
#     shape = SHAPES[sid]
#     height_scale = shape["height_scale"]
#     total_height = height * height_scale

#     # Compute actual profile control points (radius, z)
#     profile_points = []
#     for z_frac, r_key, scale in shape["points"]:
#         r = base_r[r_key] * scale
#         z = total_height * z_frac
#         profile_points.append((r, z))

#     # -------------------------------
#     # 2. Create profile curve
#     # -------------------------------
#     curve_data = bpy.data.curves.new(f"{name}_ProfileCurve", type='CURVE')
#     curve_data.dimensions = '2D'
#     curve_data.resolution_u = profile_res

#     spline = curve_data.splines.new('BEZIER')
#     spline.bezier_points.add(len(profile_points) - 1)

#     for i, (r, z) in enumerate(profile_points):
#         bp = spline.bezier_points[i]
#         bp.co = Vector((r, 0.0, z))
#         bp.handle_left_type = 'AUTO'
#         bp.handle_right_type = 'AUTO'

#     curve_obj = bpy.data.objects.new(f"{name}_Profile_shape{sid}", curve_data)
#     bpy.context.scene.collection.objects.link(curve_obj)

#     # -------------------------------
#     # 3. Convert curve → mesh
#     # -------------------------------
#     bpy.context.view_layer.objects.active = curve_obj
#     curve_obj.select_set(True)
#     bpy.ops.object.convert(target='MESH')
#     mesh_obj = curve_obj  # now a mesh object

#     # -------------------------------
#     # 4. Screw (revolve) to full 360°
#     # -------------------------------
#     screw = mesh_obj.modifiers.new("ScrewGen", "SCREW")
#     screw.axis = 'Z'
#     screw.angle = math.radians(360.0)
#     screw.steps = screw_steps
#     screw.render_steps = screw_steps
#     screw.screw_offset = 0.0
#     screw.use_smooth_shade = True

#     bpy.ops.object.modifier_apply(modifier="ScrewGen")

#     # -------------------------------
#     # 5. Optional Subsurf smoothing
#     # -------------------------------
#     if use_subsurf and subsurf_levels > 0:
#         sub = mesh_obj.modifiers.new("Subd", "SUBSURF")
#         sub.levels = subsurf_levels
#         sub.render_levels = subsurf_levels
#         bpy.ops.object.modifier_apply(modifier="Subd")

#     # -------------------------------
#     # 6. Optional wall thickness
#     # -------------------------------
#     if wall_thickness > 0.0:
#         solid = mesh_obj.modifiers.new("DoubleWall", "SOLIDIFY")
#         solid.thickness = wall_thickness
#         solid.offset = -1.0
#         bpy.ops.object.modifier_apply(modifier="DoubleWall")

#     # -------------------------------
#     # 7. Final cleanup
#     # -------------------------------
#     bpy.ops.object.shade_smooth()
#     bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

#     # Put bottom close to Z=0 for convenience
#     # (object bounds min Z will be moved to ~0)
#     bbox = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
#     min_z = min(v.z for v in bbox)
#     mesh_obj.location.z -= min_z

#     mesh_obj.name = f"{name}_shape{sid}"
#     print(f"[BottleGen] Created bottle '{mesh_obj.name}' with shape_id={sid}")
#     return mesh_obj

def create_bottle_from_scratch(
        name="WaterBottle",
        height=0.22,
        radius_top=0.015,
        radius_neck=0.012,
        radius_shoulder=0.03,
        radius_body=0.032,
        radius_base=0.027,
        wall_thickness=0.0015,
        screw_steps=32,      # radial resolution
        profile_res=16,       # vertical curve resolution
        use_subsurf=True,
        subsurf_levels=1,
        shape_id=0           # 0..9 – bottle in the reference line-up
    ):
    """
    Generate a PET water bottle body (no cap) using a Bezier profile + Screw.

    - shape_id in [0..9] maps to the 10 silhouettes from the reference image,
      ordered left→right.
    - `height` controls a *base* height; each shape has a small height_scale so
      relative heights roughly match the photo:
          final_height = height * height_scale
    - `screw_steps` controls radial mesh resolution.
    - `profile_res` controls vertical resolution of the Bezier profile.
    """

    # -------------------------------------------------------------
    # Helper: apply ring-like ribs along Z (for corrugated bottles)
    # -------------------------------------------------------------
    def _apply_ribs(obj, ribs):
        """
        ribs: list of (z_frac, band_frac, scale)
          z_frac    : 0..1 position along bottle height
          band_frac : fraction of height occupied by the band
          scale     : radial multiplier at band center (e.g. 1.05)
        """
        if not ribs:
            return

        me = obj.data
        if not me.vertices:
            return

        zs = [v.co.z for v in me.vertices]
        z_min, z_max = min(zs), max(zs)
        h = z_max - z_min
        if h <= 1e-8:
            return

        for z_frac, band_frac, scale in ribs:
            center_z = z_min + z_frac * h
            half_band = 0.5 * band_frac * h
            if half_band <= 0.0:
                continue

            for v in me.vertices:
                dz = abs(v.co.z - center_z)
                if dz > half_band:
                    continue

                # Smooth falloff in band
                t = 1.0 - dz / half_band
                s = 1.0 + (scale - 1.0) * t

                # Scale radially around Z-axis
                x, y, z = v.co.x, v.co.y, v.co.z
                r2 = x * x + y * y
                if r2 < 1e-10:
                    continue
                v.co.x = x * s
                v.co.y = y * s

    # -------------------------------------------------------------
    # 1. Shape presets (approximate the 10 reference silhouettes)
    # -------------------------------------------------------------
    base_r = {
        "base": radius_base,
        "body": radius_body,
        "shoulder": radius_shoulder,
        "neck": radius_neck,
        "top": radius_top,
    }

    # Each shape:
    #   height_scale : overall height multiplier
    #   profile      : list of (z_frac, key, scale) → Bezier control points
    #   ribs         : optional corrugation rings (z_frac, band_frac, scale)
    SHAPES = {
        # 0 – tall ribbed bottle (far left)
        0: dict(
            height_scale=1.05,
            profile=[
                (0.00, "base",      1.15),
                (0.06, "body",      0.95),
                (0.30, "body",      1.05),
                (0.58, "body",      1.02),
                (0.78, "shoulder",  0.98),
                (0.92, "neck",      0.95),
                (1.00, "top",       0.95),
            ],
            ribs=[
                (0.12, 0.05, 1.06),
                (0.20, 0.05, 1.06),
                (0.28, 0.05, 1.06),
                (0.36, 0.05, 1.06),
                (0.44, 0.05, 1.06),
                (0.52, 0.05, 1.06),
            ],
        ),

        # 1 – tall, smoother, slight flare at shoulder (2nd bottle)
        1: dict(
            height_scale=1.05,
            profile=[
                (0.00, "base",      1.10),
                (0.05, "body",      1.02),
                (0.28, "body",      1.03),
                (0.60, "body",      1.04),
                (0.78, "shoulder",  1.00),
                (0.92, "neck",      0.95),
                (1.00, "top",       0.95),
            ],
            ribs=[],
        ),

        # 2 – tall almost perfectly cylindrical smooth body (3rd bottle)
        2: dict(
            height_scale=1.00,
            profile=[
                (0.00, "base",      1.05),
                (0.06, "body",      1.03),
                (0.30, "body",      1.02),
                (0.60, "body",      1.02),
                (0.80, "shoulder",  0.98),
                (0.92, "neck",      0.95),
                (1.00, "top",       0.95),
            ],
            ribs=[],
        ),

        # 3 – shorter, wide bottle with big rounded shoulder (4th bottle)
        3: dict(
            height_scale=0.90,
            profile=[
                (0.00, "base",      1.35),
                (0.06, "body",      1.25),
                (0.32, "body",      1.28),
                (0.58, "body",      1.22),
                (0.80, "shoulder",  1.12),
                (0.93, "neck",      0.90),
                (1.00, "top",       0.90),
            ],
            ribs=[],
        ),

        # 4 – mid-height bottle with two strong ribs/bands (5th bottle)
        4: dict(
            height_scale=0.95,
            profile=[
                (0.00, "base",      1.30),
                (0.06, "body",      1.20),
                (0.30, "body",      1.26),
                (0.54, "body",      1.26),
                (0.78, "shoulder",  1.05),
                (0.92, "neck",      0.88),
                (1.00, "top",       0.88),
            ],
            ribs=[
                (0.33, 0.06, 1.08),
                (0.50, 0.06, 1.08),
            ],
        ),

        # 5 – tall, slightly slimmer, long upper body (6th bottle)
        5: dict(
            height_scale=1.10,
            profile=[
                (0.00, "base",      1.05),
                (0.05, "body",      0.96),
                (0.25, "body",      0.98),
                (0.62, "body",      0.97),
                (0.82, "shoulder",  0.93),
                (0.93, "neck",      0.86),
                (1.00, "top",       0.86),
            ],
            ribs=[],
        ),

        # 6 – small almost straight bottle (7th bottle)
        6: dict(
            height_scale=0.75,
            profile=[
                (0.00, "base",      1.15),
                (0.05, "body",      1.02),
                (0.32, "body",      1.02),
                (0.60, "body",      1.00),
                (0.80, "shoulder",  0.96),
                (0.93, "neck",      0.88),
                (1.00, "top",       0.88),
            ],
            ribs=[],
        ),

        # 7 – small rounded bottle (8th bottle)
        7: dict(
            height_scale=0.70,
            profile=[
                (0.00, "base",      1.25),
                (0.06, "body",      1.15),
                (0.32, "body",      1.18),
                (0.60, "body",      1.15),
                (0.80, "shoulder",  1.00),
                (0.93, "neck",      0.88),
                (1.00, "top",       0.88),
            ],
            ribs=[
                (0.28, 0.07, 1.05),
            ],
        ),

        # 8 – compact hourglass bottle with strong waist (9th bottle)
        8: dict(
            height_scale=0.70,
            profile=[
                (0.00, "base",      1.30),
                (0.06, "body",      1.18),
                (0.30, "body",      0.82),
                (0.55, "body",      0.78),
                (0.78, "shoulder",  1.06),
                (0.93, "neck",      0.92),
                (1.00, "top",       0.92),
            ],
            ribs=[],
        ),

        # 9 – smallest bottle, pronounced waist & round top (10th bottle)
        9: dict(
            height_scale=0.60,
            profile=[
                (0.00, "base",      1.35),
                (0.06, "body",      1.20),
                (0.28, "body",      0.78),
                (0.52, "body",      0.72),
                (0.78, "shoulder",  1.10),
                (0.93, "neck",      0.96),
                (1.00, "top",       0.96),
            ],
            ribs=[
                (0.25, 0.10, 1.05),
            ],
        ),
    }

    sid = int(shape_id) % 10
    shape = SHAPES[sid]

    height_scale = shape["height_scale"]
    total_height = height * height_scale

    # Build list of (radius, z) from profile description
    profile_points = []
    for z_frac, r_key, scale in shape["profile"]:
        r = base_r[r_key] * scale
        z = total_height * z_frac
        profile_points.append((r, z))

    # -------------------------------------------------------------
    # 2. Create profile Bezier curve
    # -------------------------------------------------------------
    curve_data = bpy.data.curves.new(f"{name}_ProfileCurve", type='CURVE')
    curve_data.dimensions = '2D'
    curve_data.resolution_u = profile_res

    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(profile_points) - 1)

    for i, (r, z) in enumerate(profile_points):
        bp = spline.bezier_points[i]
        bp.co = Vector((r, 0.0, z))
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    curve_obj = bpy.data.objects.new(f"{name}_Profile_shape{sid}", curve_data)
    bpy.context.scene.collection.objects.link(curve_obj)

    # -------------------------------------------------------------
    # 3. Convert curve → mesh and revolve with Screw
    # -------------------------------------------------------------
    bpy.context.view_layer.objects.active = curve_obj
    curve_obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    mesh_obj = curve_obj  # now a mesh

    screw = mesh_obj.modifiers.new("ScrewGen", "SCREW")
    screw.axis = 'Z'
    screw.angle = math.radians(360.0)
    screw.steps = screw_steps
    screw.render_steps = screw_steps
    screw.screw_offset = 0.0
    screw.use_smooth_shade = True

    bpy.ops.object.modifier_apply(modifier="ScrewGen")

    # -------------------------------------------------------------
    # 4. Optional Subsurf for smoothing
    # -------------------------------------------------------------
    if use_subsurf and subsurf_levels > 0:
        sub = mesh_obj.modifiers.new("Subd", "SUBSURF")
        sub.levels = subsurf_levels
        sub.render_levels = subsurf_levels
        bpy.ops.object.modifier_apply(modifier="Subd")

    # -------------------------------------------------------------
    # 5. Optional wall thickness (single solid shell)
    # -------------------------------------------------------------
    if wall_thickness > 0.0:
        solid = mesh_obj.modifiers.new("DoubleWall", "SOLIDIFY")
        solid.thickness = wall_thickness
        solid.offset = -1.0
        bpy.ops.object.modifier_apply(modifier="DoubleWall")

    # -------------------------------------------------------------
    # 6. Apply ribs / corrugations where needed
    # -------------------------------------------------------------
    _apply_ribs(mesh_obj, shape.get("ribs", []))

    # -------------------------------------------------------------
    # 7. Cleanup – smooth shading & bottom on Z=0
    # -------------------------------------------------------------
    bpy.ops.object.shade_smooth()
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    # Move so that bottom sits at Z ≈ 0
    bbox = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
    min_z = min(v.z for v in bbox)
    mesh_obj.location.z -= min_z

    mesh_obj.name = f"{name}_shape{sid}"
    print(
        f"[BottleGen] Created bottle '{mesh_obj.name}' "
        f"with shape_id={sid}, height≈{total_height:.4f}"
    )
    return mesh_obj

def create_bottle_from_scratch(
        name="WaterBottle",
        height=0.22,
        radius_top=0.015,
        radius_neck=0.012,
        radius_shoulder=0.03,
        radius_body=0.032,
        radius_base=0.027,
        wall_thickness=0.0015,
        screw_steps=24,
        profile_res=3,
        use_subsurf=True,
        subsurf_levels=1,
        shape_id=0,
    ):
    """
    Generate one of 10 PET bottle shapes matching the reference row.

    shape_id : 0..9  (left-to-right in the row)

    Common features:
    - Smooth neck (no separate cap object)
    - Tapered shoulders
    - Cylindrical or waisted body (depending on shape)
    - Optional ribs or label bands (depending on shape)
    - PET petaloid base with 5 lobes (feet + valleys + central dome)
    - Closed bottom, open neck
    """

    import math
    import bpy
    import bmesh
    from mathutils import Vector

    # ---------------------------------------------------------
    # 0. Per-shape macro parameters (height & radii presets)
    # ---------------------------------------------------------
    presets = [
        # 0 – tall, mild ribs on upper body
        dict(h_scale=1.00, base=1.00, body=1.00, shoulder=1.00, neck=1.00, top=1.00,
             z_body=0.30, z_shoulder=0.65, z_neck=0.90, z_top=1.00),

        # 1 – tall, strongly ribbed lower body
        dict(h_scale=1.10, base=1.00, body=1.02, shoulder=1.00, neck=1.00, top=1.00,
             z_body=0.30, z_shoulder=0.68, z_neck=0.90, z_top=1.00),

        # 2 – tall, smooth body (no ribs)
        dict(h_scale=1.05, base=1.00, body=1.00, shoulder=0.98, neck=0.95, top=0.95,
             z_body=0.32, z_shoulder=0.70, z_neck=0.92, z_top=1.00),

        # 3 – shorter, chunky cylinder
        dict(h_scale=0.85, base=1.15, body=1.20, shoulder=1.05, neck=0.90, top=0.90,
             z_body=0.38, z_shoulder=0.78, z_neck=0.93, z_top=1.00),

        # 4 – tall with two label bands, wide body
        dict(h_scale=1.00, base=1.05, body=1.15, shoulder=1.00, neck=0.85, top=0.85,
             z_body=0.36, z_shoulder=0.76, z_neck=0.94, z_top=1.00),

        # 5 – mid-height bottle with big upper label band
        dict(h_scale=0.95, base=1.05, body=1.10, shoulder=1.05, neck=0.85, top=0.85,
             z_body=0.38, z_shoulder=0.78, z_neck=0.94, z_top=1.00),

        # 6 – small, tall bottle
        dict(h_scale=0.75, base=0.95, body=0.90, shoulder=0.95, neck=0.90, top=0.90,
             z_body=0.32, z_shoulder=0.68, z_neck=0.92, z_top=1.00),

        # 7 – small, straight bottle
        dict(h_scale=0.65, base=1.05, body=1.00, shoulder=1.00, neck=0.90, top=0.90,
             z_body=0.34, z_shoulder=0.72, z_neck=0.93, z_top=1.00),

        # 8 – compact waisted bottle
        dict(h_scale=0.60, base=1.15, body=0.85, shoulder=1.10, neck=0.90, top=0.90,
             z_body=0.30, z_shoulder=0.70, z_neck=0.92, z_top=1.00),

        # 9 – smallest bottle, strong waist & round top
        dict(h_scale=0.55, base=1.20, body=0.80, shoulder=1.15, neck=0.95, top=0.95,
             z_body=0.32, z_shoulder=0.74, z_neck=0.94, z_top=1.00),
    ]

    # Per-shape rib / label-band style
    rib_styles = {
        0: dict(mode="upper_ribs",  rib_count=4,  amp=0.06),
        1: dict(mode="full_ribs",   rib_count=10, amp=0.12),
        2: dict(mode="none",        rib_count=0,  amp=0.0),
        3: dict(mode="none",        rib_count=0,  amp=0.0),
        4: dict(mode="double_band", rib_count=2,  amp=0.09),
        5: dict(mode="single_band", rib_count=1,  amp=0.09),
        6: dict(mode="few_ribs",    rib_count=4,  amp=0.08),
        7: dict(mode="single_band", rib_count=1,  amp=0.08),
        8: dict(mode="waist_band",  rib_count=1,  amp=0.10),
        9: dict(mode="single_band", rib_count=1,  amp=0.07),
    }

    idx = int(shape_id) % len(presets)
    p = presets[idx]
    rs = rib_styles.get(idx, rib_styles[0])

    bottle_height = float(height) * p["h_scale"]
    r_base = radius_base * p["base"]
    r_body = radius_body * p["body"]
    r_sh   = radius_shoulder * p["shoulder"]
    r_neck = radius_neck * p["neck"]
    r_top  = radius_top * p["top"]

    # ---------------------------------------------------------
    # 1. Build 2D profile curve in XZ
    # ---------------------------------------------------------
    curve_data = bpy.data.curves.new(f"{name}_ProfileCurve", type='CURVE')
    curve_data.dimensions = '2D'
    curve_data.resolution_u = profile_res

    spline = curve_data.splines.new('BEZIER')

    H = bottle_height
    z_contact  = 0.00 * H
    z_groove   = 0.02 * H
    z_lower    = 0.07 * H
    z_body     = p["z_body"] * H
    z_shoulder = p["z_shoulder"] * H
    z_neck     = p["z_neck"] * H
    z_top      = p["z_top"] * H

    profile_pts = [
        (r_base * 1.05, z_contact),   # 0 – outer contact ring
        (r_base * 0.94, z_groove),    # 1 – inner groove
        (r_base * 1.02, z_lower),     # 2 – lower body start
        (r_body,         z_body),     # 3 – main body
        (r_sh,           z_shoulder), # 4 – shoulder
        (r_neck,         z_neck),     # 5 – neck
        (r_top,          z_top),      # 6 – neck top
    ]

    spline.bezier_points.add(len(profile_pts) - 1)
    for i, (r, z) in enumerate(profile_pts):
        bp = spline.bezier_points[i]
        bp.co = Vector((r, 0.0, z))
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    curve_obj = bpy.data.objects.new(f"{name}_Profile", curve_data)
    bpy.context.scene.collection.objects.link(curve_obj)

    # ---------------------------------------------------------
    # 2. Convert curve → mesh and revolve with Screw
    # ---------------------------------------------------------
    bpy.context.view_layer.objects.active = curve_obj
    curve_obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    mesh_obj = curve_obj  # now a mesh

    screw = mesh_obj.modifiers.new("ScrewGen", "SCREW")
    screw.axis = 'Z'
    screw.angle = math.radians(360.0)
    screw.steps = screw_steps
    screw.render_steps = screw_steps
    screw.screw_offset = 0.0
    screw.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier="ScrewGen")

    me = mesh_obj.data

    # ---------------------------------------------------------
    # 3. Close the bottom with BMesh (neck stays open)
    # ---------------------------------------------------------
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    z_min = min(v.co.z for v in bm.verts)
    eps = 1e-5
    bottom_edges = [
        e for e in bm.edges
        if e.is_boundary and all(abs(v.co.z - z_min) < eps for v in e.verts)
    ]
    if bottom_edges:
        bmesh.ops.holes_fill(bm, edges=bottom_edges)
        bm.normal_update()

    bm.to_mesh(me)
    bm.free()
    me.update()

    # ---------------------------------------------------------
    # 4. Optional subsurf + wall thickness
    # ---------------------------------------------------------
    if use_subsurf and subsurf_levels > 0:
        sub = mesh_obj.modifiers.new("Subd", "SUBSURF")
        sub.levels = subsurf_levels
        sub.render_levels = subsurf_levels
        bpy.ops.object.modifier_apply(modifier="Subd")

    if wall_thickness > 0.0:
        solid = mesh_obj.modifiers.new("DoubleWall", "SOLIDIFY")
        solid.thickness = wall_thickness
        solid.offset = -1.0
        bpy.ops.object.modifier_apply(modifier="DoubleWall")

    me = mesh_obj.data

    # ---------------------------------------------------------
    # 5. Ribs / label bands + PETALOID base
    # ---------------------------------------------------------
    zs = [v.co.z for v in me.vertices]
    z_min = min(zs)
    z_max = max(zs)
    H = z_max - z_min if z_max > z_min else 1e-6

    # --- PET base parameters (same for all shapes) ---
    petal_top      = z_min + 0.09 * H          # top of foot region
    petal_order    = 5                         # 5 lobes
    petal_rad_amp  = 0.22 * p["base"]         # radial modulation
    petal_drop_amp = 0.20 * H                # feet drop
    petal_lift_amp = 0.10 * H                # valleys lift
    dome_radius    = 0.45 * r_base * p["base"]
    dome_amp       = 0.18 * H                # central dome height

    # --- rib / band style ---
    style    = rs["mode"]
    rib_cnt  = rs["rib_count"]
    rib_amp  = rs["amp"]

    if style == "full_ribs":
        rib_z0, rib_z1 = 0.16, 0.78
    elif style == "upper_ribs":
        rib_z0, rib_z1 = 0.32, 0.72
    elif style == "few_ribs":
        rib_z0, rib_z1 = 0.22, 0.58
    elif style in {"single_band", "double_band", "waist_band"}:
        rib_z0, rib_z1 = 0.30, 0.70
    else:
        rib_z0 = rib_z1 = 0.0   # no ribs

    rib_bottom = z_min + rib_z0 * H
    rib_top    = z_min + rib_z1 * H
    rib_height = max(rib_top - rib_bottom, 1e-6)

    for v in me.vertices:
        x, y, z = v.co.x, v.co.y, v.co.z
        r2 = x * x + y * y
        if r2 < 1e-12:
            continue
        r = math.sqrt(r2)
        ang = math.atan2(y, x)

        # ---------------- PETALOID BASE ----------------
        petal_scale = 1.0
        if z <= petal_top:
            # how deep into the foot region we are (0 at top → 1 at bottom)
            tz = (petal_top - z) / (petal_top - z_min)
            tz = max(0.0, min(1.0, tz))

            c5 = math.cos(petal_order * ang)  # lobe / valley indicator

            # radial: lobes outwards, valleys slightly inwards
            petal_scale = 1.0 + petal_rad_amp * (tz ** 1.3) * c5

            # vertical: feet DOWN where c5 > 0, valleys UP where c5 < 0
            out = max(0.0, c5)
            inn = max(0.0, -c5)
            v.co.z -= petal_drop_amp * (tz ** 1.8) * out
            v.co.z += petal_lift_amp * (tz ** 1.8) * inn

            # central inner dome (inside ring of feet)
            if r < dome_radius:
                d = (dome_radius - r) / dome_radius  # 0 at edge → 1 at center
                v.co.z += dome_amp * (d ** 1.6) * tz

        # ---------------- RIBS / BANDS -----------------
        rib_scale = 1.0
        if rib_amp > 0.0 and rib_bottom <= z <= rib_top:
            t = (z - rib_bottom) / rib_height  # 0..1 in rib zone

            if style in {"full_ribs", "upper_ribs", "few_ribs"}:
                phase = 2.0 * math.pi * rib_cnt * t
                fade = 0.25 + 0.75 * (1.0 - abs(2.0 * t - 1.0))
                rib_scale = 1.0 + rib_amp * fade * math.sin(phase)

            elif style == "single_band":
                center = 0.5
                width = 0.18
                d = (t - center) / width
                band = math.exp(-d * d)
                rib_scale = 1.0 + rib_amp * band

            elif style == "double_band":
                centers = (0.33, 0.66)
                width = 0.10
                band = 0.0
                for c in centers:
                    d = (t - c) / width
                    band += math.exp(-d * d)
                rib_scale = 1.0 + rib_amp * band

            elif style == "waist_band":
                center = 0.5
                width = 0.14
                d = (t - center) / width
                band = math.exp(-d * d)
                # waist band is slightly inward instead of outward
                rib_scale = 1.0 - rib_amp * band

        # apply combined radial scaling
        s = petal_scale * rib_scale
        new_r = r * s
        scale = new_r / r
        v.co.x *= scale
        v.co.y *= scale

    me.update()

    # ---------------------------------------------------------
    # 6. Final placement – put base on Z=0
    # ---------------------------------------------------------
    bpy.ops.object.shade_smooth()
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    bbox = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
    min_z_world = min(v.z for v in bbox)
    mesh_obj.location.z -= min_z_world

    mesh_obj.name = f"{name}_shape{idx}"
    print(f"[BottleGen] Created PET bottle shape {idx}")
    return mesh_obj


# def create_bottle_from_scratch(
#         name="WaterBottle",
#         height=0.22,
#         radius_top=0.015,
#         radius_neck=0.012,
#         radius_shoulder=0.03,
#         radius_body=0.032,
#         radius_base=0.027,
#         wall_thickness=0.0015,
#         screw_steps=64,
#         profile_res=4,
#         use_subsurf=True,
#         subsurf_levels=1,
#         shape_id=0,      # kept for compatibility, but ignored (single shape)
#     ):
#     """
#     Generate a PET water bottle body (no cap) that matches the tall
#     ribbed reference bottle:

#     - Tall, slightly waisted body.
#     - Deep horizontal ribs on the lower half.
#     - 5-lobe PET base.
#     - Closed bottom, open neck.
#     """

#     import bmesh
#     from mathutils import Vector

#     # ---------------------------------------------------------
#     # 1. Define the silhouette profile (radius, z)
#     # ---------------------------------------------------------
#     total_height = float(height)

#     def P(z_frac, r):
#         """Helper for (radius, z) with z as fraction of height."""
#         return (r, total_height * z_frac)

#     # Base radii – treat input as "nominal" and tweak from there
#     r_base = radius_base * 1.18
#     r_body = radius_body * 1.02
#     r_mid  = radius_body * 0.96
#     r_sh   = radius_shoulder
#     r_neck = radius_neck
#     r_top  = radius_top

#     # Carefully tuned outline to resemble the photo:
#     #   0.00–0.08  : PET feet and lower base transition
#     #   0.08–0.48  : main cylindrical ribbed body
#     #   0.48–0.70  : upper smooth body with a subtle waist
#     #   0.70–1.00  : shoulder + neck
#     profile_points = [
#         P(0.00, r_base * 1.05),   # very bottom of feet
#         P(0.02, r_base * 0.95),   # groove just above contact ring
#         P(0.08, r_base * 1.02),   # start of straight ribbed section

#         P(0.48, r_body * 1.00),   # end of ribbed section

#         P(0.58, r_mid  * 0.98),   # slight inward waist
#         P(0.70, r_body * 1.00),   # start shoulder

#         P(0.84, r_sh   * 0.98),   # shoulder
#         P(0.94, r_neck * 0.95),   # neck
#         P(1.00, r_top  * 0.90),   # very top – no cap / threads modelled
#     ]

#     # ---------------------------------------------------------
#     # 2. Create Bezier profile curve
#     # ---------------------------------------------------------
#     curve_data = bpy.data.curves.new(f"{name}_ProfileCurve", type='CURVE')
#     curve_data.dimensions = '2D'
#     curve_data.resolution_u = profile_res

#     spline = curve_data.splines.new('BEZIER')
#     spline.bezier_points.add(len(profile_points) - 1)

#     for i, (r, z) in enumerate(profile_points):
#         bp = spline.bezier_points[i]
#         bp.co = Vector((r, 0.0, z))
#         bp.handle_left_type = 'AUTO'
#         bp.handle_right_type = 'AUTO'

#     curve_obj = bpy.data.objects.new(f"{name}_Profile", curve_data)
#     bpy.context.scene.collection.objects.link(curve_obj)

#     # ---------------------------------------------------------
#     # 3. Convert curve → mesh and revolve with Screw
#     # ---------------------------------------------------------
#     bpy.context.view_layer.objects.active = curve_obj
#     curve_obj.select_set(True)
#     bpy.ops.object.convert(target='MESH')
#     mesh_obj = curve_obj  # now a mesh

#     screw = mesh_obj.modifiers.new("ScrewGen", "SCREW")
#     screw.axis = 'Z'
#     screw.angle = math.radians(360.0)
#     screw.steps = screw_steps
#     screw.render_steps = screw_steps
#     screw.screw_offset = 0.0
#     screw.use_smooth_shade = True

#     bpy.ops.object.modifier_apply(modifier="ScrewGen")

#     # ---------------------------------------------------------
#     # 4. CLOSE THE BOTTOM (keep neck open)
#     # ---------------------------------------------------------
#     me = mesh_obj.data
#     bm = bmesh.new()
#     bm.from_mesh(me)
#     bm.verts.ensure_lookup_table()
#     bm.edges.ensure_lookup_table()

#     z_min = min(v.co.z for v in bm.verts)
#     eps = 1e-5

#     bottom_edges = [
#         e for e in bm.edges
#         if e.is_boundary
#         and all(abs(v.co.z - z_min) < eps for v in e.verts)
#     ]

#     if bottom_edges:
#         # Fill the bottom boundary loop
#         bmesh.ops.holes_fill(bm, edges=bottom_edges)
#         bm.normal_update()

#     bm.to_mesh(me)
#     bm.free()
#     me.update()

#     # ---------------------------------------------------------
#     # 5. Optional Subsurf for smoothing
#     # ---------------------------------------------------------
#     if use_subsurf and subsurf_levels > 0:
#         sub = mesh_obj.modifiers.new("Subd", "SUBSURF")
#         sub.levels = subsurf_levels
#         sub.render_levels = subsurf_levels
#         bpy.ops.object.modifier_apply(modifier="Subd")

#     # ---------------------------------------------------------
#     # 6. Optional wall thickness
#     # ---------------------------------------------------------
#     if wall_thickness > 0.0:
#         solid = mesh_obj.modifiers.new("DoubleWall", "SOLIDIFY")
#         solid.thickness = wall_thickness
#         solid.offset = -1.0
#         bpy.ops.object.modifier_apply(modifier="DoubleWall")

#     me = mesh_obj.data  # refresh reference

#     # ---------------------------------------------------------
#     # 7. Add ribs and PET base deformation
#     # ---------------------------------------------------------
#     zs = [v.co.z for v in me.vertices]
#     z_min = min(zs)
#     z_max = max(zs)
#     H = z_max - z_min if z_max > z_min else 1e-6

#     # Ribs – lower ~40% of bottle
#     rib_bottom = z_min + 0.12 * H
#     rib_top    = z_min + 0.52 * H
#     rib_height = rib_top - rib_bottom

#     rib_count = 8            # the visible number of rings
#     rib_amp   = 0.10         # relative radial amplitude

#     # PET 5-lobe base
#     petal_top = z_min + 0.09 * H
#     petal_amp = 0.13
#     petal_order = 5

#     for v in me.vertices:
#         x, y, z = v.co.x, v.co.y, v.co.z
#         r2 = x * x + y * y
#         if r2 < 1e-12:
#             continue
#         r = math.sqrt(r2)

#         # --- ribs along Z ---
#         rib_scale = 1.0
#         if rib_bottom <= z <= rib_top:
#             t = (z - rib_bottom) / rib_height  # 0..1
#             phase = 2.0 * math.pi * rib_count * t
#             # Make middle ribs slightly stronger than top/bottom
#             fade = 0.3 + 0.7 * (1.0 - abs(2.0 * t - 1.0))
#             rib_scale = 1.0 + rib_amp * fade * math.sin(phase)

#         # --- 5-lobe base ---
#         petal_scale = 1.0
#         if z <= petal_top:
#             ang = math.atan2(y, x)
#             # strongest at very bottom, fade out upwards
#             t_z = max(0.0, (petal_top - (z - z_min)) / (petal_top - z_min))
#             petal_scale = 1.0 + petal_amp * (t_z ** 1.6) * math.cos(petal_order * ang)

#         s = rib_scale * petal_scale
#         new_r = r * s
#         scale = new_r / r
#         v.co.x *= scale
#         v.co.y *= scale

#     me.update()

#     # ---------------------------------------------------------
#     # 8. Final cleanup – smooth shading & place bottom on Z = 0
#     # ---------------------------------------------------------
#     bpy.ops.object.shade_smooth()
#     bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

#     bbox_world = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
#     min_z_world = min(v.z for v in bbox_world)
#     mesh_obj.location.z -= min_z_world

#     mesh_obj.name = name
#     print(f"[BottleGen] Created tall ribbed PET bottle '{mesh_obj.name}', "
#           f"height≈{total_height:.4f}")
#     return mesh_obj

def create_bottle_from_scratch(
        name="WaterBottle",
        height=0.22,
        radius_top=0.015,
        radius_neck=0.012,
        radius_shoulder=0.03,
        radius_body=0.032,
        radius_base=0.027,
        wall_thickness=0.0015,
        screw_steps=64,      # radial resolution
        profile_res=80,       # vertical curve resolution
        use_subsurf=True,
        subsurf_levels=1,
        shape_id=0           # kept for compatibility, but ignored
    ):
    """
    Create a PET water bottle (no cap) with:

    - Narrow threaded neck
    - Smooth tapered shoulders
    - Gently curved cylindrical upper body
    - Slight inward waist
    - Ribbed lower body (multiple horizontal rings)
    - Petaloid base with rounded lobes
    - Closed bottom, open neck

    NOTE: `shape_id` is ignored – this function always creates the
    same reference-style bottle.
    """

    import bmesh
    from mathutils import Vector

    # ---------------------------------------------------------
    # 1. Define the silhouette (radius, z) in the XZ plane
    # ---------------------------------------------------------
    total_height = float(height)

    def P(z_frac, r):
        """Helper for (radius, z) based on height fraction."""
        return (r, total_height * z_frac)

    # Base radii: treat given radii as nominal and tweak slightly
    r_base = radius_base * 1.12
    r_body = radius_body * 1.02
    r_waist = radius_body * 0.96
    r_sh = radius_shoulder * 0.98
    r_neck = radius_neck * 0.90
    r_top = radius_top * 0.90

    # Outline tuned for a tall PET bottle like the reference
    profile_points = [
        # Petaloid contact ring & base groove
        P(0.00, r_base * 1.05),   # very bottom (outer feet)
        P(0.02, r_base * 0.94),   # small inward groove
        P(0.07, r_base * 1.02),   # start of lower cylindrical body

        # Main body: lower ribbed zone (0.07–0.50)
        P(0.50, r_body * 1.00),

        # Upper body with slight waist
        P(0.62, r_waist),         # soft inward waist
        P(0.74, r_body * 1.00),   # upper cylindrical body

        # Shoulder & neck
        P(0.86, r_sh),            # smooth shoulder
        P(0.94, r_neck),          # neck
        P(1.00, r_top),           # top end of plastic (no cap)
    ]

    # ---------------------------------------------------------
    # 2. Create profile curve
    # ---------------------------------------------------------
    curve_data = bpy.data.curves.new(f"{name}_ProfileCurve", type='CURVE')
    curve_data.dimensions = '2D'
    curve_data.resolution_u = profile_res

    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(profile_points) - 1)

    for i, (r, z) in enumerate(profile_points):
        bp = spline.bezier_points[i]
        bp.co = Vector((r, 0.0, z))
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    curve_obj = bpy.data.objects.new(f"{name}_Profile", curve_data)
    bpy.context.scene.collection.objects.link(curve_obj)

    # ---------------------------------------------------------
    # 3. Convert to mesh and revolve with Screw
    # ---------------------------------------------------------
    bpy.context.view_layer.objects.active = curve_obj
    curve_obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    mesh_obj = curve_obj  # now a mesh

    screw = mesh_obj.modifiers.new("ScrewGen", "SCREW")
    screw.axis = 'Z'
    screw.angle = math.radians(360.0)
    screw.steps = screw_steps
    screw.render_steps = screw_steps
    screw.screw_offset = 0.0
    screw.use_smooth_shade = True

    bpy.ops.object.modifier_apply(modifier="ScrewGen")

    # ---------------------------------------------------------
    # 4. CLOSE THE BOTTOM via BMesh (top stays open)
    # ---------------------------------------------------------
    me = mesh_obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    z_min = min(v.co.z for v in bm.verts)
    eps = 1e-5

    bottom_edges = [
        e for e in bm.edges
        if e.is_boundary and all(abs(v.co.z - z_min) < eps for v in e.verts)
    ]
    if bottom_edges:
        bmesh.ops.holes_fill(bm, edges=bottom_edges)
        bm.normal_update()

    bm.to_mesh(me)
    bm.free()
    me.update()

    # ---------------------------------------------------------
    # 5. Optional Subsurf (light smoothing)
    # ---------------------------------------------------------
    if use_subsurf and subsurf_levels > 0:
        sub = mesh_obj.modifiers.new("Subd", "SUBSURF")
        sub.levels = subsurf_levels
        sub.render_levels = subsurf_levels
        bpy.ops.object.modifier_apply(modifier="Subd")

    # ---------------------------------------------------------
    # 6. Optional wall thickness
    # ---------------------------------------------------------
    if wall_thickness > 0.0:
        solid = mesh_obj.modifiers.new("DoubleWall", "SOLIDIFY")
        solid.thickness = wall_thickness
        solid.offset = -1.0
        bpy.ops.object.modifier_apply(modifier="DoubleWall")

    me = mesh_obj.data  # refresh

    # ---------------------------------------------------------
    # 7. Add ribs, petaloid base and neck threads
    # ---------------------------------------------------------
    zs = [v.co.z for v in me.vertices]
    z_min = min(zs)
    z_max = max(zs)
    H = z_max - z_min if z_max > z_min else 1e-6

    # --- Ribbed lower body (horizontal rings) ---
    rib_bottom = z_min + 0.14 * H   # start ribs a bit above the base
    rib_top    = z_min + 0.52 * H   # end before upper body / waist
    rib_height = rib_top - rib_bottom
    rib_count  = 10                 # visible rings
    rib_amp    = 0.10               # relative radial amplitude

    # --- Petaloid base (5 rounded lobes) ---
    petal_top  = z_min + 0.08 * H   # top of lobe region
    petal_amp  = 0.18               # stronger deformation at bottom
    petal_order = 5                 # 5 lobes

    # --- Threaded neck (simple helical bump) ---
    thread_bottom = z_min + 0.90 * H
    thread_top    = z_min + 0.98 * H
    thread_height = max(thread_top - thread_bottom, 1e-6)
    thread_turns  = 2.0
    thread_amp    = 0.06

    for v in me.vertices:
        x, y, z = v.co.x, v.co.y, v.co.z
        r2 = x * x + y * y
        if r2 < 1e-12:
            continue
        r = math.sqrt(r2)
        ang = math.atan2(y, x)

        scale_rib = 1.0
        scale_petal = 1.0
        scale_thread = 1.0

        # --- ribs (axisymmetric rings) ---
        if rib_bottom <= z <= rib_top:
            t = (z - rib_bottom) / rib_height  # 0..1
            phase = 2.0 * math.pi * rib_count * t
            # fade slightly at start/end of rib zone
            fade = 0.25 + 0.75 * (1.0 - abs(2.0 * t - 1.0))
            scale_rib = 1.0 + rib_amp * fade * math.sin(phase)

        # --- petaloid base ---
        if z <= petal_top:
            tz = (petal_top - (z - z_min)) / (petal_top - z_min)
            tz = max(0.0, min(1.0, tz))
            scale_petal = 1.0 + petal_amp * (tz ** 1.6) * math.cos(petal_order * ang)

        # --- neck threads (simple helical ridge) ---
        if thread_bottom <= z <= thread_top:
            t_thread = (z - thread_bottom) / thread_height  # 0..1 along neck
            # Helical phase: angle plus vertical progress
            pitch = thread_height / thread_turns
            phi = ang + 2.0 * math.pi * (z - thread_bottom) / pitch
            bump = max(0.0, math.cos(phi))  # one-sided bump
            scale_thread = 1.0 + thread_amp * bump

        s = scale_rib * scale_petal * scale_thread
        new_r = r * s
        v.co.x *= new_r / r
        v.co.y *= new_r / r

    me.update()

    

    # ---------------------------------------------------------
    # 8. Final cleanup – smooth shading & place bottom at Z=0
    # ---------------------------------------------------------
    bpy.ops.object.shade_smooth()
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    bbox = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
    min_z_world = min(v.z for v in bbox)
    mesh_obj.location.z -= min_z_world

    mesh_obj.name = name
    print(f"[BottleGen] Created PET bottle '{mesh_obj.name}', "
          f"height≈{total_height:.4f} (shape_id ignored)")
    return mesh_obj

def create_bottle_from_scratch(
        name="WaterBottle",
        height=0.22,
        radius_top=0.015,
        radius_neck=0.012,
        radius_shoulder=0.03,
        radius_body=0.032,
        radius_base=0.027,
        wall_thickness=0.0015,
        screw_steps=64,
        profile_res=80,
        use_subsurf=True,
        subsurf_levels=1,
        shape_id=0,
    ):
    """
    Generate one of 10 PET bottle shapes matching the reference image:
    a row of PET bottles of different sizes and proportions.

    shape_id : 0..9  (left-to-right in the image)

    Common features:
    - Smooth neck (no separate cap object)
    - Tapered shoulders
    - Cylindrical or waisted body (depends on shape)
    - Optional ribs or label bands (depends on shape)
    - 5-lobed petaloid base
    - Closed bottom, open neck
    """

    import bmesh
    from mathutils import Vector

    # ---------------------------------------------------------
    # 0. Per-shape macro parameters (height & radii presets)
    # ---------------------------------------------------------
    presets = [
        # 0 – tall, mild ribs on upper body
        dict(h_scale=1.00, base=1.00, body=1.00, shoulder=1.00, neck=1.00, top=1.00,
             z_body=0.30, z_shoulder=0.65, z_neck=0.90, z_top=1.00),

        # 1 – tall, strongly ribbed lower body
        dict(h_scale=1.10, base=1.00, body=1.02, shoulder=1.00, neck=1.00, top=1.00,
             z_body=0.30, z_shoulder=0.68, z_neck=0.90, z_top=1.00),

        # 2 – tall, smooth body (no ribs)
        dict(h_scale=1.05, base=1.00, body=1.00, shoulder=0.98, neck=0.95, top=0.95,
             z_body=0.32, z_shoulder=0.70, z_neck=0.92, z_top=1.00),

        # 3 – shorter, chunky cylinder
        dict(h_scale=0.85, base=1.15, body=1.20, shoulder=1.05, neck=0.90, top=0.90,
             z_body=0.38, z_shoulder=0.78, z_neck=0.93, z_top=1.00),

        # 4 – tall with two label bands, wide body
        dict(h_scale=1.00, base=1.05, body=1.15, shoulder=1.00, neck=0.85, top=0.85,
             z_body=0.36, z_shoulder=0.76, z_neck=0.94, z_top=1.00),

        # 5 – mid-height bottle with big upper label band
        dict(h_scale=0.95, base=1.05, body=1.10, shoulder=1.05, neck=0.85, top=0.85,
             z_body=0.38, z_shoulder=0.78, z_neck=0.94, z_top=1.00),

        # 6 – small, tall bottle
        dict(h_scale=0.75, base=0.95, body=0.90, shoulder=0.95, neck=0.90, top=0.90,
             z_body=0.32, z_shoulder=0.68, z_neck=0.92, z_top=1.00),

        # 7 – small straight bottle
        dict(h_scale=0.65, base=1.05, body=1.00, shoulder=1.00, neck=0.90, top=0.90,
             z_body=0.34, z_shoulder=0.72, z_neck=0.93, z_top=1.00),

        # 8 – compact waisted bottle
        dict(h_scale=0.60, base=1.15, body=0.85, shoulder=1.10, neck=0.90, top=0.90,
             z_body=0.30, z_shoulder=0.70, z_neck=0.92, z_top=1.00),

        # 9 – smallest bottle, strong waist & round top
        dict(h_scale=0.55, base=1.20, body=0.80, shoulder=1.15, neck=0.95, top=0.95,
             z_body=0.32, z_shoulder=0.74, z_neck=0.94, z_top=1.00),
    ]

    # Per-shape rib / label-band style
    rib_styles = {
        0: dict(mode="upper_ribs",  rib_count=4,  amp=0.06),
        1: dict(mode="full_ribs",   rib_count=10, amp=0.12),
        2: dict(mode="none",        rib_count=0,  amp=0.0),
        3: dict(mode="none",        rib_count=0,  amp=0.0),
        4: dict(mode="double_band", rib_count=2,  amp=0.09),
        5: dict(mode="single_band", rib_count=1,  amp=0.09),
        6: dict(mode="few_ribs",    rib_count=4,  amp=0.08),
        7: dict(mode="single_band", rib_count=1,  amp=0.08),
        8: dict(mode="waist_band",  rib_count=1,  amp=0.10),
        9: dict(mode="single_band", rib_count=1,  amp=0.07),
    }

    idx = int(shape_id) % len(presets)
    p = presets[idx]
    rs = rib_styles.get(idx, rib_styles[0])

    bottle_height = float(height) * p["h_scale"]
    r_base = radius_base * p["base"]
    r_body = radius_body * p["body"]
    r_sh   = radius_shoulder * p["shoulder"]
    r_neck = radius_neck * p["neck"]
    r_top  = radius_top * p["top"]

    # ---------------------------------------------------------
    # 1. Build 2D profile curve in XZ
    # ---------------------------------------------------------
    curve_data = bpy.data.curves.new(f"{name}_ProfileCurve", type='CURVE')
    curve_data.dimensions = '2D'
    curve_data.resolution_u = profile_res

    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(5)  # 6 points: 0..5

    def set_pt(i, r, z):
        bp = spline.bezier_points[i]
        bp.co = Vector((r, 0.0, z))
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    H = bottle_height
    z_contact  = 0.00 * H
    z_groove   = 0.02 * H
    z_lower    = 0.07 * H
    z_body     = p["z_body"] * H
    z_shoulder = p["z_shoulder"] * H
    z_neck     = p["z_neck"] * H
    z_top      = p["z_top"] * H

    # Contact ring + groove, then body / shoulder / neck / top
    set_pt(0, r_base * 1.05, z_contact)
    set_pt(1, r_base * 0.94, z_groove)
    set_pt(2, r_base * 1.02, z_lower)
    set_pt(3, r_body,        z_body)
    set_pt(4, r_sh,          z_shoulder)
    set_pt(5, r_neck,        z_neck)

    # Add one extra point for the top of the neck
    spline.bezier_points.add(1)  # now 7 points total, index 0..6
    set_pt(6, r_top, z_top)

    curve_obj = bpy.data.objects.new(f"{name}_Profile", curve_data)
    bpy.context.scene.collection.objects.link(curve_obj)

    # ---------------------------------------------------------
    # 2. Convert curve → mesh and revolve with Screw
    # ---------------------------------------------------------
    bpy.context.view_layer.objects.active = curve_obj
    curve_obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    mesh_obj = curve_obj  # now a mesh

    screw = mesh_obj.modifiers.new("ScrewGen", "SCREW")
    screw.axis = 'Z'
    screw.angle = math.radians(360.0)
    screw.steps = screw_steps
    screw.render_steps = screw_steps
    screw.screw_offset = 0.0
    screw.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier="ScrewGen")

    me = mesh_obj.data

    # ---------------------------------------------------------
    # 3. Close the bottom with BMesh (neck stays open)
    # ---------------------------------------------------------
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    z_min = min(v.co.z for v in bm.verts)
    eps = 1e-5
    bottom_edges = [
        e for e in bm.edges
        if e.is_boundary and all(abs(v.co.z - z_min) < eps for v in e.verts)
    ]
    if bottom_edges:
        bmesh.ops.holes_fill(bm, edges=bottom_edges)
        bm.normal_update()

    bm.to_mesh(me)
    bm.free()
    me.update()

    # ---------------------------------------------------------
    # 4. Optional subsurf + wall thickness
    # ---------------------------------------------------------
    if use_subsurf and subsurf_levels > 0:
        sub = mesh_obj.modifiers.new("Subd", "SUBSURF")
        sub.levels = subsurf_levels
        sub.render_levels = subsurf_levels
        bpy.ops.object.modifier_apply(modifier="Subd")

    if wall_thickness > 0.0:
        solid = mesh_obj.modifiers.new("DoubleWall", "SOLIDIFY")
        solid.thickness = wall_thickness
        solid.offset = -1.0
        bpy.ops.object.modifier_apply(modifier="DoubleWall")

    me = mesh_obj.data

    # ---------------------------------------------------------
    # 5. Ribs / label bands + 5-lobe PET base
    # ---------------------------------------------------------
    zs = [v.co.z for v in me.vertices]
    z_min = min(zs)
    z_max = max(zs)
    H = z_max - z_min if z_max > z_min else 1e-6

    # Petaloid base (same for all shapes)
    petal_top   = z_min + 0.08 * H
    petal_amp   = 0.16 * p["base"]
    petal_order = 5

    style    = rs["mode"]
    rib_cnt  = rs["rib_count"]
    rib_amp  = rs["amp"]

    if style == "full_ribs":
        rib_z0, rib_z1 = 0.16, 0.78
    elif style == "upper_ribs":
        rib_z0, rib_z1 = 0.32, 0.72
    elif style == "few_ribs":
        rib_z0, rib_z1 = 0.22, 0.58
    elif style in {"single_band", "double_band", "waist_band"}:
        rib_z0, rib_z1 = 0.30, 0.70
    else:
        rib_z0 = rib_z1 = 0.0   # no ribs

    rib_bottom = z_min + rib_z0 * H
    rib_top    = z_min + rib_z1 * H
    rib_height = max(rib_top - rib_bottom, 1e-6)

    for v in me.vertices:
        x, y, z = v.co.x, v.co.y, v.co.z
        r2 = x * x + y * y
        if r2 < 1e-12:
            continue
        r = math.sqrt(r2)
        ang = math.atan2(y, x)

        # --- PET base lobes ---
        petal_scale = 1.0
        if z <= petal_top:
            tz = (petal_top - z) / (petal_top - z_min)
            tz = max(0.0, min(1.0, tz))
            petal_scale = 1.0 + petal_amp * (tz ** 1.7) * math.cos(petal_order * ang)

        # --- ribs / bands ---
        rib_scale = 1.0
        if rib_amp > 0.0 and rib_bottom <= z <= rib_top:
            t = (z - rib_bottom) / rib_height  # 0..1 in rib zone

            if style in {"full_ribs", "upper_ribs", "few_ribs"}:
                phase = 2.0 * math.pi * rib_cnt * t
                fade = 0.25 + 0.75 * (1.0 - abs(2.0 * t - 1.0))
                rib_scale = 1.0 + rib_amp * fade * math.sin(phase)

            elif style == "single_band":
                center = 0.5
                width = 0.18
                d = (t - center) / width
                band = math.exp(-d * d)
                rib_scale = 1.0 + rib_amp * band

            elif style == "double_band":
                centers = (0.33, 0.66)
                width = 0.10
                band = 0.0
                for c in centers:
                    d = (t - c) / width
                    band += math.exp(-d * d)
                rib_scale = 1.0 + rib_amp * band

            elif style == "waist_band":
                center = 0.5
                width = 0.14
                d = (t - center) / width
                band = math.exp(-d * d)
                # waist band is slightly inward instead of outward
                rib_scale = 1.0 - rib_amp * band

        s = petal_scale * rib_scale
        new_r = r * s
        scale = new_r / r
        v.co.x *= scale
        v.co.y *= scale

    me.update()

    # ---------------------------------------------------------
    # 6. Final placement – put base on Z=0
    # ---------------------------------------------------------
    bpy.ops.object.shade_smooth()
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    bbox = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
    min_z_world = min(v.z for v in bbox)
    mesh_obj.location.z -= min_z_world

    mesh_obj.name = f"{name}_shape{idx}"
    print(f"[BottleGen] Created PET bottle shape {idx}")
    return mesh_obj


def create_bottle_from_scratch(
        name="WaterBottle",
        height=0.22,
        radius_top=0.015,
        radius_neck=0.012,
        radius_shoulder=0.03,
        radius_body=0.032,
        radius_base=0.027,
        wall_thickness=0.0015,
        screw_steps=16,
        profile_res=8,
        use_subsurf=True,
        subsurf_levels=1,
        shape_id=0,
    ):
    """
    Generate one of 10 PET bottle shapes matching the reference row.

    shape_id : 0..9  (left-to-right in the row)

    Common features:
    - Smooth neck (no separate cap object)
    - Tapered shoulders
    - Cylindrical or waisted body (depending on shape)
    - Optional ribs or label bands (depending on shape)
    - PET petaloid base with 5 lobes (feet + valleys + central dome)
    - Closed bottom, open neck
    """

    import math
    import bpy
    import bmesh
    from mathutils import Vector

    # ---------------------------------------------------------
    # 0. Per-shape macro parameters (height & radii presets)
    # ---------------------------------------------------------
    presets = [
        # 0 – tall, mild ribs on upper body
        dict(h_scale=1.00, base=1.00, body=1.00, shoulder=1.00, neck=1.00, top=1.00,
             z_body=0.30, z_shoulder=0.65, z_neck=0.90, z_top=1.00),

        # 1 – tall, strongly ribbed lower body
        dict(h_scale=1.10, base=1.00, body=1.02, shoulder=1.00, neck=1.00, top=1.00,
             z_body=0.30, z_shoulder=0.68, z_neck=0.90, z_top=1.00),

        # 2 – tall, smooth body (no ribs)
        dict(h_scale=1.05, base=1.00, body=1.00, shoulder=0.98, neck=0.95, top=0.95,
             z_body=0.32, z_shoulder=0.70, z_neck=0.92, z_top=1.00),

        # 3 – shorter, chunky cylinder
        dict(h_scale=0.85, base=1.15, body=1.20, shoulder=1.05, neck=0.90, top=0.90,
             z_body=0.38, z_shoulder=0.78, z_neck=0.93, z_top=1.00),

        # 4 – tall with two label bands, wide body
        dict(h_scale=1.00, base=1.05, body=1.15, shoulder=1.00, neck=0.85, top=0.85,
             z_body=0.36, z_shoulder=0.76, z_neck=0.94, z_top=1.00),

        # 5 – mid-height bottle with big upper label band
        dict(h_scale=0.95, base=1.05, body=1.10, shoulder=1.05, neck=0.85, top=0.85,
             z_body=0.38, z_shoulder=0.78, z_neck=0.94, z_top=1.00),

        # 6 – small, tall bottle
        dict(h_scale=0.75, base=0.95, body=0.90, shoulder=0.95, neck=0.90, top=0.90,
             z_body=0.32, z_shoulder=0.68, z_neck=0.92, z_top=1.00),

        # 7 – small, straight bottle
        dict(h_scale=0.65, base=1.05, body=1.00, shoulder=1.00, neck=0.90, top=0.90,
             z_body=0.34, z_shoulder=0.72, z_neck=0.93, z_top=1.00),

        # 8 – compact waisted bottle
        dict(h_scale=0.60, base=1.20, body=0.85, shoulder=1.10, neck=0.90, top=0.90,
             z_body=0.30, z_shoulder=0.70, z_neck=0.92, z_top=1.00),

        # 9 – smallest bottle, strong waist & round top
        dict(h_scale=0.55, base=1.20, body=0.80, shoulder=1.15, neck=0.95, top=0.95,
             z_body=0.32, z_shoulder=0.74, z_neck=0.94, z_top=1.00),
    ]

    # Per-shape rib / label-band style
    rib_styles = {
        0: dict(mode="upper_ribs",  rib_count=4,  amp=0.06),
        1: dict(mode="full_ribs",   rib_count=10, amp=0.12),
        2: dict(mode="none",        rib_count=0,  amp=0.0),
        3: dict(mode="none",        rib_count=0,  amp=0.0),
        4: dict(mode="double_band", rib_count=2,  amp=0.09),
        5: dict(mode="single_band", rib_count=1,  amp=0.09),
        6: dict(mode="few_ribs",    rib_count=4,  amp=0.08),
        7: dict(mode="single_band", rib_count=1,  amp=0.08),
        8: dict(mode="waist_band",  rib_count=1,  amp=0.10),
        9: dict(mode="single_band", rib_count=1,  amp=0.07),
    }

    idx = int(shape_id) % len(presets)
    p = presets[idx]
    rs = rib_styles.get(idx, rib_styles[0])

    bottle_height = float(height) * p["h_scale"]
    r_base = radius_base * p["base"]
    r_body = radius_body * p["body"]
    r_sh   = radius_shoulder * p["shoulder"]
    r_neck = radius_neck * p["neck"]
    r_top  = radius_top * p["top"]

    # ---------------------------------------------------------
    # 1. Build 2D profile curve in XZ
    # ---------------------------------------------------------
    curve_data = bpy.data.curves.new(f"{name}_ProfileCurve", type='CURVE')
    curve_data.dimensions = '2D'
    curve_data.resolution_u = profile_res

    spline = curve_data.splines.new('BEZIER')

    H = bottle_height
    z_contact  = 0.00 * H
    z_groove   = 0.02 * H
    z_lower    = 0.07 * H
    z_body     = p["z_body"] * H
    z_shoulder = p["z_shoulder"] * H
    z_neck     = p["z_neck"] * H
    z_top      = p["z_top"] * H

    profile_pts = [
        (r_base * 1.05, z_contact),   # 0 – outer contact ring
        (r_base * 0.94, z_groove),    # 1 – inner groove
        (r_base * 1.02, z_lower),     # 2 – lower body start
        (r_body,         z_body),     # 3 – main body
        (r_sh,           z_shoulder), # 4 – shoulder
        (r_neck,         z_neck),     # 5 – neck
        (r_top,          z_top),      # 6 – neck top
    ]

    spline.bezier_points.add(len(profile_pts) - 1)
    for i, (r, z) in enumerate(profile_pts):
        bp = spline.bezier_points[i]
        bp.co = Vector((r, 0.0, z))
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    curve_obj = bpy.data.objects.new(f"{name}_Profile", curve_data)
    bpy.context.scene.collection.objects.link(curve_obj)

    # ---------------------------------------------------------
    # 2. Convert curve → mesh and revolve with Screw
    # ---------------------------------------------------------
    bpy.context.view_layer.objects.active = curve_obj
    curve_obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    mesh_obj = curve_obj  # now a mesh

    screw = mesh_obj.modifiers.new("ScrewGen", "SCREW")
    screw.axis = 'Z'
    screw.angle = math.radians(360.0)
    screw.steps = screw_steps
    screw.render_steps = screw_steps
    screw.screw_offset = 0.0
    screw.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier="ScrewGen")

    me = mesh_obj.data

    # ---------------------------------------------------------
    # 3. Close the bottom with BMesh (neck stays open)
    # ---------------------------------------------------------
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    z_min = min(v.co.z for v in bm.verts)
    eps = 1e-5
    bottom_edges = [
        e for e in bm.edges
        if e.is_boundary and all(abs(v.co.z - z_min) < eps for v in e.verts)
    ]
    if bottom_edges:
        bmesh.ops.holes_fill(bm, edges=bottom_edges)
        bm.normal_update()

    bm.to_mesh(me)
    bm.free()
    me.update()

    # ---------------------------------------------------------
    # 4. Optional subsurf + wall thickness
    # ---------------------------------------------------------
    if use_subsurf and subsurf_levels > 0:
        sub = mesh_obj.modifiers.new("Subd", "SUBSURF")
        sub.levels = subsurf_levels
        sub.render_levels = subsurf_levels
        bpy.ops.object.modifier_apply(modifier="Subd")

    if wall_thickness > 0.0:
        solid = mesh_obj.modifiers.new("DoubleWall", "SOLIDIFY")
        solid.thickness = wall_thickness
        solid.offset = -1.0
        bpy.ops.object.modifier_apply(modifier="DoubleWall")

    me = mesh_obj.data

    # ---------------------------------------------------------
    # 5. Ribs / label bands + PETALOID base
    # ---------------------------------------------------------
    zs = [v.co.z for v in me.vertices]
    z_min = min(zs)
    z_max = max(zs)
    H = z_max - z_min if z_max > z_min else 1e-6

    # --- PET base parameters (same for all shapes) ---
    petal_top      = z_min + 0.09 * H          # top of foot region
    petal_order    = 5                         # 5 lobes
    petal_rad_amp  = 0.22 * p["base"]         # radial modulation
    petal_drop_amp = 0.20 * H                # feet drop
    petal_lift_amp = 0.010 * H                # valleys lift
    dome_radius    = 0.45 * r_base * p["base"]
    dome_amp       = 0.018 * H                # central dome height

    # --- rib / band style ---
    style    = rs["mode"]
    rib_cnt  = rs["rib_count"]
    rib_amp  = rs["amp"]

    if style == "full_ribs":
        rib_z0, rib_z1 = 0.16, 0.78
    elif style == "upper_ribs":
        rib_z0, rib_z1 = 0.32, 0.72
    elif style == "few_ribs":
        rib_z0, rib_z1 = 0.22, 0.58
    elif style in {"single_band", "double_band", "waist_band"}:
        rib_z0, rib_z1 = 0.30, 0.70
    else:
        rib_z0 = rib_z1 = 0.0   # no ribs

    rib_bottom = z_min + rib_z0 * H
    rib_top    = z_min + rib_z1 * H
    rib_height = max(rib_top - rib_bottom, 1e-6)

    for v in me.vertices:
        x, y, z = v.co.x, v.co.y, v.co.z
        r2 = x * x + y * y
        if r2 < 1e-12:
            continue
        r = math.sqrt(r2)
        ang = math.atan2(y, x)

        # # ---------------- PETALOID BASE ----------------
        # petal_scale = 1.0
        # if z <= petal_top:
        #     # how deep into the foot region we are (0 at top → 1 at bottom)
        #     tz = (petal_top - z) / (petal_top - z_min)
        #     tz = max(0.0, min(1.0, tz))

        #     c5 = math.cos(petal_order * ang)  # lobe / valley indicator
        #     # phase = math.radians(80)   # adjust 5°–25° depending on how inward you want  
        #     # c5 = math.cos(petal_order * (ang + phase))

        #     # radial: lobes outwards, valleys slightly inwards
        #     petal_scale = 1.0 + petal_rad_amp * (tz ** 1.3) * c5

        #     # vertical: feet DOWN where c5 > 0, valleys UP where c5 < 0
        #     out = max(0.0, c5)
        #     inn = max(0.0, -c5)
        #     v.co.z -= petal_drop_amp * (tz ** 1.8) * out
        #     v.co.z += petal_lift_amp * (tz ** 1.8) * inn

        #     # central inner dome (inside ring of feet)
        #     if r < dome_radius:
        #         d = (dome_radius - r) / dome_radius  # 0 at edge → 1 at center
        #         v.co.z += dome_amp * (d ** 1.6) * tz
                # ---------------- PETALOID BASE ----------------
        petal_scale = 1.0
        if z <= petal_top:
            # how deep into the foot region we are (0 at top → 1 at bottom)
            tz = (petal_top - z) / (petal_top - z_min)
            tz = max(0.0, min(1.0, tz))

            # 5-lobe pattern around the bottle
            c5 = math.cos(petal_order * ang)

            # --- radial profile for the FOOT TIP position ---
            #   we want the lowest point NOT at the outer wall,
            #   but at some inner radius (foot_center_r).
            outer_r      = r_base * 1.05 * p["base"]
            foot_center_r = outer_r * 0.3        # 70% of outer radius → peak moves inward
            foot_width    = outer_r * 0.35       # how wide the foot is radially

            # gaussian-style radial factor: max at foot_center_r
            dr = (r - foot_center_r) / max(foot_width, 1e-6)
            radial_factor = math.exp(-dr * dr)   # 0..1, max at foot_center_r

            # radial: lobes outwards, valleys slightly inwards
            petal_scale = 1.0 + petal_rad_amp * (tz ** 1.3) * c5 * radial_factor

            # vertical: feet DOWN where c5 > 0, valleys UP where c5 < 0
            out = max(0.0, c5)
            inn = max(0.0, -c5)

            v.co.z -= petal_drop_amp * (tz ** 1.8) * out * radial_factor
            v.co.z += petal_lift_amp * (tz ** 1.8) * inn * radial_factor

            # central inner dome (inside ring of feet)
            if r < dome_radius:
                d = (dome_radius - r) / dome_radius  # 0 at edge → 1 at center
                v.co.z += dome_amp * (d ** 1.6) * tz


        # ---------------- RIBS / BANDS -----------------
        rib_scale = 1.0
        if rib_amp > 0.0 and rib_bottom <= z <= rib_top:
            t = (z - rib_bottom) / rib_height  # 0..1 in rib zone

            if style in {"full_ribs", "upper_ribs", "few_ribs"}:
                phase = 2.0 * math.pi * rib_cnt * t
                fade = 0.25 + 0.75 * (1.0 - abs(2.0 * t - 1.0))
                rib_scale = 1.0 + rib_amp * fade * math.sin(phase)

            elif style == "single_band":
                center = 0.5
                width = 0.18
                d = (t - center) / width
                band = math.exp(-d * d)
                rib_scale = 1.0 + rib_amp * band

            elif style == "double_band":
                centers = (0.33, 0.66)
                width = 0.10
                band = 0.0
                for c in centers:
                    d = (t - c) / width
                    band += math.exp(-d * d)
                rib_scale = 1.0 + rib_amp * band

            elif style == "waist_band":
                center = 0.5
                width = 0.14
                d = (t - center) / width
                band = math.exp(-d * d)
                # waist band is slightly inward instead of outward
                rib_scale = 1.0 - rib_amp * band

        # apply combined radial scaling
        s = petal_scale * rib_scale
        new_r = r * s
        scale = new_r / r
        v.co.x *= scale
        v.co.y *= scale

    me.update()

    # ---------------------------------------------------------
    # 6. Final placement – put base on Z=0
    # ---------------------------------------------------------
    bpy.ops.object.shade_smooth()
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    bbox = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
    min_z_world = min(v.z for v in bbox)
    mesh_obj.location.z -= min_z_world

    mesh_obj.name = f"{name}_shape{idx}"
    print(f"[BottleGen] Created PET bottle shape {idx}")
    return mesh_obj






def create_label_material():
    """
    Create a material that shows the bottle label texture.
    Uses Generated coordinates so it will always map around the bottle.
    """

    scale = 1
    tex_scale=(scale, scale)      # (X, Y) scale of the label texture
    #tex_scale=(1, 1)
    tex_rotation_deg=-90.0       # rotation in degrees (around Z, i.e. 2D image rotation)


    mat = bpy.data.materials.new(name="BottleLabel")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Nodes
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (200, 0)

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (0, 0)
    tex.image = bpy.data.images.load(LABEL_IMAGE_PATH, check_existing=True)

    # tex.extension = 'EXTEND'
    # tex.interpolation = 'Closest'

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-400, 0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-200, 0)

    # # Scale: X = around the bottle, Y = up the bottle
    mapping.inputs["Scale"].default_value[0] = tex_scale[0]
    mapping.inputs["Scale"].default_value[1] = tex_scale[1]
    mapping.inputs["Scale"].default_value[2] = 2.7

    # mapping.inputs['Scale'].default_value = scale

    # Rotation: rotate in the XY plane, like rotating the image itself
    rot_rad = math.radians(tex_rotation_deg)
    # mapping.inputs["Rotation"].default_value[2] = rot_rad  # Z axis
    mapping.inputs["Rotation"].default_value[0] = rot_rad
    # mapping.inputs["Rotation"].default_value[1] = rot_rad

    mapping.inputs["Location"].default_value[1] = 0.3

    # Use Generated coordinates so we don't depend on UVs
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex.inputs["Vector"])

    # Plug texture into base color
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    # # Slightly paper/plastic-like label
    # bsdf.inputs["Roughness"].default_value = 0.4
    # bsdf.inputs["Specular"].default_value = 0.2
    # bsdf.inputs["Alpha"].default_value = 1.0   # opaque label

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return mat

# def create_label_material():
#     """
#     Create a material that shows the bottle label texture.
#     - Uses Generated coordinates.
#     - Automatically orients the image (rotate 90 if needed).
#     - Scales so the label band (by Z fractions) is filled vertically.
#     - Disables repetition (no tiling).
#     """

#     # --- Load image ---
#     img = bpy.data.images.load(LABEL_IMAGE_PATH, check_existing=True)

#     # Detect portrait vs landscape to auto-rotate if needed
#     w, h = img.size
#     portrait = h > w  # taller than wide → usually needs 90° rotation

#     mat = bpy.data.materials.new(name="BottleLabel")
#     mat.use_nodes = True

#     nodes = mat.node_tree.nodes
#     links = mat.node_tree.links
#     nodes.clear()

#     # Nodes
#     out = nodes.new("ShaderNodeOutputMaterial")
#     out.location = (400, 0)

#     bsdf = nodes.new("ShaderNodeBsdfPrincipled")
#     bsdf.location = (200, 0)

#     tex = nodes.new("ShaderNodeTexImage")
#     tex.location = (0, 0)
#     tex.image = img

#     # IMPORTANT: don't repeat the texture outside 0–1
#     tex.extension = 'CLIP'   # or 'EXTEND' if you prefer edge stretching

#     tex_coord = nodes.new("ShaderNodeTexCoord")
#     tex_coord.location = (-600, 0)

#     mapping = nodes.new("ShaderNodeMapping")
#     mapping.location = (-300, 0)

#     # --- AUTO SCALE TO FILL LABEL BAND VERTICALLY ---

#     # Generated Z goes from 0 (bottom of bottle) to 1 (top of bottle)
#     # Your label band is between LABEL_Z_LOW_FRAC and LABEL_Z_HIGH_FRAC.
#     band_height = LABEL_Z_HIGH_FRAC - LABEL_Z_LOW_FRAC

#     # We want that band range [z_low, z_high] in Generated space to map
#     # to [0, 1] in texture V (vertical). Approximate with:
#     # V = Z * scale_y + offset_y
#     # solve:
#     #   Z = LABEL_Z_LOW_FRAC   -> V = 0
#     #   Z = LABEL_Z_HIGH_FRAC  -> V = 1

#     scale_y = 1.0 / band_height
#     offset_y = -LABEL_Z_LOW_FRAC * scale_y
#     offset_y = 0

#     # Mapping node uses Location (translation) and Scale
#     # Here we assume:
#     #   out = in * Scale + Location
#     mapping.inputs["Scale"].default_value[1] = scale_y      # vertical scale
#     mapping.inputs["Location"].default_value[1] = offset_y  # vertical offset

#     # You can leave X scale at 1 so it wraps around once
#     mapping.inputs["Scale"].default_value[0] = 1.0
#     mapping.inputs["Scale"].default_value[2] = 1.0

#     # --- AUTO ROTATION ---

#     # In your working version you rotated around X; let's keep that.
#     rot_deg = -90.0 if portrait else 0.0
#     rot_deg = -90.0 
#     rot_rad = math.radians(rot_deg)

#     mapping.inputs["Rotation"].default_value[0] = rot_rad   # X rotation
#     # mapping.inputs["Rotation"].default_value[2] = rot_rad  # use Z if that works better in your setup

#     # --- Wire up nodes ---

#     # Use Generated coordinates so we don't depend on UVs
#     links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
#     links.new(mapping.outputs["Vector"], tex.inputs["Vector"])

#     # Plug texture into base color
#     links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

#     # Optional look tweaks:
#     # bsdf.inputs["Roughness"].default_value = 0.4
#     # bsdf.inputs["Specular"].default_value = 0.2

#     links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

#     return mat



def assign_label_to_middle_band(obj, label_mat,
                                z_low_frac=LABEL_Z_LOW_FRAC,
                                z_high_frac=LABEL_Z_HIGH_FRAC):
    """
    Simple version for single-mesh bottles:
    - Ensures slot 0 is a base plastic material.
    - Adds `label_mat` as slot 1.
    - Assigns label (slot 1) only to polygons whose center Z lies
      between z_low_frac and z_high_frac of the object's height.
    """

    me = obj.data

    # --- Ensure there is a base material in slot 0 ---
    if len(me.materials) == 0:
        base_mat = bpy.data.materials.new(name="BottleBase")
        base_mat.use_nodes = True
        nodes = base_mat.node_tree.nodes
        links = base_mat.node_tree.links
        nodes.clear()

        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (300, 0)

        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (100, 0)
        bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Roughness"].default_value = 0.25

        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        me.materials.append(base_mat)  # slot 0

    # --- Ensure label_mat is in the slots (usually slot 1) ---
    existing_index = None
    for i, m in enumerate(me.materials):
        if m is label_mat or (m and m.name == label_mat.name):
            existing_index = i
            break

    if existing_index is None:
        me.materials.append(label_mat)
        label_index = len(me.materials) - 1
    else:
        label_index = existing_index

    # --- Compute Z band in world space ---
    mw = obj.matrix_world
    zs = [(mw @ poly.center).z for poly in me.polygons]
    if not zs:
        print("[BOTTLE_CRUSH] assign_label_to_middle_band: no polygons.")
        return

    z_min, z_max = min(zs), max(zs)
    height = z_max - z_min
    if height <= 1e-6:
        print("[BOTTLE_CRUSH] assign_label_to_middle_band: height is zero.")
        return

    z_low = z_min + z_low_frac * height
    z_high = z_min + z_high_frac * height

    labeled = 0
    for poly in me.polygons:
        cz = (mw @ poly.center).z
        if z_low <= cz <= z_high:
            poly.material_index = label_index   # label slot
            labeled += 1
        else:
            # make sure non-band faces use base plastic (slot 0)
            if poly.material_index >= len(me.materials):
                poly.material_index = 0

    print(f"[BOTTLE_CRUSH] assign_label_to_middle_band: labeled {labeled} faces.")




def import_water_bottle(filepath):
    """
    Import FBX and try to pick the water-bottle mesh.
    Returns the chosen bottle object.
    """
    log(f"Importing water bottle FBX from: {filepath}")
    NAME_HINTS = ["bottle", "water", "plastic"]

    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=filepath)
    after = set(bpy.context.scene.objects)

    new_objects = list(after - before)
    mesh_candidates = [o for o in new_objects if o.type == 'MESH']

    # Prefer something with a sensible name
    for hint in NAME_HINTS:
        for o in mesh_candidates:
            if hint.lower() in o.name.lower():
                log(f"Using bottle object by hint '{hint}': {o.name}")
                return o

    if not mesh_candidates:
        raise RuntimeError("No mesh objects found in imported FBX.")

    # Fall back to the largest mesh (by world-space bbox volume) to avoid
    # grabbing a cap/ring when object names don't contain hints.
    bpy.context.view_layer.update()

    def bbox_volume(o):
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        xs = [p.x for p in bb]
        ys = [p.y for p in bb]
        zs = [p.z for p in bb]
        return (max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs))

    bottle_obj = max(mesh_candidates, key=bbox_volume)
    log(f"Using bottle object (largest bbox fallback): {bottle_obj.name}")
    return bottle_obj


def make_object_transparent(obj, blend_mode, roughness, transmission, alpha, color_code):

    transparent_material = bpy.data.materials.new(name="TransparentMaterial")
    transparent_material.use_nodes = True
    nodes = transparent_material.node_tree.nodes
    links = transparent_material.node_tree.links
    nodes.clear()

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    links.new(principled_bsdf_node.outputs[0], output_node.inputs[0])

    principled_bsdf_node.inputs['Base Color'].default_value = color_code
    principled_bsdf_node.inputs['Roughness'].default_value = roughness
    principled_bsdf_node.inputs["Transmission"].default_value = transmission
    principled_bsdf_node.inputs['Alpha'].default_value = alpha
    principled_bsdf_node.inputs["Transmission Roughness"].default_value = 0.2

    transparent_material.blend_method = blend_mode
    transparent_material.shadow_method = 'HASHED'
    transparent_material.use_screen_refraction = True
    transparent_material.refraction_depth = 0.05

    # ⬇️ IMPORTANT: do NOT clear existing materials – keep the label slot!
    if obj.data.materials:
        obj.data.materials[0] = transparent_material   # plastic in slot 0
    else:
        obj.data.materials.append(transparent_material)


# =========================================================
# PIN GROUP / CLOTH / DEFORM
# (unchanged sections except where commented)
# =========================================================

def create_pin_group_top_bottom(obj, lower_bound=0.1, upper_bound=0.9):
    """
    Pin group = all vertices EXCEPT the middle band.
    Lower/upper bounds are fractions of the object's height.
    """
    PIN_GROUP_NAME = "Bottle_Pin_TopBottom"
    log(f"Creating pin group (top+bottom) '{PIN_GROUP_NAME}'...")

    existing = obj.vertex_groups.get(PIN_GROUP_NAME)
    if existing:
        obj.vertex_groups.remove(existing)

    vg = obj.vertex_groups.new(name=PIN_GROUP_NAME)

    mw = obj.matrix_world
    zs = [(mw @ v.co).z for v in obj.data.vertices]
    z_min, z_max = min(zs), max(zs)
    h = z_max - z_min

    free_low = z_min + lower_bound * h
    free_high = z_min + upper_bound * h

    pin_indices = [
        v.index
        for v in obj.data.vertices
        if (mw @ v.co).z < free_low or (mw @ v.co).z > free_high
    ]

    if pin_indices:
        vg.add(pin_indices, 1.0, 'REPLACE')
        log(f"Pinned {len(pin_indices)} vertices (top+bottom). Middle is free.")
    else:
        log("WARNING: No vertices selected for pin group; check height thresholds.")

    return vg


def setup_cloth_sim(obj, pin_group, pressure_force, shrink_factor):
    log("Adding cloth simulation to bottle...")

    CLOTH_MOD_NAME = "Bottle_Cloth"

    cloth_mod = obj.modifiers.new(name=CLOTH_MOD_NAME, type='CLOTH')
    settings = cloth_mod.settings

    settings.quality = 10
    settings.air_damping = 0.5

    # Shrinking + negative pressure = vacuum-like collapse
    settings.shrink_min = shrink_factor
    settings.use_pressure = True
    settings.uniform_pressure_force = pressure_force
    settings.fluid_density = 0.25

    # Softer bottle-like material – fairly wrinkly
    settings.tension_stiffness = 10.0
    settings.compression_stiffness = 10.0
    settings.shear_stiffness = 10.0
    settings.bending_stiffness = 0.3

    settings.tension_damping = 0.8
    settings.compression_damping = 0.8
    settings.shear_damping = 0.8
    settings.bending_damping = 4.0

    if pin_group is not None:
        settings.vertex_group_mass = pin_group.name
        settings.pin_stiffness = 1.0

    # --- IMPORTANT FOR INNER LAYER ---
    # Enable self-collision so inner + outer surfaces don't pass through each other
    coll = cloth_mod.collision_settings
    coll.use_collision = True
    coll.use_self_collision = True
    coll.distance_min = 0.0015
    coll.self_distance_min = 0.001
    coll.collision_quality = 4

    eff = settings.effector_weights
    if eff is not None:
        eff.gravity = 0.0  # mainly rely on pressure

    log("Cloth simulation configured.")
    return cloth_mod


def press_band_inward(
    obj,
    affected_band_size=0.25,
    scale_factor_y=0.2,
    center_offset=0.0,
):
    """
    Large crease band: press a horizontal band inward on local Y.
    """
    log("Creating big crease band via direct vertex adjustment...")

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    mw = obj.matrix_world
    zs_world = [(mw @ v.co).z for v in obj.data.vertices]
    z_min, z_max = min(zs_world), max(zs_world)
    h = z_max - z_min
    if h == 0:
        log("WARNING: height is zero; aborting band press.")
        return

    target_z = z_min + (0.5 + center_offset) * h
    band_half_height = affected_band_size * h

    ys_local = [v.co.y for v in obj.data.vertices]
    center_y = 0.5 * (min(ys_local) + max(ys_local))

    moved = 0
    for v in obj.data.vertices:
        z_world = (mw @ v.co).z
        dz = abs(z_world - target_z)
        if dz <= band_half_height:
            t = 1.0 - (dz / band_half_height)
            scale_y = 1.0 + (scale_factor_y - 1.0) * t
            ly = v.co.y
            v.co.y = center_y + (ly - center_y) * scale_y
            moved += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    log(
        f"Band press: center_offset={center_offset:.3f}, "
        f"target_z={target_z:.4f}, band_half_height={band_half_height:.4f}, "
        f"moved {moved} vertices."
    )


def create_vertical_bend_curve(obj, bend_offset_factor_x=0.0, bend_offset_factor_y=0.25):
    """
    Simple vertical Bezier curve used for overall bottle bending.
    """
    log("Adding vertical (Z-axis) Bezier curve for bottle bending...")

    CURVE_NAME = "Bottle_Bend_Curve"

    bbox_world = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(bbox_world, Vector((0, 0, 0))) / 8.0
    z_min = min(v.z for v in bbox_world)
    z_max = max(v.z for v in bbox_world)
    height = z_max - z_min

    base_z = z_min

    bpy.ops.curve.primitive_bezier_curve_add(
        enter_editmode=False,
        align='WORLD',
        location=(center.x, center.y, base_z),
        rotation=(0.0, 0.0, 0.0),
    )
    curve_obj = bpy.context.active_object
    curve_obj.name = CURVE_NAME

    curve = curve_obj.data
    spline = curve.splines[0]
    p0, p1 = spline.bezier_points

    span = height
    bend_offset_x = height * bend_offset_factor_x
    bend_offset_y = height * bend_offset_factor_y

    p0.co = Vector((0.0, 0.0, 0.0))
    p1.co = Vector((0.0, 0.0, span))

    p0.handle_right = p0.co + Vector((bend_offset_x, bend_offset_y, span * 0.25))
    p0.handle_left = p0.co
    p1.handle_left = p1.co + Vector((bend_offset_x, bend_offset_y, -span * 0.25))
    p1.handle_right = p1.co

    curve.dimensions = '3D'
    curve.resolution_u = 24

    log(f"Curve '{curve_obj.name}' created with bend offsets.")
    return curve_obj


def add_curve_modifier_after_cloth(obj, curve_obj):
    log("Adding Curve modifier AFTER Cloth...")
    CURVE_MOD_NAME = "Bottle_Curve_Deform"
    mod = obj.modifiers.new(name=CURVE_MOD_NAME, type='CURVE')
    mod.object = curve_obj
    mod.deform_axis = 'POS_Z'
    log("Curve modifier added below Cloth in modifier stack.")


def rotate_object_vertices_z(obj, angle_degrees):
    """
    Rotate vertices around Z (randomizing artwork / label orientation).
    """
    log(f"Rotating vertices of {obj.name} around Z by {angle_degrees} degrees...")

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    angle_radians = math.radians(angle_degrees)
    rot_mat = Matrix.Rotation(angle_radians, 4, 'Z')
    origin = obj.location.copy()

    moved = 0
    for v in obj.data.vertices:
        world_co = obj.matrix_world @ v.co
        rel = world_co - origin
        rotated = rot_mat @ rel
        v.co = obj.matrix_world.inverted() @ (origin + rotated)
        moved += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    log(f"Applied Z rotation of {angle_degrees}° to {moved} vertices.")


# =========================================================
# LATTICE HELPERS (for flattening & height squash)
# =========================================================

def create_bottle_lattice(obj,
                          lattice_name="BottleLattice",
                          points_u=2, points_v=2, points_w=4):
    """
    Create a lattice tightly enclosing obj.
    """
    bpy.context.view_layer.update()
    mw = obj.matrix_world

    xs, ys, zs = [], [], []
    for corner in obj.bound_box:
        w = mw @ mathutils.Vector(corner)
        xs.append(w.x)
        ys.append(w.y)
        zs.append(w.z)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    cz = (z_min + z_max) * 0.5

    sx = (x_max - x_min) * 0.5
    sy = (y_max - y_min) * 0.5
    sz = (z_max - z_min) * 0.5

    lat_data = bpy.data.lattices.new(lattice_name + "Data")
    lat_data.points_u = points_u
    lat_data.points_v = points_v
    lat_data.points_w = points_w

    lat_obj = bpy.data.objects.new(lattice_name, lat_data)
    bpy.context.scene.collection.objects.link(lat_obj)

    lat_obj.location = (cx, cy, cz)
    lat_obj.scale = (sx, sy, sz)
    lat_obj.rotation_euler = obj.matrix_world.to_euler()

    lat_mod = obj.modifiers.new(name=lattice_name + "Mod", type='LATTICE')
    lat_mod.object = lat_obj

    return lat_obj, lat_mod


def flatten_bottle_y_lattice(obj, scale_factor=0.15, keep_center=True, points_v=4):
    """
    Flatten bottle thickness (Y axis) – like being squashed onto its side.
    """
    lat_obj, lat_mod = create_bottle_lattice(
        obj,
        lattice_name="BottleYLattice",
        points_u=2, points_v=points_v, points_w=4,
    )

    lat = lat_obj.data
    pu, pv, pw = lat.points_u, lat.points_v, lat.points_w
    pts = lat.points

    def squash_y(y):
        if keep_center:
            return y * scale_factor
        else:
            t = (y + 1.0) / 2.0
            return -1.0 + 2.0 * scale_factor * t

    for w in range(pw):
        for v in range(pv):
            for u in range(pu):
                idx = w * (pv * pu) + v * pu + u
                p = pts[idx]
                co = p.co_deform.copy()
                co.y = squash_y(co.y)
                p.co_deform = co

    return lat_obj, lat_mod


def squash_bottle_z_lattice(obj, scale_factor=0.5, anchor_mode='BOTTOM', points_w=6):
    """
    Shorten bottle height (Z) using a lattice – gives a more crushed look.
    """
    lat_obj, lat_mod = create_bottle_lattice(
        obj,
        lattice_name="BottleZLattice",
        points_u=2, points_v=2, points_w=points_w,
    )

    lat = lat_obj.data
    pu, pv, pw = lat.points_u, lat.points_v, lat.points_w
    pts = lat.points

    anchor_mode = anchor_mode.upper()

    def squash_z(z):
        if anchor_mode == 'BOTTOM':
            t = (z + 1.0) / 2.0
            return -1.0 + 2.0 * scale_factor * t
        elif anchor_mode == 'TOP':
            t = (1.0 - z) / 2.0
            return 1.0 - 2.0 * scale_factor * t
        else:  # MIDDLE
            return z * scale_factor

    for w in range(pw):
        for v in range(pv):
            for u in range(pu):
                idx = w * (pv * pu) + v * pu + u
                p = pts[idx]
                co = p.co_deform.copy()
                co.z = squash_z(co.z)
                p.co_deform = co

    return lat_obj, lat_mod

# =========================================================
# ORIGIN TO SURFACE COM
# =========================================================

def set_origin_to_evaluated_surface_com(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)
    me = obj_eval.to_mesh()
    if not me or len(me.polygons) == 0:
        if me:
            obj_eval.to_mesh_clear()
        return

    me.calc_loop_triangles()
    verts = me.vertices
    com = mathutils.Vector((0.0, 0.0, 0.0))
    total_area = 0.0
    for tri in me.loop_triangles:
        v0 = verts[tri.vertices[0]].co
        v1 = verts[tri.vertices[1]].co
        v2 = verts[tri.vertices[2]].co
        area = mathutils.geometry.area_tri(v0, v1, v2)
        centroid = (v0 + v1 + v2) / 3.0
        com += centroid * area
        total_area += area

    if total_area > 0:
        com /= total_area
    obj_eval.to_mesh_clear()

    M = obj.matrix_world
    obj.data.transform(mathutils.Matrix.Translation(-com))
    obj.matrix_world = M @ mathutils.Matrix.Translation(M.to_3x3() @ com)

def create_vertical_bend_curve_2_peaks(
    can_obj,
    height_factor=0.5,         # height as a fraction of can height
    end_offset_factor_y=0.02,  # top & bottom Y shift as a fraction of can height
    points_per_arc=64,         # resolution per semicircle
    smooth=True                # NURBS for smoothness
):
    """
    Build a vertical curve made of TWO semicircles in the Y–Z plane:
      • Semicircle 1: bottom (z=0) -> mid (z=H/2), bulging toward +Y
      • Semicircle 2: mid (z=H/2) -> top (z=H), bulging toward -Y
    The bottom and top endpoints are both shifted off the Z axis on +Y by
    end_offset_factor_y * height.

    The curve's vertical extent is EXACTLY the can's height.
    Designed for a Curve modifier set to deform along POS_Z.
    """

    # --- Bounds & dimensions in world space
    bbox_world = [can_obj.matrix_world @ Vector(c) for c in can_obj.bound_box]
    center = sum(bbox_world, Vector((0, 0, 0))) / 8.0
    z_min = min(v.z for v in bbox_world)
    z_max = max(v.z for v in bbox_world)
    height = max(1e-9, (z_max - z_min))  # guard

    height *= height_factor

    # --- Geometry: two stacked semicircles filling the height exactly
    # Total height H = 4R  =>  R = H/4
    R = 0.25 * height
    offset = end_offset_factor_y * height

    # --- Create curve data
    curve_data = bpy.data.curves.new("Can_Bend_CurveData", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 24

    # Choose spline type & allocate points
    npts = points_per_arc * 2  # two semicircles
    if smooth:
        spline = curve_data.splines.new(type='NURBS')
        spline.points.add(npts - 1)
        spline.order_u = 4
        spline.use_endpoint_u = True
    else:
        spline = curve_data.splines.new(type='POLY')
        spline.points.add(npts - 1)

    # --- Fill points (X=0, path lies in Y–Z)
    # Semicircle 1: center (y=0, z=R), theta ∈ [-π/2, +π/2]
    #   y = offset + R*cos(theta)
    #   z = R      + R*sin(theta)
    idx = 0
    for k in range(points_per_arc):
        t = k / (points_per_arc - 1)
        theta = -0.5 * math.pi + t * math.pi
        y = offset + R * math.cos(theta)
        z = R + R * math.sin(theta)               # spans 0 .. 2R
        spline.points[idx].co = (0.0, y, z, 1.0)
        idx += 1

    # Semicircle 2: center (y=0, z=3R), theta ∈ [-π/2, +π/2]
    #   y = offset - R*cos(theta)   (bulge to -Y)
    #   z = 3R    + R*sin(theta)
    for k in range(points_per_arc):
        t = k / (points_per_arc - 1)
        theta = -0.5 * math.pi + t * math.pi
        y = -offset - R * math.cos(theta)
        z = 3.0 * R + R * math.sin(theta)         # spans 2R .. 4R (=H)
        spline.points[idx].co = (0.0, y, z, 1.0)
        idx += 1

    # --- Create object and align the curve's z=0 to the can's bottom
    curve_obj = bpy.data.objects.new("Can_Bend_Curve", curve_data)
    bpy.context.scene.collection.objects.link(curve_obj)
    curve_obj.location = (center.x, center.y, z_min)  # so curve z=0 == can bottom

    # Optional: keep transforms clean for predictable deformation
    # (You can also do this manually via Ctrl+A > Scale in the UI)
    try:
        bpy.context.view_layer.objects.active = curve_obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass

    # Log/debug (if you have a log() helper in your file)
    if 'log' in globals():
        log(
            f"Curve '{curve_obj.name}': two semicircles, H={height:.4f}, "
            f"R={R:.4f}, end_offset_y={end_offset_factor_y:.3f}*H"
        )
    return curve_obj


def make_frozen_copy(obj, name_suffix="_EXPORT"):
    """Duplicate `obj` and apply all modifiers so the result is just a mesh."""
    # Make sure obj is active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Duplicate
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    dup.name = obj.name + name_suffix

    # Apply all modifiers on the duplicate
    for mod in list(dup.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)

    return dup

def save_object_to_blend(obj, filepath):
    """
    Save `obj` and its data (mesh, materials, images) into a separate .blend.
    No cameras/lights/other objects.
    """
    ids = set()

    # Object and its mesh
    ids.add(obj)
    if obj.data:
        ids.add(obj.data)

    # Materials & images used in those materials
    for slot in obj.material_slots:
        mat = slot.material
        if mat:
            ids.add(mat)
            if mat.use_nodes and mat.node_tree:
                for node in mat.node_tree.nodes:
                    if isinstance(node, bpy.types.ShaderNodeTexImage) and node.image:
                        ids.add(node.image)

    # Write a tiny library .blend with just these datablocks
    bpy.data.libraries.write(filepath, ids, path_remap='RELATIVE')

    print("Saved object library:", filepath)

# def save_single_object_blend(obj, filepath):
#     """
#     Save a .blend file containing only this object and the data it depends on.
#     Nothing else.
#     """
#     ids = set()

#     # Object and its mesh
#     ids.add(obj)
#     if obj.data:
#         ids.add(obj.data)

#     # Materials + images used by that object
#     for slot in obj.material_slots:
#         mat = slot.material
#         if mat:
#             ids.add(mat)
#             if mat.use_nodes and mat.node_tree:
#                 for node in mat.node_tree.nodes:
#                     if isinstance(node, bpy.types.ShaderNodeTexImage) and node.image:
#                         ids.add(node.image)

#     # Save only these datablocks
#     bpy.data.libraries.write(filepath, ids, path_remap='RELATIVE')

#     print("Saved:", filepath)


def append_single_object(blend_path, object_name):
    """
    Append ONE object from a .blend file.
    The object will be copied into the current scene.
    """
    directory = blend_path + "/Object/"
    filename  = object_name

    bpy.ops.wm.append(
        filepath=directory + filename,
        directory=directory,
        filename=filename,
        link=False   # set True if you want linked library objects
    )

# =========================================================
# MAIN
# =========================================================

def main():
    clear_scene()

    bottle_obj = import_water_bottle(FBX_FILEPATH)
    #bottle_obj = create_bottle_from_scratch()
    #bottle_obj = create_bottle_from_scratch(shape_id=0)

    # for i in range(10):
    #     obj = create_bottle_from_scratch(name=f"WaterBottle_{i}", shape_id=i)
    #     obj.location.x = i * 0.09  # spread them along X


    bpy.context.view_layer.objects.active = bottle_obj
    bottle_obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bottle_obj.select_set(False)

    label_mat = create_label_material()
    assign_label_to_middle_band(bottle_obj, label_mat)
    

    # Randomize label orientation
    angle = np.random.uniform(0, 360)
    #angle = 90
    #rotate_object_vertices_z(bottle_obj, angle)
    bottle_obj.rotation_euler.z += math.radians(angle)

    # 1) Pre-crease: a strong band dent to hint a fold
    band_half_size = np.random.uniform(0.2, 0.5)
    scale_factor_y = np.random.uniform(0.1, 0.4)  # strong inward
    center_offset = np.random.uniform(-0.25, 0.25)  # fold not exactly center
    press_band_inward(bottle_obj, band_half_size, scale_factor_y, center_offset)

    press_band_inward_twice_probability = 0.3
    if np.random.rand() < press_band_inward_twice_probability:
        band_half_size = np.random.uniform(0.2, 0.5)
        scale_factor_y = np.random.uniform(0.1, 0.4)
        center_offset = np.random.uniform(-0.25, 0.25)
        #press_band_inward(bottle_obj, band_half_size, scale_factor_y, center_offset)

    # 2) Cloth "vacuum" collapse, middle free, top+bottom lightly pinned
    lower_bound = np.random.uniform(0.05, 0.1)
    upper_bound = np.random.uniform(0.75, 0.9)
    pin_group = create_pin_group_top_bottom(bottle_obj, lower_bound, upper_bound)

    pressure_force = np.random.uniform(-300, -50)   # strong vacuum
    shrink_factor = np.random.uniform(-0.4, -0.02)  # contract bottle surface
    #setup_cloth_sim(bottle_obj, pin_group, pressure_force, shrink_factor)

    # 3) Bend bottle with a curve (big folded arc)
    two_peaks_vertical_squash_probability = 0.3
    if np.random.rand() < two_peaks_vertical_squash_probability:
        height_factor = np.random.uniform(0.1, 0.9)
        end_offset_factor_y = np.random.uniform(-0.15, 0.15)
        curve_obj = create_vertical_bend_curve_2_peaks(
            bottle_obj,
            height_factor=height_factor,
            end_offset_factor_y=end_offset_factor_y,
            points_per_arc=64,
            smooth=True
        )
    else:
        bend_offset_factor_x = np.random.uniform(-0.1, 0.1)
        bend_offset_factor_y = np.random.uniform(-0.6, 0.6)
        print(f"Bend offset factor X: {bend_offset_factor_x}, Y: {bend_offset_factor_y}")
        curve_obj = create_vertical_bend_curve(bottle_obj, bend_offset_factor_x, bend_offset_factor_y)

    #add_curve_modifier_after_cloth(bottle_obj, curve_obj)

    flatten_probability = 0.5
    if np.random.rand() < flatten_probability:
        y_scale = np.random.uniform(0.1, 0.5)
        #flatten_bottle_y_lattice(bottle_obj, scale_factor=y_scale, keep_center=True, points_v=4)

    z_scale = np.random.uniform(0.35, 1)
    #squash_bottle_z_lattice(bottle_obj, scale_factor=z_scale, anchor_mode='BOTTOM', points_w=6)

    # Let cloth evaluate at a mid frame
    bpy.context.scene.frame_set(5)
    log("Timeline set to frame 5")

    # Apply transforms & set origin
    bpy.ops.object.select_all(action='DESELECT')
    bottle_obj.select_set(True)
    bpy.context.view_layer.objects.active = bottle_obj

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    set_origin_to_evaluated_surface_com(bottle_obj)

    # --- TRANSPARENT PLASTIC WITH LABEL ---
    # Color code is RGBA; keep RGB white so the label colors show nicely.
    make_object_transparent(
        bottle_obj,
        blend_mode='BLEND',
        roughness=0.1,
        transmission=0.9,
        alpha=0.1,
        color_code=(1.0, 1.0, 1.0, 1.0)
    )

    bpy.ops.object.select_all(action='DESELECT')
    bottle_obj.select_set(True)
    bpy.context.view_layer.objects.active = bottle_obj

    #bottle_obj = bpy.data.objects["Bottle"]  # your simulated/crushed bottle
    frozen_bottle = make_frozen_copy(bottle_obj)
    frozen_bottle.name = "MyFinalBottle" 
    save_object_to_blend(frozen_bottle, BLEND_EXPORT_PATH)
    #save_single_object_blend(frozen_bottle, BLEND_EXPORT_PATH)

    # Save as .blend to preserve materials, modifiers, etc.
    log(f"Saving .blend to: {BLEND_EXPORT_PATH}")
    #bpy.ops.wm.save_as_mainfile(filepath=BLEND_EXPORT_PATH)
    log("Done.")

    # append_single_object(
    #     "/home/tem/Waste-Dataset-Generation/res_fbx_objects/bottle_crushed.blend",
    #     "MyFinalBottle"
    # )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")

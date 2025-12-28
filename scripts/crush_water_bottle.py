# Blender 3.6.x
# Water Bottle Crushing (vacuum + flatten + fold)
# ------------------------------------------------
# - Clears scene
# - Imports water bottle FBX (aligned to Z)
# - Adds a strong cloth "vacuum" collapse
# - Adds a large crease band
# - Flattens thickness and shortens height with lattices
# - Bends bottle along a vertical curve (S-shaped fold)
# - Exports the crushed bottle as FBX

import bpy  # type: ignore
import math
import numpy as np
from mathutils import Vector, Matrix
import mathutils


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


def import_water_bottle(filepath):
    """
    Import FBX and try to pick the water-bottle mesh.
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

    bottle_obj = mesh_candidates[0]
    log(f"Using bottle object: {bottle_obj.name}")
    return bottle_obj


# =========================================================
# PIN GROUP / CLOTH / DEFORM
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

    affected_band_size : half-height of band as fraction of object height
    scale_factor_y     : Y scale at band center (<1 for inward dent)
    center_offset      : -0.5 bottom .. 0 middle .. +0.5 top
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

    # Center of band in world Z
    target_z = z_min + (0.5 + center_offset) * h
    band_half_height = affected_band_size * h

    ys_local = [v.co.y for v in obj.data.vertices]
    center_y = 0.5 * (min(ys_local) + max(ys_local))

    moved = 0
    for v in obj.data.vertices:
        z_world = (mw @ v.co).z
        dz = abs(z_world - target_z)
        if dz <= band_half_height:
            t = 1.0 - (dz / band_half_height)  # 1 at center, 0 at edges
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


# =========================================================
# MAIN
# =========================================================

def main():
    # --- EDIT THESE PATHS ---
    fbx_filepath = r"/home/tem/Waste-Dataset-Generation/fbx_standard/pet/pet1.fbx"
    export_path = r"/home/tem/Waste-Dataset-Generation/res_fbx_objects/bottle_crushed.fbx"

    clear_scene()

    bottle_obj = import_water_bottle(fbx_filepath)

    bpy.context.view_layer.objects.active = bottle_obj
    bottle_obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bottle_obj.select_set(False)

    # Randomize label orientation
    angle = np.random.uniform(0, 360)
    rotate_object_vertices_z(bottle_obj, angle)

    # 1) Pre-crease: a strong band dent to hint a fold
    band_half_size = np.random.uniform(0.2, 0.5)
    scale_factor_y = np.random.uniform(0.1, 0.4)  # strong inward
    center_offset = np.random.uniform(-0.25, 0.25)  # fold not exactly center
    press_band_inward(bottle_obj, band_half_size, scale_factor_y, center_offset)

    press_band_inward_twice_probability = 0.3
    if np.random.rand() < press_band_inward_twice_probability:
        band_half_size = np.random.uniform(0.2, 0.5)
        scale_factor_y = np.random.uniform(0.1, 0.4)  # strong inward
        center_offset = np.random.uniform(-0.25, 0.25)  # fold not exactly center
        press_band_inward(bottle_obj, band_half_size, scale_factor_y, center_offset)

    # 2) Cloth "vacuum" collapse, middle free, top+bottom lightly pinned
    lower_bound = np.random.uniform(0.05, 0.1)
    upper_bound = np.random.uniform(0.75, 0.9)
    pin_group = create_pin_group_top_bottom(bottle_obj, lower_bound, upper_bound)

    pressure_force = np.random.uniform(-300, -50)   # strong vacuum
    shrink_factor = np.random.uniform(-0.4, -0.02)   # contract bottle surface
    setup_cloth_sim(bottle_obj, pin_group, pressure_force, shrink_factor)

    # 3) Bend bottle with a curve (big folded arc)
    # bend_offset_factor_x = np.random.uniform(-0.1, 0.1)
    # bend_offset_factor_y = np.random.uniform(-0.6, 0.6)
    # curve_obj = create_vertical_bend_curve(
    #     bottle_obj,
    #     bend_offset_factor_x=bend_offset_factor_x,
    #     bend_offset_factor_y=bend_offset_factor_y,
    # )
    two_peaks_vertical_squash_probability = 0.3
    if np.random.rand() < two_peaks_vertical_squash_probability:
        height_factor = np.random.uniform(0.1, 0.9)
        end_offset_factor_y = np.random.uniform(-0.15, 0.15)
        curve_obj = create_vertical_bend_curve_2_peaks(bottle_obj, height_factor=height_factor, end_offset_factor_y=end_offset_factor_y, points_per_arc=64, smooth=True)
    else:
        bend_offset_factor_x = np.random.uniform(-0.1, 0.1)
        bend_offset_factor_y = np.random.uniform(-0.6, 0.6)
        print(f"Bend offset factor X: {bend_offset_factor_x}, Y: {bend_offset_factor_y}")
        curve_obj = create_vertical_bend_curve(bottle_obj, bend_offset_factor_x, bend_offset_factor_y)
    add_curve_modifier_after_cloth(bottle_obj, curve_obj)

    flatten_probability = 0.5
    if np.random.rand() < flatten_probability:
        # 4) Flatten thickness (Y) – bottle becomes very thin like in reference
        y_scale = np.random.uniform(0.1, 0.5)
        flatten_bottle_y_lattice(bottle_obj, scale_factor=y_scale, keep_center=True, points_v=4)

    # 5) Shorten height (Z) a bit for a “crumpled and shoved down” look
    z_scale = np.random.uniform(0.35, 1)
    squash_bottle_z_lattice(bottle_obj, scale_factor=z_scale, anchor_mode='BOTTOM', points_w=6)

    # Let cloth evaluate at a mid frame
    bpy.context.scene.frame_set(5)
    log("Timeline set to frame 5")

    # Prepare for export
    bpy.ops.object.select_all(action='DESELECT')
    bottle_obj.select_set(True)
    bpy.context.view_layer.objects.active = bottle_obj

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    set_origin_to_evaluated_surface_com(bottle_obj)

    bpy.ops.object.select_all(action='DESELECT')
    bottle_obj.select_set(True)
    bpy.context.view_layer.objects.active = bottle_obj

    bpy.ops.export_scene.fbx(
        filepath=export_path,
        use_selection=True,
        path_mode='COPY',
        embed_textures=True,
    )
    log(f"Exported crushed bottle to: {export_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")

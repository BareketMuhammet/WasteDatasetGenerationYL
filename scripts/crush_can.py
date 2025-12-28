# Blender 3.6.9
# Soda Can Crushing (Cloth first, midline press-in, then Curve bend)
# -----------------------------------------------------------------
# - Clears scene
# - Imports soda can FBX (aligned to Z)
# - Creates pin group: ALL vertices EXCEPT middle band (top+bottom pinned)
# - Adds Cloth sim (runs on straight/pressed can)
# - Presses a horizontal midline inward on Y with proportional edit
# - Creates vertical Bezier curve along Z and adds Curve modifier AFTER Cloth
# - Gravity off, negative pressure, timeline to frame 5

import bpy # type: ignore
import math
import numpy as np
from mathutils import Vector


# =========================================================
# UTILITIES
# =========================================================

def log(msg):
    print(f"[SODA_CAN_SETUP] {msg}")

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
    ):
        for datablock in list(datablock_collection):
            if datablock.users == 0:
                datablock_collection.remove(datablock)

def import_soda_can(filepath):
    log(f"Importing soda can FBX from: {filepath}")
    CAN_OBJECT_NAME_HINT = "Can"

    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=filepath)
    after = set(bpy.context.scene.objects)

    new_objects = list(after - before)
    mesh_candidates = [o for o in new_objects if o.type == 'MESH']

    if CAN_OBJECT_NAME_HINT:
        for o in mesh_candidates:
            if CAN_OBJECT_NAME_HINT.lower() in o.name.lower():
                log(f"Using can object by hint: {o.name}")
                return o

    if not mesh_candidates:
        raise RuntimeError("No mesh objects found in imported FBX.")

    can_obj = mesh_candidates[0]
    log(f"Using can object: {can_obj.name}")
    return can_obj

def create_pin_group_top_bottom(can_obj, lower_bound=0.15, upper_bound=0.85):
    """
    Pin group = all vertices EXCEPT the middle band.
    Middle band (MIDDLE_LOW..MIDDLE_HIGH) is free to deform/crush.
    """
    PIN_GROUP_NAME = "Can_Pin_TopBottom"
    log(f"Creating pin group (top+bottom) '{PIN_GROUP_NAME}'...")

    existing = can_obj.vertex_groups.get(PIN_GROUP_NAME)
    if existing:
        can_obj.vertex_groups.remove(existing)

    vg = can_obj.vertex_groups.new(name=PIN_GROUP_NAME)

    mw = can_obj.matrix_world
    zs = [(mw @ v.co).z for v in can_obj.data.vertices]
    z_min, z_max = min(zs), max(zs)
    h = z_max - z_min

    free_low = z_min + lower_bound * h
    free_high = z_min + upper_bound * h

    pin_indices = [
        v.index
        for v in can_obj.data.vertices
        if (mw @ v.co).z < free_low or (mw @ v.co).z > free_high
    ]

    if pin_indices:
        vg.add(pin_indices, 1.0, 'REPLACE')
        log(f"Pinned {len(pin_indices)} vertices (top+bottom). Middle is free.")
    else:
        log("WARNING: No vertices selected for pin group; check height thresholds.")

    return vg

def setup_cloth_sim(can_obj, pin_group, pressure_force, shrink_factor):
    log("Adding cloth simulation to soda can...")

    # pressure_force More negative values cause stronger inward collapse
    # shrink_factor small negative value simulates “contracting metal” → increases fine creasing.

    CLOTH_MOD_NAME = "Can_Cloth"

    cloth_mod = can_obj.modifiers.new(name=CLOTH_MOD_NAME, type='CLOTH')
    settings = cloth_mod.settings

    settings.quality = 10

    # Air damping
    settings.air_damping = 0.5

    # Shrink / shape
    settings.shrink_min = shrink_factor

    # Pressure
    settings.use_pressure = True
    settings.uniform_pressure_force = pressure_force
    settings.fluid_density = 0.2

    # Stiffness
    settings.tension_stiffness = 15.0 # Resistance to stretching Lower = more wrinkly and saggy
    settings.compression_stiffness = 15.0 # Resistance to compression Lower = more folds under pressure
    settings.shear_stiffness = 15.0 # Resistance to shearing Lower = allows more diagonal wrinkles
    settings.bending_stiffness = 0.5 # Resistance to bending Lower = smaller, sharper wrinkles; Higher = smoother, large-scale bending

    # Damping
    settings.tension_damping = 1.0 # How fast motion smooths out, Lower damping = wrinkles form and oscillate longer
    settings.compression_damping = 1.0
    settings.shear_damping = 1.0
    settings.bending_damping = 5.0

    # Pinning: vertex_group_mass + pin_stiffness
    if pin_group is not None:
        settings.vertex_group_mass = pin_group.name
        settings.pin_stiffness = 1.0

    # Field Weights: disable gravity
    eff = settings.effector_weights
    if eff is not None:
        eff.gravity = 0.0

    log("Cloth simulation configured.")
    return cloth_mod

# def press_can_middle_inward(can_obj):
#     """
#     Select a horizontal ring of vertices near the middle
#     and scale along Y with proportional editing to press inward.
#     """
#     log("Pressing can middle inward with proportional scaling...")

#     bpy.context.view_layer.objects.active = can_obj
#     bpy.ops.object.mode_set(mode='OBJECT')

#     mw = can_obj.matrix_world
#     zs = [(mw @ v.co).z for v in can_obj.data.vertices]
#     z_min, z_max = min(zs), max(zs)
#     h = z_max - z_min

#     target_z = z_min + 0.5 * h
#     tol = MIDLINE_SELECT_TOLERANCE * h

#     # Clear selection
#     for v in can_obj.data.vertices:
#         v.select = False

#     # Select vertices close to the midline
#     selected_count = 0
#     for v in can_obj.data.vertices:
#         if abs((mw @ v.co).z - target_z) <= tol:
#             v.select = True
#             selected_count += 1
#             #print(v.co.z)

#     if selected_count == 0:
#         log("WARNING: No vertices selected for midline press; skipping.")
#         bpy.ops.object.mode_set(mode='OBJECT')
#         return

#     # Enter Edit mode for vertex transform
#     bpy.ops.object.mode_set(mode='EDIT')
#     ts = bpy.context.tool_settings

#      # Use proportional *connected* so only nearby geometry is affected
#     ts.proportional_edit = 'CONNECTED'

#     # # Required settings from your request
#     # ts.use_proportional_edit_objects = True  # affects object-mode proportional, set anyway
#     # if hasattr(ts, "use_transform_correct_face_attributes"):
#     #     ts.use_transform_correct_face_attributes = True
#     # if hasattr(ts, "use_transform_correct_keep_connected"):
#     #     ts.use_transform_correct_keep_connected = True

#     # Proportional size relative to can height
#     #prop_size = MIDLINE_PROP_SIZE_FACTOR * h
#     #prop_size = 0.03
    
#     MIDLINE_PROP_SIZE_FACTOR = 0.2
#     prop_size = MIDLINE_PROP_SIZE_FACTOR * h
#     #ts.proportional_distance = prop_dist

#     # Apply resize on Y with proportional edit enabled
#     # bpy.ops.transform.resize(
#     #     value=(1.0, MIDLINE_SCALE_Y, 1.0),
#     #     orient_type='LOCAL',
#     #     constraint_axis=(False, True, False),
#     #     mirror=False,
#     #     use_proportional_edit=True,
#     #     proportional_edit_falloff='SMOOTH',
#     #     proportional_size=prop_dist,
#     #     use_proportional_connected=False
#     # )   

#     bpy.ops.transform.resize(
#         value=(1.0, MIDLINE_SCALE_Y, 1.0),
#         orient_type='LOCAL',
#         constraint_axis=(False, True, False),
#         use_proportional_edit=True,
#         proportional_edit_falloff='SMOOTH',
#         proportional_size=prop_size,
#         use_proportional_connected=True
#     )
    


#     bpy.ops.object.mode_set(mode='OBJECT')
#     log(f"Midline pressed inward on Y with proportional size {prop_size:.4f} (selected {selected_count} verts).")

# def press_can_middle_inward(can_obj, affected_band_size=0.3, scale_factor_y=0.2):
#     """
#     Press only a narrow horizontal band around the middle of the can inward on Y
#     by directly editing vertex positions. No proportional-edit operators,
#     so the top and bottom will NOT follow.
#     """
#     log("Pressing can middle inward via direct vertex adjustment...")

#     bpy.context.view_layer.objects.active = can_obj
#     bpy.ops.object.mode_set(mode='OBJECT')

#     mw = can_obj.matrix_world

#     # World-space Z extents to find vertical middle
#     zs_world = [(mw @ v.co).z for v in can_obj.data.vertices]
#     z_min, z_max = min(zs_world), max(zs_world)
#     h = z_max - z_min

#     if h == 0:
#         log("WARNING: Can height is zero; aborting midline press.")
#         return

#     target_z = z_min + 0.5 * h
#     band_half_height = affected_band_size * h  # how tall the affected band is

#     # Local-space Y center (used as pivot for inward press)
#     ys = [v.co.y for v in can_obj.data.vertices]
#     center_y = 0.5 * (min(ys) + max(ys))

#     moved = 0

#     for v in can_obj.data.vertices:
#         # Compute vertical distance in world space
#         z_world = (mw @ v.co).z
#         dz = abs(z_world - target_z)

#         if dz <= band_half_height:
#             # Smooth falloff: 1 at center, 0 at edge of band
#             if band_half_height > 0:
#                 t = 1.0 - (dz / band_half_height)
#             else:
#                 t = 1.0

#             # Blend between no change (1.0) and MIDLINE_SCALE_Y (<1) based on t
#             # So only vertices near exact midline get full indentation.
#             scale_y = 1.0 + (scale_factor_y - 1.0) * t

#             # Apply in local space
#             ly = v.co.y
#             v.co.y = center_y + (ly - center_y) * scale_y

#             moved += 1

#     bpy.ops.object.mode_set(mode='OBJECT')
#     log(f"Midline press applied to {moved} vertices within band +/- {band_half_height:.4f}.")


def press_can_middle_inward(
    can_obj,
    affected_band_size=0.3,
    scale_factor_y=0.2,
    center_offset=0.0
):
    """
    Press a horizontal band of the can inward on Y by directly editing vertices.

    Params:
        can_obj:            The can mesh object.
        affected_band_size: Half-height of the affected band as a fraction of can height.
                            e.g. 0.1 => band spans 20% of height total.
        scale_factor_y:     Target Y scale at the band center (< 1.0 for inward dent).
        center_offset:      Vertical offset of band center, in normalized units:
                            0.0 = middle, -0.5 = bottom, +0.5 = top.
    """
    log("Pressing can band inward via direct vertex adjustment...")

    bpy.context.view_layer.objects.active = can_obj
    bpy.ops.object.mode_set(mode='OBJECT')

    mw = can_obj.matrix_world

    # World-space Z extents to find can height
    zs_world = [(mw @ v.co).z for v in can_obj.data.vertices]
    z_min, z_max = min(zs_world), max(zs_world)
    h = z_max - z_min

    if h == 0:
        log("WARNING: Can height is zero; aborting band press.")
        return

    # Base middle
    base_center_z = z_min + 0.5 * h

    # Shift center by normalized offset:
    # center_offset in [-0.5, 0.5] maps bottom..top
    target_z = z_min + (0.5 + center_offset) * h

    # Half-height of affected band
    band_half_height = affected_band_size * h
    if band_half_height <= 0:
        log("WARNING: affected_band_size too small; aborting band press.")
        return

    # Local-space Y center as pivot
    ys = [v.co.y for v in can_obj.data.vertices]
    center_y = 0.5 * (min(ys) + max(ys))

    moved = 0

    for v in can_obj.data.vertices:
        z_world = (mw @ v.co).z
        dz = abs(z_world - target_z)

        if dz <= band_half_height:
            # Smooth falloff: 1 at target_z, 0 at band edge
            t = 1.0 - (dz / band_half_height)

            # Interpolate between no change (1.0) and scale_factor_y at center
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




def create_vertical_bend_curve(can_obj, bend_offset_factor_x=0.1, bend_offset_factor_y=0.1):
    log("Adding vertical (Z-axis) Bezier curve for can bending...")

    CURVE_NAME = "Can_Bend_Curve"

    bbox_world = [can_obj.matrix_world @ Vector(c) for c in can_obj.bound_box]
    center = sum(bbox_world, Vector((0, 0, 0))) / 8.0
    z_min = min(v.z for v in bbox_world)
    z_max = max(v.z for v in bbox_world)
    height = z_max - z_min

    # Place at can center X/Y, bottom Z
    base_z = z_min

    bpy.ops.curve.primitive_bezier_curve_add(
        enter_editmode=False,
        align='WORLD',
        location=(center.x, center.y, base_z),
        rotation=(0.0, 0.0, 0.0)
    )
    curve_obj = bpy.context.active_object
    curve_obj.name = CURVE_NAME

    curve = curve_obj.data
    spline = curve.splines[0]
    p0, p1 = spline.bezier_points

    span = height * 0.5
    bend_offset_x = height * bend_offset_factor_x
    bend_offset_y = height * bend_offset_factor_y 

    # Vertical along local Z
    p0.co = Vector((0.0, 0.0, 0.0))
    p1.co = Vector((0.0, 0.0, span))

    p0.handle_right = p0.co + Vector((bend_offset_x, bend_offset_y, span * 0.25))
    p0.handle_left  = p0.co
    p1.handle_left  = p1.co + Vector((bend_offset_x, bend_offset_y, -span * 0.25))
    p1.handle_right = p1.co

    curve.dimensions = '3D'
    curve.resolution_u = 24

    log(f"Curve '{curve_obj.name}' created: vertical with slight bend.")
    return curve_obj

def add_curve_modifier_after_cloth(can_obj, curve_obj):
    log("Adding Curve modifier AFTER Cloth (cloth runs first)...")
    CURVE_MOD_NAME = "Can_Curve_Deform"
    mod = can_obj.modifiers.new(name=CURVE_MOD_NAME, type='CURVE')
    mod.object = curve_obj
    mod.deform_axis = 'POS_Z'
    # Since Cloth was created first, Curve is appended after it.
    log("Curve modifier added below Cloth in modifier stack.")

import math
from mathutils import Matrix, Vector

def rotate_object_vertices_z(can_obj, angle_degrees):
    """
    Apply a *real* rotation to all vertices of an object around the Z axis,
    centered on the object's origin. This permanently changes vertex positions.

    Params:
        can_obj: The mesh object to rotate (must be a mesh).
        angle_degrees: Rotation angle in degrees (positive = counterclockwise).
    """
    log(f"Rotating vertices of {can_obj.name} around Z by {angle_degrees} degrees...")

    bpy.context.view_layer.objects.active = can_obj
    bpy.ops.object.mode_set(mode='OBJECT')

    angle_radians = math.radians(angle_degrees)

    # Rotation matrix around Z
    rot_mat = Matrix.Rotation(angle_radians, 4, 'Z')

    # Optionally rotate around object's origin or a custom pivot
    origin = can_obj.location.copy()

    moved = 0
    for v in can_obj.data.vertices:
        # Convert to world space
        world_co = can_obj.matrix_world @ v.co
        # Translate to origin-relative
        rel = world_co - origin
        # Apply rotation
        rotated = rot_mat @ rel
        # Translate back and convert to local space
        v.co = can_obj.matrix_world.inverted() @ (origin + rotated)
        moved += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    log(f"Applied Z rotation of {angle_degrees}° to {moved} vertices.")

def compress_can_along_z(can_obj, scale_factor=0.5, anchor_mode='BOTTOM'):
    """
    Compress the can along its Z axis by editing vertices directly.

    Params:
        can_obj:     The can mesh object.
        scale_factor:
            < 1.0 => squash (shorter can)
            = 1.0 => no change
            > 1.0 => stretch (taller can)
        anchor_mode: 'BOTTOM', 'TOP', or 'MIDDLE'
            Defines which vertical region stays most in place.
    """
    log(f"Compressing can along Z: scale_factor={scale_factor}, anchor_mode={anchor_mode}")

    bpy.context.view_layer.objects.active = can_obj
    bpy.ops.object.mode_set(mode='OBJECT')

    mw = can_obj.matrix_world

    # Compute world-space vertical range
    zs = [(mw @ v.co).z for v in can_obj.data.vertices]
    z_min, z_max = min(zs), max(zs)
    height = z_max - z_min

    if height == 0:
        log("WARNING: Can height is zero; aborting Z compression.")
        return

    # Choose pivot
    if anchor_mode.upper() == 'TOP':
        pivot_z = z_max
    elif anchor_mode.upper() == 'MIDDLE':
        pivot_z = 0.5 * (z_min + z_max)
    else:  # 'BOTTOM' or default
        pivot_z = z_min

    moved = 0
    inv_mw = mw.inverted()

    for v in can_obj.data.vertices:
        w = mw @ v.co
        # Distance from pivot normalized by original height
        t = (w.z - pivot_z) / height
        # New position: pivot + scaled offset
        w.z = pivot_z + t * height * scale_factor
        v.co = inv_mw @ w
        moved += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    log(f"Z compression applied to {moved} vertices (anchor={anchor_mode}).")


def compress_can_along_y(can_obj, scale_factor=0.5, keep_center=True):
    """
    Compress the can along its Y axis (side-to-side) by editing vertices.

    Params:
        can_obj:     The can mesh object.
        scale_factor:
            < 1.0 => thinner can in Y
            = 1.0 => no change
            > 1.0 => fatter in Y
        keep_center:
            If True, squeeze around the mid Y so the can stays centered.
            If False, uses object origin as the pivot.
    """
    log(f"Compressing can along Y: scale_factor={scale_factor}, keep_center={keep_center}")

    bpy.context.view_layer.objects.active = can_obj
    bpy.ops.object.mode_set(mode='OBJECT')

    mw = can_obj.matrix_world

    # Determine pivot in LOCAL space (since we only scale Y)
    if keep_center:
        ys_local = [v.co.y for v in can_obj.data.vertices]
        pivot_y = 0.5 * (min(ys_local) + max(ys_local))
    else:
        pivot_y = 0.0  # use object origin's local Y

    moved = 0

    for v in can_obj.data.vertices:
        ly = v.co.y
        # Scale around pivot_y
        v.co.y = pivot_y + (ly - pivot_y) * scale_factor
        moved += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    log(f"Y compression applied to {moved} vertices (pivot_y={pivot_y:.4f}).")

import bpy
import mathutils

# ------------------------------
# Lattice helper
# ------------------------------

def create_can_lattice(can_obj,
                       lattice_name="CanLattice",
                       points_u=2, points_v=2, points_w=4):
    """
    Create a lattice object that tightly bounds the can and is aligned to it.

    Returns: (lat_obj, lat_mod) where:
        lat_obj is the lattice object
        lat_mod is the Lattice modifier on can_obj
    """
    # Ensure transforms are up to date
    bpy.context.view_layer.update()

    mw = can_obj.matrix_world

    # Compute world-space bounding box of the can
    xs, ys, zs = [], [], []
    for corner in can_obj.bound_box:
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

    # Lattice data
    lat_data = bpy.data.lattices.new(lattice_name + "Data")
    lat_data.points_u = points_u
    lat_data.points_v = points_v
    lat_data.points_w = points_w

    # Lattice object
    lat_obj = bpy.data.objects.new(lattice_name, lat_data)
    bpy.context.scene.collection.objects.link(lat_obj)

    # Align lattice to can:
    # - place at bbox center
    # - size from bbox
    # - copy can rotation so axes line up
    lat_obj.location = (cx, cy, cz)
    lat_obj.scale = (sx, sy, sz)
    lat_obj.rotation_euler = can_obj.matrix_world.to_euler()

    # Add lattice modifier to can
    lat_mod = can_obj.modifiers.new(name=lattice_name + "Mod", type='LATTICE')
    lat_mod.object = lat_obj

    return lat_obj, lat_mod


# ------------------------------
# Compress along Z (height)
# ------------------------------

def compress_can_along_z_lattice(can_obj,
                                 scale_factor=0.5,
                                 anchor_mode='BOTTOM',
                                 points_w=6):
    """
    Squash the can along Z using a lattice.

    scale_factor:
        < 1.0 -> shorter (crushed)
        = 1.0 -> no change
        > 1.0 -> taller (stretch)
    anchor_mode:
        'BOTTOM' -> bottom stays, top moves
        'TOP'    -> top stays, bottom moves
        'MIDDLE' -> squash symmetrically around center
    """
    lat_obj, lat_mod = create_can_lattice(
        can_obj,
        lattice_name="CanZLattice",
        points_u=2, points_v=2, points_w=points_w
    )

    lat = lat_obj.data
    pu, pv, pw = lat.points_u, lat.points_v, lat.points_w
    pts = lat.points

    anchor_mode = anchor_mode.upper()

    def squash_z(z):
        # lattice local z is in [-1, 1]
        if anchor_mode == 'BOTTOM':
            # bottom (-1) fixed, scale upwards
            t = (z + 1.0) / 2.0          # 0 at bottom, 1 at top
            return -1.0 + 2.0 * scale_factor * t
        elif anchor_mode == 'TOP':
            # top (1) fixed, scale downwards
            t = (1.0 - z) / 2.0          # 0 at top, 1 at bottom
            return 1.0 - 2.0 * scale_factor * t
        else:
            # MIDDLE: symmetric around 0
            return z * scale_factor

    # Modify Z for each lattice point
    for w in range(pw):
        for v in range(pv):
            for u in range(pu):
                idx = w * (pv * pu) + v * pu + u
                p = pts[idx]
                co = p.co_deform.copy()
                co.z = squash_z(co.z)
                p.co_deform = co

    # Optionally apply & remove lattice for a baked result:
    # bpy.context.view_layer.objects.active = can_obj
    # bpy.ops.object.modifier_apply(modifier=lat_mod.name)
    # bpy.data.objects.remove(lat_obj, do_unlink=True)


# ------------------------------
# Compress along Y (side-to-side)
# ------------------------------

def compress_can_along_y_lattice(can_obj,
                                 scale_factor=0.5,
                                 keep_center=True,
                                 points_v=4):
    """
    Squash the can along Y using a lattice.

    scale_factor:
        < 1.0 -> thinner in Y
        = 1.0 -> no change
        > 1.0 -> fatter in Y
    keep_center:
        True  -> squash around centerline (symmetrical)
        False -> squash toward negative Y side
    """
    lat_obj, lat_mod = create_can_lattice(
        can_obj,
        lattice_name="CanYLattice",
        points_u=2, points_v=points_v, points_w=4
    )

    lat = lat_obj.data
    pu, pv, pw = lat.points_u, lat.points_v, lat.points_w
    pts = lat.points

    def squash_y(y):
        # y in [-1, 1] in lattice space
        if keep_center:
            # symmetric around 0
            return y * scale_factor
        else:
            # anchor at -1 (one side fixed)
            t = (y + 1.0) / 2.0          # 0 at -1, 1 at +1
            return -1.0 + 2.0 * scale_factor * t

    for w in range(pw):
        for v in range(pv):
            for u in range(pu):
                idx = w * (pv * pu) + v * pu + u
                p = pts[idx]
                co = p.co_deform.copy()
                co.y = squash_y(co.y)
                p.co_deform = co

    # Optionally:
    # bpy.context.view_layer.objects.active = can_obj
    # bpy.ops.object.modifier_apply(modifier=lat_mod.name)
    # bpy.data.objects.remove(lat_obj, do_unlink=True)

def set_origin_to_evaluated_surface_com(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)
    me = obj_eval.to_mesh()  # evaluated mesh (includes modifiers/sim)
    if not me or len(me.polygons) == 0:
        if me: obj_eval.to_mesh_clear()
        return

    # Area-weighted surface COM (approx via triangulated faces)
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

    # Shift mesh data so COM becomes the origin.
    # Move geometry in object space, and move object back to keep world pose.
    M = obj.matrix_world
    obj.data.transform(mathutils.Matrix.Translation(-com))
    # Adjust object location by the world delta of that object-space shift
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

    fbx_filepath = r"/home/tem/Waste-Dataset-Generation/fbx_standard/cans/can_standard.fbx" 
    vertical_squash_probability = 0.3
    two_peaks_vertical_squash_probability = 0.3


    clear_scene()

    can_obj = import_soda_can(fbx_filepath)

    bpy.context.view_layer.objects.active = can_obj
    can_obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    can_obj.select_set(False)

    # rotate the can around z in random angle to change the texture direction
    angle = np.random.uniform(0, 360)
    rotate_object_vertices_z(can_obj, angle)

    if np.random.rand() < vertical_squash_probability:
        print("Applying vertical squash")

        # Pin group on straight can
        lower_bound = np.random.uniform(0.1, 0.25)
        upper_bound = np.random.uniform(0.65, 0.85)
        print(f"Lower bound: {lower_bound}, upper bound: {upper_bound}")
        pin_group = create_pin_group_top_bottom(can_obj, lower_bound, upper_bound)

        # Cloth effect simulation
        pressure_force = np.random.uniform(-300, -50)
        shrink_factor = np.random.uniform(-0.5, -0.02)
        print(f"Pressure force: {pressure_force}, shrink factor: {shrink_factor}")
        setup_cloth_sim(can_obj, pin_group, pressure_force, shrink_factor)

        # Compress along Z
        z_scale = np.random.uniform(0.1, 0.5)
        compress_can_along_z_lattice(can_obj, scale_factor=z_scale, anchor_mode='BOTTOM')

    else:
        print("Applying horizontal squash") 

        # press can middle inward
        band_half_size = np.random.uniform(0.1, 0.3)
        scale_factor_y = np.random.uniform(0.7, 1.0)
        center_offset = np.random.uniform(-0.1, 0.1)
        press_can_middle_inward(can_obj, band_half_size, scale_factor_y, center_offset)

        # Pin group on straight can
        lower_bound = np.random.uniform(0.1, 0.25)
        upper_bound = np.random.uniform(0.65, 0.85)
        print(f"Lower bound: {lower_bound}, upper bound: {upper_bound}")
        pin_group = create_pin_group_top_bottom(can_obj, lower_bound, upper_bound)

        # Cloth effect simulation
        pressure_force = np.random.uniform(-300, -50)
        shrink_factor = np.random.uniform(-0.5, -0.02)
        print(f"Pressure force: {pressure_force}, shrink factor: {shrink_factor}")
        setup_cloth_sim(can_obj, pin_group, pressure_force, shrink_factor)

        # Bend
        # Create curve & add Curve modifier        
        if np.random.rand() < two_peaks_vertical_squash_probability:
            height_factor = np.random.uniform(0.3, 0.6)
            end_offset_factor_y = np.random.uniform(-0.15, 0.15)
            curve_obj = create_vertical_bend_curve_2_peaks(can_obj, height_factor=height_factor, end_offset_factor_y=end_offset_factor_y, points_per_arc=64, smooth=True)
        else:
            bend_offset_factor_x = np.random.uniform(0, 0)
            bend_offset_factor_y = np.random.uniform(0, 1)
            print(f"Bend offset factor X: {bend_offset_factor_x}, Y: {bend_offset_factor_y}")
            curve_obj = create_vertical_bend_curve(can_obj, bend_offset_factor_x, bend_offset_factor_y)
        add_curve_modifier_after_cloth(can_obj, curve_obj)

        # Compress along Y
        y_scale = np.random.uniform(0.1, 0.5)
        compress_can_along_y_lattice(can_obj, scale_factor=y_scale, keep_center=True)

    # Set timeline for play the simulation
    bpy.context.scene.frame_set(5)
    log("Timeline set to frame 5")

    # --- Set origin to Center of Mass (Surface) and export selection to FBX ---
    # Select only the result can object
    bpy.ops.object.select_all(action='DESELECT')
    can_obj.select_set(True)
    bpy.context.view_layer.objects.active = can_obj

    # Apply transformations
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Set origin to Center of Mass (Surface)
    set_origin_to_evaluated_surface_com(can_obj)

    # Select only the result can object
    bpy.ops.object.select_all(action='DESELECT')
    can_obj.select_set(True)
    bpy.context.view_layer.objects.active = can_obj

    # FBX export path (edit this to your target)
    export_path = r"/home/tem/Waste-Dataset-Generation/res_fbx_objects/can_crushed.fbx"

    # Export only the selected object with embedded textures and copy path mode
    bpy.ops.export_scene.fbx(
        filepath=export_path,
        use_selection=True,     # Limit to selected objects
        path_mode='COPY',       # Path mode: Copy
        embed_textures=True     # Embed textures
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")



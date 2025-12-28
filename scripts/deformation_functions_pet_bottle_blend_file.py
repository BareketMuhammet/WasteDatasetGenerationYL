
import bpy  # type: ignore
import math
import numpy as np
from mathutils import Vector, Matrix
import mathutils
import os
import glob
from pathlib import Path


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


def create_label_material(label_image_path):
    """
    Create a material that shows the bottle label texture.
    Uses Generated coordinates so it will always map around the bottle.
    """

    scale = 1
    tex_scale=(scale, scale)      # (X, Y) scale of the label texture
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
    tex.image = bpy.data.images.load(label_image_path, check_existing=True)

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


def assign_label_to_middle_band(obj, label_mat,
                                z_low_frac,
                                z_high_frac):
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
    principled_bsdf_node.inputs["IOR"].default_value = 2

    transparent_material.blend_method = blend_mode
    transparent_material.shadow_method = 'HASHED'
    transparent_material.use_screen_refraction = True
    transparent_material.refraction_depth = 0.05
    

    # ⬇️ IMPORTANT: do NOT clear existing materials – keep the label slot!
    if obj.data.materials:
        obj.data.materials[0] = transparent_material   # plastic in slot 0
    else:
        obj.data.materials.append(transparent_material)


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

def list_fbx_files(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.fbx"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .fbx files found under: {input_dir}")
    return files

def list_png_files(label_image_dir):
    label_image_paths = sorted(
        os.path.join(label_image_dir, name)
        for name in os.listdir(label_image_dir)
        if name.lower().endswith(".png")
    )
    if not label_image_paths:
        raise FileNotFoundError(f"No .png label images found in '{label_image_dir}'.")
    return label_image_paths

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def make_name(prefix, idx):
    return f"{prefix}_{idx:06d}"

def deform_save_pet_bottle(source_fbx_path, label_image_path, save_path, object_name, blend_mode, roughness, transmission, alpha, color_code, flatten_probability = 0.5, press_band_inward_twice_probability = 0.3, two_peaks_vertical_squash_probability = 0.3, label_probability = 0.5):

    clear_scene()

    bottle_obj = import_water_bottle(source_fbx_path)

    bpy.context.view_layer.objects.active = bottle_obj
    bottle_obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bottle_obj.select_set(False)

    if np.random.rand() < label_probability:
        # Add a label
        label_z_low_frac = np.random.uniform(0.25, 0.35)
        label_z_high_frac = np.random.uniform(0.55, 0.7)
        label_mat = create_label_material(label_image_path)
        assign_label_to_middle_band(bottle_obj, label_mat, label_z_low_frac, label_z_high_frac)    

    # Randomize label orientation
    angle = np.random.uniform(0, 360)
    #rotate_object_vertices_z(bottle_obj, angle) # this doesn't work
    bottle_obj.rotation_euler.z += math.radians(angle)

    # 1) Pre-crease: a strong band dent to hint a fold
    band_half_size = np.random.uniform(0.2, 0.5)
    scale_factor_y = np.random.uniform(0.1, 0.4)  # strong inward
    center_offset = np.random.uniform(-0.25, 0.25)  # fold not exactly center
    press_band_inward(bottle_obj, band_half_size, scale_factor_y, center_offset)

    if np.random.rand() < press_band_inward_twice_probability:
        band_half_size = np.random.uniform(0.2, 0.5)
        scale_factor_y = np.random.uniform(0.1, 0.4)
        center_offset = np.random.uniform(-0.25, 0.25)
        press_band_inward(bottle_obj, band_half_size, scale_factor_y, center_offset)

    # 2) Cloth "vacuum" collapse, middle free, top+bottom lightly pinned
    lower_bound = np.random.uniform(0.05, 0.1)
    upper_bound = np.random.uniform(0.75, 0.9)
    pin_group = create_pin_group_top_bottom(bottle_obj, lower_bound, upper_bound)

    pressure_force = np.random.uniform(-300, -50)   # strong vacuum
    shrink_factor = np.random.uniform(-0.4, -0.02)  # contract bottle surface
    setup_cloth_sim(bottle_obj, pin_group, pressure_force, shrink_factor)

    # 3) Bend bottle with a curve (big folded arc)
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

    add_curve_modifier_after_cloth(bottle_obj, curve_obj)

    if np.random.rand() < flatten_probability:
        y_scale = np.random.uniform(0.1, 0.5)
        flatten_bottle_y_lattice(bottle_obj, scale_factor=y_scale, keep_center=True, points_v=4)

    z_scale = np.random.uniform(0.35, 1)
    squash_bottle_z_lattice(bottle_obj, scale_factor=z_scale, anchor_mode='BOTTOM', points_w=6)

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
        blend_mode=blend_mode,
        roughness=roughness,
        transmission=transmission,
        alpha=alpha,
        color_code=color_code
    )

    bpy.ops.object.select_all(action='DESELECT')
    bottle_obj.select_set(True)
    bpy.context.view_layer.objects.active = bottle_obj

    frozen_bottle = make_frozen_copy(bottle_obj)
    # frozen_bottle.name = "DeformedBottle"     
    frozen_bottle.name = object_name

    # Save as .blend to preserve materials, modifiers, etc.
    log(f"Saving .blend to: {save_path}")
    save_object_to_blend(frozen_bottle, save_path)
    log("Done.")


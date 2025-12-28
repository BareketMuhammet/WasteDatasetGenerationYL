import os
import random
import bpy # type: ignore
import bmesh # type: ignore
from mathutils import Vector  # type: ignore
from typing import Literal
import numpy as np
import math


def set_scene_settings(
    scene: bpy.types.Scene,
    render_engine: str,
    cycles_samples: int,
    frame_width: int,
    frame_height: int) -> None:

    scene.render.engine = render_engine
    scene.cycles.samples = cycles_samples # Set the number of samples for Cycles rendering

    scene.cycles.use_denoising = True # Enable denoising
    scene.render.film_transparent = True # Enable transparent background
    scene.render.resolution_x = frame_width
    scene.render.resolution_y = frame_height
    scene.cursor.location = Vector((0, 0, 0)) # Set the cursor location to the center of the scene
    scene.view_layers["ViewLayer"].use_pass_z = True # This is important for depth pass
    scene.view_layers["ViewLayer"].use_pass_object_index = True # This is important for object masking
    #scene.tool_settings.unified_paint_settings.use_unified_size = False # This is important for sculpting for brush size
    scene.gravity[2] = -9.8 # Set gravity to -9.8 m/s^2 (default value in Blender)
    scene.use_nodes = True # Enable nodes for the scene
    bpy.context.window.scene = scene # Set the current scene to the new scene

def set_render_device(
    scene: bpy.types.Scene,
    device: Literal["CPU", "GPU"],
    
) -> None:
    """
    Set the render device for the scene
    
    Parameters
    ----------
    scene : bpy.types.Scene
        The scene object
    device : Literal["CPU", "GPU"]
        The name of the device to use (e.g. CPU, GPU)
    """

    print(f"Current render engine is {scene.render.engine}")
    if scene.render.engine != "CYCLES":
        print("Render engine is not CYCLES, setting it to CYCLES")
        scene.render.engine = "CYCLES"
    else:
        print("Render engine is already CYCLES, doing nothing")

    if device == "CPU":
        print("Setting cycles device to CPU")
        scene.cycles.device = 'CPU'
    elif device == "GPU":
        print("Setting cycles device to GPU")
        scene.cycles.device = 'GPU'
    else:
        print(f"Device {device} is not supported, setting it to GPU")
        scene.cycles.device = 'GPU'

    print("Enabling all available devices")
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        print(f"Device {device.name} is currently {'enabled' if device.use else 'disabled'}")
        device.use = True
        print(f"Device {device.name} enabled")

def delete_all_nodes(scene):
    nodes = scene.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

def print_render_device(scene):
    print("Render Engine: ", scene.render.engine)
    print("Compute Device Type: ", bpy.context.preferences.addons['cycles'].preferences.compute_device_type)
    print("Cycles Device: ", scene.cycles.device)

def print_render_devices_info():
    for device in bpy.context.preferences.addons["cycles"].preferences.devices:
       if device["use"] == 1:
           in_use = 'IN USE'
       print("Device", device["name"], in_use)

def delete_all():
    # Clear objects
    bpy.ops.object.select_all(action='DESELECT')    
    bpy.ops.object.select_all(action='SELECT')    
    bpy.ops.object.delete()
    
    # Clear materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    # Clear textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)
    
    # Clear images
    for image in bpy.data.images:
        #bpy.data.images.remove(image)
        if image.name == "Render Result":
            image.user_clear()
        bpy.data.images.remove(image)
    
    # Clear meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    
    # Clear cameras
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)
    
    # Clear lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)
    
    # Clear node groups
    for node_group in bpy.data.node_groups:
        bpy.data.node_groups.remove(node_group)
    
    # Optionally clear collections
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
    
    # Refresh the scene
    bpy.context.view_layer.update()

def delete_all_objects(scene, without):
    for obj in scene.objects:
        if obj.type not in without and obj.name not in without:
            obj.select_set(True)
        else:
            obj.select_set(False)
    bpy.ops.object.delete()

def clear_cache():
    bpy.data.orphans_purge()

def add_object(
    type: str = "CUBE",
    name: str = "Cube",
    location: tuple = (0, 0, 0),
    rotation: tuple = (0, 0, 0),
    scale: tuple = (1, 1, 1)) -> bpy.types.Object:
    """
    Add a primitive object to the scene.

    Parameters
    ----------
    type : str, optional
        Type of the object, one of {'CUBE', 'SPHERE', 'CONE', 'CYLINDER', 'PLANE'} (default is 'CUBE')
    name : str, optional
        Name of the object (default is 'Cube')
    location : tuple, optional
        Location of the object (default is (0, 0, 0))
    rotation : tuple, optional
        Rotation of the object (default is (0, 0, 0))
    scale : tuple, optional
        Scale of the object (default is (1, 1, 1))

    Returns
    -------
    bpy.types.Object
        The created object
    """
    if type == "CUBE":
        bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation, scale=scale)
        obj = bpy.context.object
        obj.name = name
        return obj
    elif type == "SPHERE":
        bpy.ops.mesh.primitive_uv_sphere_add(location=location, rotation=rotation, scale=scale)
        obj = bpy.context.object
        obj.name = name
        return obj
    elif type == "CONE":
        bpy.ops.mesh.primitive_cone_add(location=location, rotation=rotation, scale=scale)
        obj = bpy.context.object
        obj.name = name
        return obj
    elif type == "CYLINDER":
        bpy.ops.mesh.primitive_cylinder_add(location=location, rotation=rotation, scale=scale)
        obj = bpy.context.object
        obj.name = name
        return obj
    elif type == "PLANE":
        bpy.ops.mesh.primitive_plane_add(location=location, rotation=rotation, scale=scale)
        obj = bpy.context.object
        obj.name = name
        return obj
    else:
        raise ValueError("Invalid object type")

def set_object_active(obj):
    
    if obj is None:
        print(f"Error: Object '{obj.name}' does not exist.")
        return
    
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    bpy.context.view_layer.objects.active = obj

    print(f"Object '{obj.name}' is now active.")
 
def create_camera(sensor_width,
                             sensor_height,
                             focal_length,
                             location=(0, 0, 0),
                             rotation=(0, 0, 0)
                             ) -> bpy.types.Camera:
    
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.object
    camera.name = "camera"
    camera.data.lens = focal_length
    camera.data.lens_unit = 'MILLIMETERS'
    camera.data.sensor_width = sensor_width
    camera.data.sensor_height = sensor_height
    camera.data.sensor_fit = 'AUTO'
    
    return camera

def add_light(  name, type = 'POINT', shape="", size=1, size_y=1, radius=0.5,
                location=(0, 0, 0),
                rotation=(0, 0, 0),
                color=(1, 1, 1),
                power=1000.0, diffuse=1, specular=1, volume=1):
    """
        type: String --> 'AREA', 'POINT', 'SUN', 'SPOT'
        shape: String --> 'SQUARE', 'RECTANGLE', 'DISK', 'ELLIPSE'
    """

    bpy.ops.object.light_add(type=type, location=location, rotation=rotation)
    bpy.context.object.data.name = name
    bpy.context.object.data.color = color #rgb
    bpy.context.object.data.energy = power
    bpy.context.object.data.diffuse_factor = diffuse
    bpy.context.object.data.specular_factor = specular
    bpy.context.object.data.volume_factor = volume

    if type == 'AREA':
        if shape == 'SQUARE':
            bpy.context.object.data.shape = 'SQUARE'
            bpy.context.object.data.size = size

        elif shape == 'RECTANGLE':
            bpy.context.object.data.shape = 'RECTANGLE'
            bpy.context.object.data.size = size
            bpy.context.object.data.size_y = size_y

        elif shape == 'DISK':
            bpy.context.object.data.shape = 'DISK'
            bpy.context.object.data.size = size

        else:
            bpy.context.object.data.shape = 'ELLIPSE'
            bpy.context.object.data.size = size
            bpy.context.object.data.size_y = size_y

    if type == 'POINT' or type == 'SPOT':
        bpy.context.object.data.shadow_soft_size = radius

def create_render_node(nodes, location_x=0, location_y=0):
    render_node = nodes.new(type="CompositorNodeRLayers")
    render_node.name = "RenderNode"
    render_node.location = (location_x, location_y)
    return render_node

def create_id_musk_node(nodes, name, index, location_x=0, location_y=0):
    id_mask_node = nodes.new(type="CompositorNodeIDMask")
    id_mask_node.name = name
    id_mask_node.location = (location_x, location_y)
    id_mask_node.index = index
    return id_mask_node

def create_output_node(nodes, name, file_format='PNG', color_depth='8', color_mode='RGB', location_x=600, location_y=0):
    file_output_node = nodes.new(type='CompositorNodeOutputFile')
    file_output_node.name = name
    file_output_node.location = (location_x, location_y)
    file_output_node.format.file_format = file_format
    file_output_node.format.color_depth = color_depth # bit depth
    file_output_node.format.color_mode = color_mode
    return file_output_node

def create_depth_output_node(nodes, name, file_format='OPEN_EXR', color_depth='32', exr_codec='NONE', location_x=600, location_y=200):
    depth_output_node = nodes.new(type='CompositorNodeOutputFile')
    depth_output_node.name = name
    depth_output_node.location = (location_x, location_y)
    depth_output_node.format.file_format = file_format
    depth_output_node.format.color_depth = color_depth # bit depth
    depth_output_node.format.exr_codec = exr_codec
    return depth_output_node

def print_node_info(node):
    print("Node Name:", node.name)
    print("Node Type:", node.type)
    print("Node Location:", node.location)
    print("-----------------------------------")
    
    print("Node Inputs:")
    for input in node.inputs:
        print("  Input Name:", input.name)
        print("  Input Type:", input.type)
        print("  Input Default Value:", input.default_value)
    print("-----------------------------------")
        
    print("Node Outputs:")
    for output in node.outputs:
        print("  Output Name:", output.name)
        print("  Output Type:", output.type)
        print("  Output Default Value:", output.default_value)
    print("-----------------------------------")

def delete_input_by_name(node, name):
    
    if name.lower() == "all":
        for input in node.inputs:
            node.inputs.remove(input)
    else:
        for input in node.inputs:
            if input.name == name:
                node.inputs.remove(input)
                break

def add_output_slot(node, name):
    """
    This Function Adds a new output slot to the File Output node
    """
    node.file_slots.new(name)

def link_nodes(links, from_node, from_node_output, to_node, to_node_input):
    links.new(from_node.outputs[from_node_output], to_node.inputs[to_node_input])

def add_texture(object, texture_image_path, scale):    
    material = bpy.data.materials.new(name="RepeatingTextureMaterial")
    material.use_nodes = True
    
    nodes = material.node_tree.nodes
    links = material.node_tree.links    
    
    for node in nodes:
        nodes.remove(node)
    
    # Add necessary nodes
    texture_coordinate_node = nodes.new(type='ShaderNodeTexCoord') # this node will be used to get the UV coordinates pattern
    tecture_mapping_node = nodes.new(type='ShaderNodeMapping') # this node will be used to control the texture scaling
    texture_image_node = nodes.new(type='ShaderNodeTexImage') # this node will be used to load the texture image and apply the scale
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled') # this node will be used to apply the render effects like the lighting
    output_node = nodes.new(type='ShaderNodeOutputMaterial') # output the material to the object 
    
    # Set the location of the nodes in the program view (this is not necessary)
    texture_coordinate_node.location = (-800, 400)
    tecture_mapping_node.location = (-600, 400)
    texture_image_node.location = (-400, 400)
    principled_bsdf.location = (-200, 400)
    output_node.location = (0, 400)
    
    # Link nodes
    links.new(texture_coordinate_node.outputs['UV'], tecture_mapping_node.inputs['Vector']) # we choose the UV coordinate
    links.new(tecture_mapping_node.outputs['Vector'], texture_image_node.inputs['Vector'])
    links.new(texture_image_node.outputs['Color'], principled_bsdf.inputs['Base Color'])
    links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Set the scale for repeating texture
    tecture_mapping_node.inputs['Scale'].default_value = scale

    # Set the image for the texture
    texture_image = bpy.data.images.load(texture_image_path)
    texture_image_node.image = texture_image
    
    # Assign the material to the object
    if object.data.materials:
        object.data.materials[0] = material
    else:
        object.data.materials.append(material)

def add_rigid_body(obj, type, collision_shape, collision_margin, mass, friction):
    set_object_active(obj)
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = type
    obj.rigid_body.collision_shape = collision_shape
    obj.rigid_body.collision_margin = collision_margin
    obj.rigid_body.mass = mass
    obj.rigid_body.friction = friction

def get_visible_area_dimensions(camera, image_width, image_height, distance):
    cam_data = camera.data
    fov_radians = cam_data.angle
    fov_degrees = math.degrees(fov_radians)
    aspect_ratio = image_width / image_height    
    if aspect_ratio >= 1.0:
        horizontal_fov = fov_degrees
        vertical_fov = 2 * math.degrees(math.atan(math.tan(fov_radians / 2) / aspect_ratio))
    else:
        vertical_fov = fov_degrees
        horizontal_fov = 2 * math.degrees(math.atan(math.tan(fov_radians / 2) * aspect_ratio))        
    visible_area_width = 2 * distance * math.tan(math.radians(horizontal_fov) / 2)
    visible_area_height = 2 * distance * math.tan(math.radians(vertical_fov) / 2)    
    return visible_area_width, visible_area_height

def add_working_area(camera, image_width, image_height, distance, background_image, scale):
    visible_area_width, visible_area_height = get_visible_area_dimensions(camera, image_width, image_height, distance)

    dimension_x = visible_area_width
    dimension_y = visible_area_height
    dimension_z = 2.0 * distance

    # Create a cube to represent the working area
    dimensions = (dimension_x, dimension_y, dimension_z)
    bpy.ops.mesh.primitive_cube_add(location = (0, 0, 0), rotation = (0, 0, 0))
    bpy.context.object.name = "working_area"
    bpy.context.object.dimensions = dimensions

    working_area = bpy.data.objects.get("working_area")
    add_texture(working_area, background_image, scale)

    return working_area

def change_mode(mode: str):
    
    if bpy.context.active_object is None:
        print("No active object selected.")
        return

    current_mode = bpy.context.object.mode

    if current_mode == mode:
        print(f"Already in {mode} mode.")
        return

    try:
        bpy.ops.object.mode_set(mode=mode.upper())
        print(f"Switched to {mode} mode for object: {bpy.context.object.name}")
    except RuntimeError as e:
        print(f"Error: Unable to switch to {mode} mode. Reason: {e}")

def select_verteces(selection_method, vertex, visited, connected_vertices):
    # selections_methods : "diagonal, circular, random"
    
    slope = 0.3
    threshold = 0.2
    
    
    if vertex in visited:
        return
    
    if len(visited) >= 100:
        return
    
    visited.add(vertex)
    connected_vertices.append(vertex.co)
    vertex.select = True
    
    for edge in vertex.link_edges:
        other_vert = edge.other_vert(vertex)
        select_verteces(other_vert, visited, connected_vertices)
     
def apply_sculpt_effect(obj):
    change_mode(obj, 'SCULPT')
    mesh = bmesh.from_edit_mesh(obj.data)
    mesh.verts.ensure_lookup_table()
    
    verts_size = len(mesh.verts)
    print(f"Number of vertices: {verts_size}")
    
    # Deselect all vertices to start fresh
    bpy.ops.mesh.select_all(action='DESELECT')
    
    start_vertex_index = random.randint(0, verts_size - 1)
    start_vertex = mesh.verts[start_vertex_index]
    
def add_object_from_fbx_file(file_path, location = (0,0,0), rotation = (0, 0, 0)):
    bpy.ops.import_scene.fbx(filepath = file_path)
    imported_objects = bpy.context.selected_objects
    for obj in imported_objects:
        bpy.context.view_layer.objects.active = obj
        obj.parent = None
        obj.constraints.clear()
        obj.location = location
        obj.rotation_euler = rotation
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.context.view_layer.update()
    return imported_objects

def make_object_transparent(obj, blend_mode, roughness, transmission, alpha, color_code):
    obj.data.materials.clear()
    transparent_material = bpy.data.materials.new(name="TransparentMaterial")

    transparent_material.use_nodes = True
    nodes = transparent_material.node_tree.nodes
    links = transparent_material.node_tree.links

    nodes.clear()

    # Create nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')

    links.new(principled_bsdf_node.outputs[0], output_node.inputs[0])

    principled_bsdf_node.inputs['Base Color'].default_value = color_code
    principled_bsdf_node.inputs['Roughness'].default_value = roughness
    principled_bsdf_node.inputs["Transmission"].default_value = transmission
    principled_bsdf_node.inputs['Alpha'].default_value = alpha

    principled_bsdf_node.inputs["Transmission Roughness"].default_value = 0.2 # this to make it plastic

    transparent_material.blend_method = blend_mode # 'OPAQUE', 'BLEND', 'CLIP', 'HASHED'
    #transparent_material.blend_method = 'BLEND'
    transparent_material.shadow_method = 'HASHED'
    transparent_material.use_screen_refraction = True
    transparent_material.refraction_depth = 0.05

    # principled_bsdf_node.inputs['Base Color'].default_value = color_code # Transmission color
    
    # principled_bsdf_node.inputs["Transmission"].default_value = 1

    # # control how the object is blended with the background (like the photo)
    # principled_bsdf_node.inputs['Alpha'].default_value = 1 

    # IOR (Index of Refraction)
    # How much rays bend entering/leaving the object.
    # Typical values: air ~1.0, water ~1.33, glass ~1.45–1.52, acrylic ~1.49.
    principled_bsdf_node.inputs["IOR"].default_value = 2

    #principled_bsdf_node.inputs["Subsurface"].default_value = 0.08

    # Transmission — “how much light goes through”
    # Transmission controls how transparent a material is due to refraction
    # 0.0	Light stops at the surface (opaque)	Looks like solid plastic or paint
    # 0.5	Half the light goes through	Semi-translucent plastic
    # 1.0	All light passes through	Clear glass, water, PET

    # Transmission Roughness — “how blurry the light looks inside”
    # Transmission Roughness controls how clear or cloudy that transmitted light is.
    # 0.0	Light passes perfectly	Clear glass, crystal
    # 0.1–0.3	Light is slightly diffused	Soft, hazy transparency (frosted glass, plastic)
    # > 0.5	Strong internal blur	Milky, semi-opaque material (cloudy plastic, wax)

    # Roughness
    # Micro-surface roughness for both reflection and refraction.
    # 0.0 = perfectly clear, sharp refraction.
    # 0.1–0.3 = frosted glass (blurry).
    # principled_bsdf_node.inputs['Roughness'].default_value = 0.15

    # Specular controls the strength of mirror-like reflections 
    # from direct light sources, the bright highlights you see on a surface when light hits it.
    # principled_bsdf_node.inputs["Specular"].default_value = 0.5    

    # Emission
    # This input gives the material its own light output , as if the object is glowing.
    # for example:
    # principled_bsdf_node.inputs["Emission"].default_value = (1.0, 1.0, 1.0, 1.0)
    # principled_bsdf_node.inputs["Emission Strength"].default_value = 1
    # will make the object glow like a light source,

    # Thin translucent plastic look
    # principled_bsdf_node.inputs["Base Color"].default_value = (1, 1, 1, 1)  # light off-white
    # principled_bsdf_node.inputs["Alpha"].default_value = 0.8      # semi-transparent
    # principled_bsdf_node.inputs["Roughness"].default_value = 0.4   # soft reflection
    # principled_bsdf_node.inputs["Specular"].default_value = 0.5    # moderate shine
    # principled_bsdf_node.inputs["Transmission"].default_value = 0.96  # not refractive
    #principled_bsdf_node.inputs["IOR"].default_value = 2.5

    # # Optional subtle subsurface scattering (adds light diffusion)
    # principled_bsdf_node.inputs["Subsurface"].default_value = 0.05
    # principled_bsdf_node.inputs["Subsurface Color"].default_value = (1.0, 0.98, 0.95, 1.0)

    # # --- PET realistic settings ---
    # principled_bsdf_node.inputs["Base Color"].default_value = (0.9, 0.95, 1.0, 1.0)
    # principled_bsdf_node.inputs["Metallic"].default_value = 0.0
    # principled_bsdf_node.inputs["Specular"].default_value = 0.55
    # principled_bsdf_node.inputs["Roughness"].default_value = 0.25
    # principled_bsdf_node.inputs["IOR"].default_value = 1.45
    # principled_bsdf_node.inputs["Transmission"].default_value = 1
    # principled_bsdf_node.inputs["Transmission Roughness"].default_value = 0.2
    # principled_bsdf_node.inputs["Alpha"].default_value = 1.0
    # principled_bsdf_node.inputs["Emission"].default_value = (0.0, 0.0, 0.0, 1.0)
    # principled_bsdf_node.inputs["Emission Strength"].default_value = 0.0
    # transparent_material.blend_method = 'BLEND'
    # transparent_material.shadow_method = 'HASHED'
    # transparent_material.use_screen_refraction = True
    # transparent_material.refraction_depth = 0.05



    obj.data.materials.append(transparent_material)
        
        
def remove_transparency(obj):
    if not obj.data or not hasattr(obj.data, "materials"):
        return
    
    for material in obj.data.materials:
        if material is None:
            continue

        material.blend_method = 'OPAQUE'
        material.shadow_method = 'OPAQUE'

        if not material.use_nodes:
            continue

        nodes = material.node_tree.nodes
        principled = None

        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
                break

        if principled:
            # Set Alpha to 1
            if "Alpha" in principled.inputs:
                principled.inputs["Alpha"].default_value = 1.0
            # Set Transmission to 0
            if "Transmission" in principled.inputs:
                principled.inputs["Transmission"].default_value = 0.0


def set_color_management(
    *,
    display_device: str = "sRGB",
    view_transform: str = "Standard",   # e.g. "Standard", "Filmic", "Filmic Log" (if available)
    look: str = "None",                 # e.g. "None", "Base Contrast", "Medium High Contrast"
    exposure: float = 0.0,              # in stops
    gamma: float = 1.0,                 # 1.0 = neutral
    use_curve_mapping: bool = False,    # enable per-view curve mapping
    s_curve_strength: float = 0.0,      # 0.0 disabled; 0.1–0.3 = subtle S-curve
    verbose: bool = False               # print fallbacks/warnings
):
    """
    Adjust Blender Color Management for camera-like output or custom grading.

    Returns
    -------
    dict : previous settings so you can restore later with `restore_color_management(prev)`.

    Notes
    -----
    - 'view_transform' and 'look' must exist in the current OCIO config; the function will fall
      back to a safe option if not found.
    - If s_curve_strength > 0, a gentle S-curve is applied to the view curve mapping.
    """
    scene = bpy.context.scene
    vs = scene.view_settings
    ds = scene.display_settings

    # --- capture previous settings for restoration ---
    prev = {
        "display_device": ds.display_device,
        "view_transform": vs.view_transform,
        "look": vs.look,
        "exposure": vs.exposure,
        "gamma": vs.gamma,
        "use_curve_mapping": vs.use_curve_mapping,
        "curve_points": None,
    }

    # Store existing curve points if curve mapping is enabled
    if vs.use_curve_mapping:
        cm = vs.curve_mapping
        c = cm.curves[3]  # Combined
        prev["curve_points"] = [(p.location[0], p.location[1]) for p in c.points]

    # --- apply display device ---
    try:
        ds.display_device = display_device
    except TypeError:
        if verbose:
            print(f"[ColorMgmt] Display device '{display_device}' unavailable. Keeping '{ds.display_device}'.")

    # --- apply view transform with fallback ---
    available_views = getattr(vs, "bl_rna", None)
    ok = True
    try:
        vs.view_transform = view_transform
    except TypeError:
        ok = False
    except Exception:
        ok = False
    if not ok:
        if verbose:
            print(f"[ColorMgmt] View transform '{view_transform}' unavailable. Falling back to 'Standard' or 'Filmic'.")
        try:
            vs.view_transform = "Standard"
        except Exception:
            vs.view_transform = "Filmic"

    # --- apply look with fallback ---
    ok = True
    try:
        vs.look = look
    except TypeError:
        ok = False
    except Exception:
        ok = False
    if not ok:
        if verbose:
            print(f"[ColorMgmt] Look '{look}' unavailable. Using 'None'.")
        vs.look = "None"

    # --- exposure & gamma ---
    vs.exposure = float(exposure)
    vs.gamma = float(gamma)

    # --- optional view S-curve ---
    vs.use_curve_mapping = bool(use_curve_mapping or (s_curve_strength > 0.0))
    if vs.use_curve_mapping:
        cm = vs.curve_mapping
        cm.initialize()  # reset to default
        # Gentle S-curve on Combined channel; strength scales how far midpoints push
        c = cm.curves[3]  # Combined
        # Remove any intermediate points
        while len(c.points) > 2:
            c.points.remove(c.points[1])

        # Base positions for a mild S-curve
        mid_low_x, mid_high_x = 0.25, 0.75
        # Scale offsets by s_curve_strength
        # Default mild S at strength=0.2 -> approx 0.25->0.23 and 0.75->0.77
        low_y  = 0.25 - (0.10 * s_curve_strength)
        high_y = 0.75 + (0.10 * s_curve_strength)

        c.points.new(mid_low_x, low_y)
        c.points.new(mid_high_x, high_y)
        cm.update()

    if verbose:
        print("[ColorMgmt] Applied:",
              f"display={ds.display_device}, view={vs.view_transform}, look={vs.look}, "
              f"exposure={vs.exposure}, gamma={vs.gamma}, curves={vs.use_curve_mapping}")

    return prev


def restore_color_management(prev: dict):
    """Restore settings captured by set_color_management()."""
    scene = bpy.context.scene
    vs = scene.view_settings
    ds = scene.display_settings

    ds.display_device = prev.get("display_device", ds.display_device)

    # Restore view and look with fallback guards
    for attr in ("view_transform", "look"):
        val = prev.get(attr, getattr(vs, attr))
        try:
            setattr(vs, attr, val)
        except Exception:
            # fall back silently
            pass

    vs.exposure = prev.get("exposure", vs.exposure)
    vs.gamma = prev.get("gamma", vs.gamma)

    # Restore curves if present
    vs.use_curve_mapping = prev.get("use_curve_mapping", False)
    if vs.use_curve_mapping and prev.get("curve_points") is not None:
        cm = vs.curve_mapping
        cm.initialize()
        c = cm.curves[3]
        while len(c.points) > 2:
            c.points.remove(c.points[1])
        # Rebuild prior points
        for x, y in prev["curve_points"]:
            # Skip endpoints (0,0) and (1,1), which exist by default
            if (x, y) not in [(0.0, 0.0), (1.0, 1.0)]:
                c.points.new(x, y)
        cm.update()

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
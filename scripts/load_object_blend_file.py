

import bpy  # type: ignore

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

object_name = "pet_0001"
append_single_object(
        "/home/tem/Waste-Dataset-Generation/res_fbx_objects/pet_blend/" + object_name + ".blend",
        object_name
    )
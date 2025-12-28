import os
import OpenEXR # type: ignore
import Imath  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

def read_exr_file(file_path, channel = "depth.V"):
    exr_file = OpenEXR.InputFile(file_path)    
    header = exr_file.header()
    if channel not in header['channels']:
        print("The EXR file does not contain a " + channel + " channel.")
        return np.zeros((1, 1))   
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1    
    pt = Imath.PixelType(Imath.PixelType.FLOAT)    
    channel_data = exr_file.channel(channel, pt)    
    data_array = np.frombuffer(channel_data, dtype=np.float32)
    data_array.shape = (height, width)
    data = data_array.copy()
    exr_file.close()    
    return data

def list_exr_file_channels(exr_file):
    exr_file = OpenEXR.InputFile(exr_file)
    header = exr_file.header()
    channels = header['channels'].keys()
    exr_file.close()
    return list(channels)



def convert_exr_to_npz(exr_path: str, npz_path: str) -> bool:
    """Convert an EXR image into a compressed NumPy archive."""
    try:
        exr_data = read_exr_file(exr_path)
        if exr_data is None:
            print(f"Failed to load EXR image from {exr_path}")
            return False
        np.savez_compressed(npz_path, exr_data)
        print(f"Converted {exr_path} to {npz_path} successfully.")
        return True
    except Exception as exc:
        print(f"An error occurred while converting EXR to NPZ: {exc}")
        return False
















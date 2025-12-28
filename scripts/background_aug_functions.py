
import cv2
import numpy as np


def to_u8(x): return np.clip(x, 0, 255).astype(np.uint8)

def noisy_mask(shape, min_scale=5, max_scale=100, thresh=0.5, blur=5, debug=False):
    scale = np.random.randint(min_scale, max_scale)
    if debug:
        print("creating noisy mask")
        print("scale", scale)
    noise = cv2.resize(np.random.rand(shape[0]//scale, shape[1]//scale).astype(np.float32),
                      (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0,0), blur)
    return (noise > thresh).astype(np.float32)

def oil_effect(img, strength=0.45, noisy_mask_scale=[5, 100]):
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    mask = noisy_mask((h,w), min_scale=noisy_mask_scale[0], max_scale=noisy_mask_scale[1])
    mask = cv2.GaussianBlur(mask, (0,0), 9)
    darkened = out * (1 - np.dstack([mask]*3)*strength)
    dist = cv2.distanceTransform((mask*255).astype(np.uint8), cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist/dist.max()
    spec = cv2.GaussianBlur(dist, (0,0), 9) * 35
    darkened += np.dstack([spec]*3)
    return to_u8(darkened)

def dust_effect(img, amount=0.85):
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    p = max(0.0005, min(0.02, 0.001 * amount * 1000))
    specks = (np.random.rand(h,w) > (1.0 - p)).astype(np.float32)
    specks = cv2.GaussianBlur(specks, (0,0), 0.8)
    specks = cv2.normalize(specks, None, 0, 80*amount, cv2.NORM_MINMAX)
    out += np.dstack([specks]*3)
    out = 15 + 0.85*(out - 15)
    return to_u8(out)

def blisters_effect(img, amount=1, count=18, rmin=15, rmax=45):
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    bump = np.zeros((h,w), np.float32)
    for _ in range(count):
        c = (np.random.randint(int(w*0.1), int(w*0.9)), np.random.randint(int(h*0.1), int(h*0.9)))
        r = np.random.randint(rmin, rmax)
        tmp = np.zeros((h,w), np.float32)
        cv2.circle(tmp, c, r, 1.0, -1)
        tmp = cv2.GaussianBlur(tmp, (0,0), r/2)
        bump += tmp
    bump = bump / (bump.max()+1e-6)
    light = cv2.GaussianBlur(bump, (0,0), 8)
    shadow = cv2.GaussianBlur(np.roll(np.roll(bump, 3, axis=0), -3, axis=1), (0,0), 8)
    factor = amount * 60
    out += np.dstack([light*factor - shadow*factor]*3)
    return to_u8(out)

def rotate_and_center_crop(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate image by angle and center-crop back to original size to avoid padded borders."""
    if not np.isfinite(angle_degrees) or abs(angle_degrees) < 1e-3:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])

    new_w = int(np.ceil((h * sin_val) + (w * cos_val)))
    new_h = int(np.ceil((h * cos_val) + (w * sin_val)))

    rotation_matrix[0, 2] += (new_w / 2.0) - center[0]
    rotation_matrix[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    start_x = int(round((new_w - w) / 2.0))
    start_y = int(round((new_h - h) / 2.0))
    end_x = start_x + w
    end_y = start_y + h

    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(new_w, end_x)
    end_y = min(new_h, end_y)

    cropped = rotated[start_y:end_y, start_x:end_x]

    if cropped.shape[0] != h or cropped.shape[1] != w:
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

    return cropped
# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def translate_image(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    h, w = img.shape
    translated = np.zeros_like(img)
    translated[shift_y:h, shift_x:w] = img[0:h-shift_y, 0:w-shift_x]
    return translated

def rotate_90_clockwise(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=-1)

def stretch_image(img: np.ndarray, scale_x: float) -> np.ndarray:
    h, w = img.shape
    new_w = int(w * scale_x)
    stretched = np.zeros((h, new_w))
    x_indices = (np.linspace(0, w - 1, new_w)).astype(int)
    stretched[:, :] = img[:, x_indices]
    return stretched

def mirror_image(img: np.ndarray) -> np.ndarray:
    return np.fliplr(img)

def barrel_distortion(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    distorted = np.zeros_like(img)
    cx, cy = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            factor = 1 + 0.0005 * (r ** 2)
            new_x = int(cx + (x - cx) * factor)
            new_y = int(cy + (y - cy) * factor)
            if 0 <= new_x < w and 0 <= new_y < h:
                distorted[y, x] = img[new_y, new_x]
    return distorted

def apply_geometric_transformations(img: np.ndarray) -> dict:
    return {
        "translated": translate_image(img, shift_x=10, shift_y=10),
        "rotated": rotate_90_clockwise(img),
        "stretched": stretch_image(img, scale_x=1.5),
        "mirrored": mirror_image(img),
        "distorted": barrel_distortion(img)
    }
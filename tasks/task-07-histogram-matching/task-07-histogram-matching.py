# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched_img = np.zeros_like(source_img)
    
    for channel in range(3):
        source_channel = source_img[:, :, channel]
        reference_channel = reference_img[:, :, channel]
        
        source_hist = np.histogram(source_channel, bins=256, range=(0, 256))[0]
        reference_hist = np.histogram(reference_channel, bins=256, range=(0, 256))[0]

        source_cdf = np.cumsum(source_hist) / np.sum(source_hist)
        reference_cdf = np.cumsum(reference_hist) / np.sum(reference_hist)

        transform_map = np.zeros(256)
        for j in range(256):
            diff = np.abs(source_cdf[j] - reference_cdf[:])
            index = diff.argmin()
            transform_map[j] = index

        matched_channel = transform_map[source_channel].astype(np.uint8)
        matched_img[:, :, 2-channel] = matched_channel
        
    return matched_img
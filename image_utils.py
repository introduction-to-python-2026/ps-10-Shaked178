import numpy as np
from PIL import Image
from scipy.signal import convolve2d # Ensure this is imported for edge_detection

def load_image(image_path):
    # Reads an image from the given path and returns it as a NumPy array.
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array

def edge_detection(image):
    # Converts a color image to grayscale and applies Sobel filters for edge detection.
    # Handles RGBA, RGB, and already grayscale images.

    # Grayscale conversion
    if image.ndim == 3 and image.shape[-1] == 4:
        # Assuming RGBA, taking RGB channels
        gray_image = np.mean(image[:, :, :3], axis=2)
    elif image.ndim == 3 and image.shape[-1] == 3:
        # Assuming RGB
        gray_image = np.mean(image, axis=2)
    elif image.ndim == 2:
        # Already grayscale
        gray_image = image
    else:
        raise ValueError("Unsupported image array format for grayscale conversion.")

    # Sobel kernels
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Convolution
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='symm')
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='symm')

    # Edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG

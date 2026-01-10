import numpy as np
from PIL import Image
from scipy.signal import convolve2d 



def load_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array

#נותנת לנו את מערך הנתונים

def edge_detection(image):
    if image.ndim == 3 and image.shape[-1] == 4:
        gray_image = np.mean(image[:, :, :3], axis=2)
    elif image.ndim == 3 and image.shape[-1] == 3:
        gray_image = np.mean(image, axis=2)
    elif image.ndim == 2:
        gray_image = image
    else:
        raise ValueError("Unsupported image array format for grayscale conversion.")


    
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

    
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='symm')
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='symm')

    
#יוצרת לנו תמונה אבל רק של הקצוות    
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG

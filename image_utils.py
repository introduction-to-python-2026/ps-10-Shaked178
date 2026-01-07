from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def load_image(image_path):
     img = Image.open(image_path)
     return np.array(img)


image_array = load_image('/content/MIKIֹֹ10.png')
print(f"Image shape: {image_array.shape}")
print(f"Image data type: {image_array.dtype}")


def edge_detection(image):
    if image_array.shape[-1] == 4:
      gray_image = np.mean(image_array[:, :, :3], axis=2)
  else:
      gray_image = np.mean(image_array, axis=2)

  
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

  
  edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

  return edgeMAG


edge_magnitude_image = edge_detection(image_array)

print(edge_magnitude_image)

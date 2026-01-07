from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def load_image(image_path):
   MIKI_path = "/content/MIKIֹֹ10.png"
   MIKI = Image.open(MIKI_path)
   MIKI = np.array(MIKI)
print("Image shape:", MIKI.shape )
print("Image data type:", MIKI.dtype)




def edge_detection(image):
    if image.ndim == 3 and image.shape[-1] == 4:
      gray_image = np.mean(image[:, :, :3], axis=2)
    elif image.ndim == 3 and image.shape[-1] == 3:
      gray_image = np.mean(image, axis=2)
    elif image.ndim == 2: # Already grayscale
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


    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG

edge_magnitude_image = edge_detection(MIKI)

plt.figure(figsize=(8, 6))
plt.imshow(edge_magnitude_image, cmap='gray')
plt.title('Edge Detected Miki')
plt.axis('off')
plt.show()

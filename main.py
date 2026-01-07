from image_utils import load_image
from image_utils import edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt


claen_miki = median(MIKI, ball(3))

clean_gray_miki = edge_detection(claen_miki)


plt.figure(figsize=(8, 6))
plt.imshow(clean_gray_miki, cmap='gray') 
plt.title('Clean Edge Detected Miki')
plt.axis('off')
plt.show()

plt.figure(figsize=(6,4))
plt.hist(clean_gray_miki.ravel(), bins=256)
plt.title("Histogram of clean_gray_miki")
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.show()

threshold = 120
binary_miki = (clean_gray_miki > threshold).astype(int)

plt.imshow(binary_miki, cmap='gray')
plt.title("Binary Edge Miki")
plt.axis('off')
plt.show()

from PIL import Image

Image.fromarray((binary_miki * 255).astype('uint8')).save("binary_miki.png")

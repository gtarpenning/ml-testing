from PIL import Image
from resizeimage import resizeimage
import numpy as np
from matplotlib import pyplot as plt

FILENAME = './data/pink-shirt.jpg'
img = Image.open(FILENAME).convert('L')  # convert image to 8-bit grayscale
img = resizeimage.resize_cover(img, [28, 28])
WIDTH, HEIGHT = img.size
print(WIDTH, HEIGHT)

data = list(img.getdata()) # convert image data to a list of integers
# convert that to 2D list (list of lists of integers)
data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
plt.imshow(data, interpolation='nearest')
plt.show()

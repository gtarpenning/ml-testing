from PIL import Image
from resizeimage import resizeimage
import requests
from io import BytesIO
import numpy as np


class MakeMnist(object):
    def __init__(self, linkArray):
        self.links = linkArray

    def get_pixels(self, img):
        img = resizeimage.resize_cover(img, [28, 28])
        WIDTH, HEIGHT = img.size
        data = list(img.getdata())
        data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
        return np.array(data)

    def get_img(self, link):
        r = requests.get(link)
        i = Image.open(BytesIO(r.content)).convert('L')
        return(i)

    def display_img(self, array):
        from matplotlib import pyplot as plt
        plt.imshow(array, interpolation='nearest')
        plt.show()


def main():
    linkArray = [
        'https://assets.academy.com/mgen/95/10787095.jpg'
    ]
    mnistData = []
    database = MakeMnist(linkArray)
    for link in database.links:
        img = database.get_img(link)
        mnistData.append(database.get_pixels(img))

    database.display_img(mnistData[0])

if __name__ == '__main__':
    main()

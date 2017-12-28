from PIL import Image
from resizeimage import resizeimage
import requests
from io import BytesIO
import numpy as np
from bs4 import BeautifulSoup as bs
import csv


class MakeMnist(object):
    def __init__(self):
        self.categories = {
            '0': 'T-shirt',
            '1': 'Trouser',
            '2': 'Pullover',
            '3': 'Dress',
            '4': 'Coat',
            '5': 'Sandal',
            '6': 'Shirt',
            '7': 'Sneaker',
            '8': 'Bag',
            '9': 'Ankle boot'
        }
        self.links = []
        self.final_data = []

    def get_pixels(self, link):
        img = resizeimage.resize_cover(self.get_img(link), [28, 28])
        WIDTH, HEIGHT = img.size
        data = list(img.getdata())
        data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
        return np.array(data)

    def get_img(self, link):
        r = requests.get(link)
        i = Image.open(BytesIO(r.content)).convert('L')
        return i

    def display_img(self, array):
        from matplotlib import pyplot as plt
        plt.imshow(array, interpolation='nearest')
        plt.show()

    def _get_soup(self, url):
        return bs(requests.get(url).text,'html.parser')

    def get_link_array(self):
        for query in self.categories:
            print(query, self.categories[query])
            soup = self._get_soup('https://www.google.com/search?q=' + self.categories[query] +
                    '&source=lnms&tbm=isch&sa=X')
            for img in soup.findAll('img'):
                self.links.append([query, img.get('src')])

    def create_csv(self, filename, data):
        csvFile = open(filename, 'w')
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)

    def format_data(self, cat, data):
        """ format data as rows x 784 """
        r = []
        r.append(cat)
        for row in data:
            for p in row:
                r.append(p)
        return r

    def load_csv(self, filename):
        with open(filename, newline='') as csvFile:
            reader = csv.reader(csvFile)
            data = []
            for row in reader:
                data.append(row)

            print(len(data), len(data[0]))

    def make_dataset(self):
        labelRow = list(range(0, 785))
        labelRow[0] = 'label'
        self.final_data.append(labelRow)
        for link in self.links:
            self.final_data.append(self.format_data(link[0], self.get_pixels(link[1])))


def main():
    db = MakeMnist()
    db.get_link_array()
    db.make_dataset()
    db.create_csv('./data/scrape-data.csv', db.final_data)
    db.load_csv('./data/scrape-data.csv')


if __name__ == '__main__':
    main()

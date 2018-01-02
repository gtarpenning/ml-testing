from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from random import randint


class MnistModel(object):
    def __init__(self):
        """ Model Defaults """
        self.batch_size = 100
        self.epochs = 50
        self.num_classes = 10
        self.pic_size = [28, 28]

        self.train_data = []
        self.test_data = []
        self.model = ''

        self.input_shape = ''
        self.f_data = {}

    def set_model_vars(self, batch, epochs, classes):
        if batch:
            self.batch_size = batch
        if epochs:
            self.epochs = epochs
        if classes:
            self.num_classes = classes

    def load_data(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)

    def format_data(self):
        self.input_shape = (self.pic_size[0], self.pic_size[1], 1)
        X = np.array(self.train_data.iloc[:, 1:])
        y = to_categorical(np.array(self.train_data.iloc[:, 0]))

        X_train, X_val, self.f_data['y_train'], self.f_data['y_val'] = train_test_split(X, y, test_size=0.2, random_state=13)
        X_test = np.array(self.test_data.iloc[:, 1:])
        self.f_data['y_test'] = to_categorical(np.array(self.test_data.iloc[:, 0]))
        X_train = X_train.reshape(X_train.shape[0], self.pic_size[0], self.pic_size[1], 1)
        X_test = X_test.reshape(X_test.shape[0], self.pic_size[0], self.pic_size[1], 1)
        X_val = X_val.reshape(X_val.shape[0], self.pic_size[0], self.pic_size[1], 1)

        self.f_data['X_train'] = (X_train.astype('float32') / 255)
        self.f_data['X_test'] = (X_test.astype('float32') / 255)
        self.f_data['X_val'] = (X_val.astype('float32') / 255)

    def load_model(self, path):
        self.model = load_model(path)

    def construct_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                kernel_initializer='he_normal', input_shape=self.input_shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        history = self.model.fit(self.f_data['X_train'], self.f_data['y_train'],
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                validation_data=(self.f_data['X_val'], self.f_data['y_val']))

    def run(self):
        score = self.model.evaluate(self.f_data['X_test'],
                self.f_data['y_test'], verbose=0)
        print(score)

    def save(self, path):
        self.model.save(path)


def main():
    clothing_mnist = MnistModel()
    clothing_mnist.set_model_vars(100, 10, 10)
    clothing_mnist.load_data('./data/f-mnist-train.csv', './data/f-mnist-test.csv')
    clothing_mnist.format_data()
    clothing_mnist.construct_model()
    clothing_mnist.run()
    clothing_mnist.save('./models/10-mnist.h5')


if __name__ == '__main__':
    main()

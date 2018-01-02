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


data_train = pd.read_csv('./data/fashion-mnist_train.csv')
data_test = pd.read_csv('./data/scrape-data.csv')

""" Model Variables """
batch_size = 100
num_classes = 10
epochs = 50

# Labels and their classes
CLASSES = {
    '0': 'T-shirt/top',
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

answer = input("Hit enter to load saved model ")


# image dimensions
img_rows, img_cols = 28, 28
# 1 is the depth
input_shape = (img_rows, img_cols, 1)
#Test data, excluding labels
X = np.array(data_train.iloc[:, 1:])

# This makes the y data into a 10 dimensional class descriptions
# Example: [6] --> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
y = to_categorical(np.array(data_train.iloc[:, 0]))

#Here we split validation data to optimize classifier during training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

#Test data, excluding labels
X_test = np.array(data_test.iloc[:, 1:])
# This makes the y data into a 10 dimensional class descriptions
# Example: [6] --> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
y_test = to_categorical(np.array(data_test.iloc[:, 0]))

# We need these three sets to be of the shape (size, height, width, depth)
# Example: (60000, 28, 28, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

# They also need to be floats, and we divide them by 255 (color normalization 0-1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255

model = ''

if answer == '':
    model = load_model("model.h5")
    print("Loaded model from disk")
else:
    # Here we define the model, could be a number of things, Sequential is simple
    model = Sequential()
    """This is the input layer, input shape is (rows, columns, depth).
    the first 2 parameters represent: the number of convolution filters to use,
    and the dimensions of the convolsion kernal."""
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                    kernel_initializer='he_normal', input_shape=input_shape))
    # Now we add more layers, colvulsion, max pooling, and dropout
    """
    Dropout layers: prevents overfitting of the data, really important.
    MaxPooling2D: reduces the number of parameters, slides a (x,x) filter over the
                previous filter and taking the max of the (x) values.
    """
    # Below are the different Convulsion layers, increasing by 2x.
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.4))

    # For model architecture, we need a fully connected layer and an output layer
    # Flatten() just makes the Convulsion layer weights 1 dimensional.
    model.add(Flatten())
    # Dense layers: first param is output, Keras automatically handles input
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    # Final output layer needs to have dim num_classes, because it is output
    model.add(Dense(num_classes, activation='softmax'))

    # Here we compile the model, defining the type of omptimizer (Adam).
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])


    # This is the fit, where we set batch size, epochs, returning History, which
    # allows for priting along the way
    history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_val, y_val))

    # serialize model to JSON
    model.save('model.h5')  # creates a HDF5 file 'my_model.h5'

score = model.evaluate(X_test, y_test, verbose=0)
print(score)


""" This is the advanced printing section, NOT REQUIRED """

# get the predictions for the test data
predicted_classes = model.predict_classes(X_test)

#get the indices to be plotted
y_true = data_test.iloc[:, 0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))

# What do activations look like?
test_im = X_test[randint(0, len(X_test))]
plt.imshow(test_im.reshape(28,28), cmap='viridis', interpolation='none')
plt.show()

# See activation from second level
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(input=model.input, output=layer_outputs)
activations = activation_model.predict(test_im.reshape(1,28,28,1))

first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()


# Plot the activations for all levels
layer_names = []
for layer in model.layers[:-1]:
    layer_names.append(layer.name)
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    if layer_name.startswith('conv'):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()
del model

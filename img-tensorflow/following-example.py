import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tf.logging.set_verbosity(tf.logging.ERROR)
print('TF version: ', tf.__version__)


MODEL_DIR = './fashion-model'
TEST_FILE = './data/fashion-mnist_test.csv'
TRAIN_FILE = './data/fashion-mnist_train.csv'

BATCH_SIZE = 40
TRAIN_STEPS = 2000

feature_columns = [tf.feature_column.numeric_column('pixels', shape=[28, 28])]
classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    n_classes=10,
    model_dir=MODEL_DIR
)

# Uses Tensorflow Queues to load in the data
# Function returns a features dictionary (strings), and a labels tensor w/ classes

def generate_labelled_input_fn(csv_files, batch_size):
    def input_fn():
        file_queue = tf.train.string_input_producer(csv_files)
        reader = tf.TextLineReader(skip_header_lines=1)
        _, rows = reader.read_up_to(file_queue, num_records=100*batch_size)
        expanded_rows = tf.expand_dims(rows, axis=-1)

        shuffled_rows = tf.train.shuffle_batch(
            [expanded_rows],
            batch_size=batch_size,
            capacity=20*batch_size,
            min_after_dequeue=5*batch_size,
            enqueue_many=True
        )

        record_defaults = [[0] for _ in range(28*28+1)]

        columns = tf.decode_csv(shuffled_rows, record_defaults=record_defaults)

        labels = columns[0]

        pixels = tf.concat(columns[1:], axis=1)

        return {'pixels': pixels}, labels

    return input_fn


# This is the entire training using the Queue function
classifier.train(
    input_fn=generate_labelled_input_fn([TRAIN_FILE], BATCH_SIZE),
    steps=TRAIN_STEPS
)

# Testing the trained classifier on test data
print(classifier.evaluate(
    input_fn=generate_labelled_input_fn([TEST_FILE], BATCH_SIZE),
    steps=100
))

# We have a trained model with 82% accuracy !!!
# Now to make predictions

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

# Grabbing a test row
test_data = pd.read_csv(TEST_FILE)
sample_row = test_data.sample()

# Formatting the row into label and pixels
sample = list(sample_row.iloc[0])
label = sample[0]
pixels = sample[1:]

# Draw the actual image based on the pixel data
image_array = np.asarray(pixels, dtype=np.float32).reshape((28, 28))
plt.imshow(image_array, cmap='gray')
plt.show()

# This is the prediction flow
def generate_prediction_input_fn(image_arrays):
    def input_fn():
        queue = tf.train.input_producer(
            tf.constant(np.asarray(image_arrays)),
            num_epochs=1
        )

        image = queue.dequeue()
        return {'pixels': [image]}

    return input_fn

# Predict the output for the 'image_array' array we made earlier
predictions = classifier.predict(
    generate_prediction_input_fn([image_array]),
    predict_keys=['probabilities', 'classes']
)

# Raw print of the output
prediction = next(predictions)
print('Prediction output: {}'.format(prediction))

# Formatted print of output, labels, and confidence
print('Actual label: {} - {}'.format(label, CLASSES[str(label)]))
predicted_class = prediction['classes'][0].decode('utf-8')
probability = prediction['probabilities'][int(predicted_class)]
print('Predicted class: {} - {} with probability {}'.format(
    predicted_class,
    CLASSES[predicted_class],
    probability
))

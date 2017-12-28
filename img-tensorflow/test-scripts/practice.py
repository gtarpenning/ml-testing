import tensorflow as tf
from random import shuffle
import glob
import pandas
import numpy as np


def data_bin(train_path):
    csv = pandas.read_csv(train_path).values
    feature_bin = []
    label_bin = []
    for row in csv:
       feature_bin.append(row[1:])
       zeros_bin = np.zeros(10)
       zeros_bin[row[0]] = 1
       label_bin.append(zeros_bin)

    label_bin = np.array(label_bin)
    feature_bin = np.array(feature_bin)

    return label_bin, feature_bin


# X is a placeholder for the 784 dim vector
x = tf.placeholder(tf.float32, [None, 784])

# Weights and Biases Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Defining the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# placeholder for cross entropy (how far away from the truth vector)
y_ = tf.placeholder(tf.float32, [None, 10])

# Actual Cross Entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Training! Gradian Descent Algorithm
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# varible init
init_op = tf.global_variables_initializer()

# LOAD Training DATA

img_train_path = './data/fashion-mnist_train.csv'
train_label_bin, train_feature_bin = data_bin(img_train_path)

img_test_path = './data/fashion-mnist_test.csv'
test_label_bin, test_feature_bin = data_bin(img_train_path)

# Start a new session
with tf.Session() as sess:
    sess.run(init_op)

    # Training step
    for _ in range(5):
        idx = np.random.randint(len(train_feature_bin), size=100)
        batch_xs = train_feature_bin[idx,]
        batch_ys = train_label_bin[idx,]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Prediction
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # Turns correct prediction into accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Test!
    print(sess.run(accuracy, feed_dict={x: test_feature_bin, y_: test_label_bin}))

import tensorflow as tf
import pandas
import numpy as np
import random


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

def getBatch(x, y):
    rand_list = []
    for i in range(len(x)):
        rand_list.append([x[i], y[i]])

    random.shuffle(rand_list)
    batch = rand_list[:batch_size]
    x_batch = [item[0] for item in rand_list[:batch_size]]
    y_batch = [item[1] for item in rand_list[:batch_size]]

    x = [item[0] for item in rand_list[batch_size:]]
    y = [item[1] for item in rand_list[batch_size:]]

    return np.array(x_batch), np.array(y_batch), x, y


img_train_path = './data/fashion-mnist_train.csv'
train_label_bin, train_feature_bin = data_bin(img_train_path)

img_test_path = './data/fashion-mnist_test.csv'
test_label_bin, test_feature_bin = data_bin(img_train_path)

# Python optimisation variables
learning_rate = 0.1
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                    + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Set up the global variable initializer
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TRAINING

with tf.Session() as sess:
    # init Variables
    sess.run(init_op)

    total_batch = int(len(train_label_bin) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        i_count = []
        tf = train_feature_bin
        tl = train_label_bin
        for i in range(total_batch):
            batch_x, batch_y, tf, tl = getBatch(tf, tl)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: test_feature_bin, y: test_label_bin}))

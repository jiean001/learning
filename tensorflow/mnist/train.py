from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
import numpy as np

NUM_CLASS = 10

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# download the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


# the convolution layer
# output = max_pool_2x2(activation_function(conv2d(inputs, weights) + biases))
def add_conv_layer(inputs, n_layer, W_shape, b_shape, activation_function=tf.nn.relu):
    layer_name = "convolution_layer%s" %n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights"):
            Weights = weight_variable(W_shape)
            tf.summary.histogram(layer_name + "/Weights", Weights)
        with tf.name_scope("biases"):
            biases = bias_variable(b_shape)
            tf.summary.histogram(layer_name + "/biases", biases)
        with tf.name_scope("Conv_Wx_plus_b"):
            Conv_Wx_plus_b = conv2d(inputs, Weights) + biases
            tf.summary.histogram(layer_name + "/Conv_Wx_plus_b", Conv_Wx_plus_b)
        with tf.name_scope("hidden_conv"):
            hidden_conv = activation_function(Conv_Wx_plus_b)
            tf.summary.histogram(layer_name + "/hidden_conv", hidden_conv)
        with tf.name_scope("hidden_pool"):
            hidden_pool = max_pool_2x2(hidden_conv)
            tf.summary.histogram(layer_name + "/hidden_pool", hidden_pool)
        return hidden_pool


# the full connect layer
# output = dropout(activation_function(matmul(Weights, input.reshape) + biases), keep_prob)
def add_fc_layer(inputs, n_layer, W_shape, b_shape, in_shape, keep_prob, activation_function=tf.nn.relu):
    layer_name = "full_connection_layer%s" %n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope("reshape_input"):
            reshape_input = tf.reshape(inputs, in_shape)
            tf.summary.histogram(layer_name + "/reshape_input", reshape_input)
        with tf.name_scope("Weights"):
            Weights = weight_variable(W_shape)
            tf.summary.histogram(layer_name + "/Weights", Weights)
        with tf.name_scope("biases"):
            biases = bias_variable(b_shape)
            tf.summary.histogram(layer_name + "/biases", biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(reshape_input, Weights) + biases
            tf.summary.histogram(layer_name + "/Wx_plus_b", Wx_plus_b)
        with tf.name_scope("hidden_fc"):
            hidden_fc = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + "/hidden_fc", hidden_fc)
        with tf.name_scope("dropout"):
            drop = tf.nn.dropout(hidden_fc, keep_prob)
            tf.summary.histogram(layer_name + "/dropout", drop)
        return drop


# the output layer
# outputs = tf.nn.softmax(tf.matmul(inputs, Weights) + biases)
def add_output_layer(inputs, W_shape, b_shape):
    layer_name = "output_layer"
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights"):
            Weights = weight_variable(W_shape)
            tf.summary.histogram(layer_name + "/Weights", Weights)
        with tf.name_scope("biases"):
            biases = bias_variable(b_shape)
            tf.summary.histogram(layer_name + "/biases", biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            tf.summary.histogram(layer_name + "/Wx_plus_b", Wx_plus_b)
        with tf.name_scope("outputs"):
            outputs = tf.nn.softmax(Wx_plus_b)
            tf.summary.histogram(layer_name + "/outputs", outputs)
        return outputs

with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope("keep_prob"):
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("inputs_layer"):
    x_image = tf.reshape(x, [-1, 28, 28, 1])


conv1_W_shape = [5, 5, 1, 32]
conv1_b_shape = [32]
# the convolution layer 1's shape is None * 14 * 14 * 32
conv1_layer = add_conv_layer(inputs=x_image, n_layer=1, W_shape=conv1_W_shape, b_shape=conv1_b_shape)

conv2_W_shape = [5, 5, 32, 64]
conv2_b_shaoe = [64]
# the convolution layer 2's shape is None * 7 * 7 *64
conv2_layer = add_conv_layer(inputs=conv1_layer, n_layer=2, W_shape=conv2_W_shape, b_shape=conv2_b_shaoe)

fc1_W_shape = [7 * 7 * 64, 1024]
fc1_b_shape = [1024]
in_reshape = [-1, 7 * 7 * 64]
# the full connection layer's shape is None * 1024
fc1_layer = add_fc_layer(inputs=conv2_layer, in_shape=in_reshape,
                   n_layer=1, W_shape=fc1_W_shape, b_shape=fc1_b_shape, keep_prob=keep_prob)

fc2_W_shape = [1024, 10]
fc2_b_shape = [10]
# the softmax layer's shape is None * 10
y_conv = add_output_layer(inputs=fc1_layer, W_shape=fc2_W_shape, b_shape=fc2_b_shape)

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./tmp/", sess.graph)

sess.run (tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
             x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
saver.save(sess, './form/model.ckpt')

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


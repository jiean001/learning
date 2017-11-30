from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.misc
import pprint
from concat_tmp import *

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
# todo
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). "
                                          "If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. "
                                           "If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, weights, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

# utils
def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)


def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
      if crop_w is None:
            crop_w = crop_h
      h, w = x.shape[:2]
      j = int(round((h - crop_h)/2.))
      i = int(round((w - crop_w)/2.))
      return scipy.misc.imresize(
          x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

class LetterGenerator:
    def __init__(self, sess, input_height=28, input_width=108,
                 batch_size=100, sample_number=1, output_height=108, output_width=108, cnn_f_dim=64,
                 fc_dim=1024, n_dataset_name='./data/normalization_pic', gt_dataset_name="./data/groundtruth_pic",
                 input_fname_patter='*.jpg', checkpoint_dir="./checkpoint", sample_dir="./samples"):
        """
        :param sess: TensorFlow session
        :param input_height: the height of input images
        :param input_width: the width of input images
        :param batch_size: the size of batch. should be specified before training
        :param sample_number: the number of samples
        :param output_height: ...
        :param output_width: ...
        :param cnn_f_dim: the feature number peer layer
        :param fc_dim: Dimension of gen filers in last conv layer and the first decon layer
        :param n_dataset_name: normalization images
        :param gt_dataset_name: the ground truth images
        :param input_fname_patter: ...
        :param checkpoint_dir: ...
        :param sample_dir: ...
        """
        self.sess = sess
        self.batch_size = batch_size
        self.sample_number = sample_number

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.output_c_dim = 3
        self.input_c_dim = 4

        self.cnn_f_dim = cnn_f_dim
        self.fc_dim = fc_dim

        self.stain_data = Data(sess)

        # batch normalization: deals with poor initialization helps gradient flow
        # convolution layer
        self.conv_bn1 = batch_norm(name="conv_bn1")
        self.conv_bn2 = batch_norm(name="conv_bn2")
        self.conv_bn3 = batch_norm(name="conv_bn3")

        # deconvolution layer
        self.dconv_bn0 = batch_norm(name="dconv_bn0")
        self.dconv_bn1 = batch_norm(name="dconv_bn1")
        self.dconv_bn2 = batch_norm(name="dconv_bn2")
        self.dconv_bn3 = batch_norm(name="dconv_bn3")

        self.n_dataset_name = n_dataset_name
        self.gt_dataset_name = gt_dataset_name
        self.input_fname_patter = input_fname_patter

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        self.data_n_images = glob(os.path.join("./data", self.n_dataset_name, self.input_fname_patter))
        self.data_gt_images = glob(os.path.join("./data", self.gt_dataset_name, self.input_fname_patter))
        self.build_model()

    def build_model(self):
        n_image_dim = [self.input_height, self.input_width, 1]
        gt_image_dim = [self.input_height, self.input_width, 3]
        sample_image_dim = [self.input_height, self.input_width, 4]
        real_image_dim = [self.input_height, self.input_width, 3]
        output_image_dim = [self.output_height, self.output_width, 3]

        # 输入的标准化的灰度letter+样本letter组成的4-d图像
        self.inputs_sample_images = tf.placeholder(tf.float32, [self.batch_size] + sample_image_dim, name='inputs_sample_images')
        self.inputs_n_image = tf.placeholder(tf.float32, [self.batch_size] + n_image_dim, name='normalization_image')
        self.inputs_gt_image = tf.placeholder(tf.float32, [self.batch_size] + gt_image_dim, name='ground_truth_image')
        self.inputs_real_image = tf.placeholder(tf.float32, [self.batch_size] + real_image_dim, name='real_image')

        inputs_n_image = self.inputs_n_image
        inputs_gt_image = self.inputs_gt_image
        inputs_sample_image = self.inputs_sample_images
        inputs_real_image = self.inputs_real_image

        # 用4维图像去做卷积，得到一个feature map
        self.C = self.conv_layer(inputs_sample_image, reuse=False)
        # 卷积过后得到的feature map，经过deconvolution得到一副生成的rgb图像
        self.D = self.deconv_layer(self.C)
        # 描述损失
        self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.D, inputs_real_image), 2.0))

        '''
        self.C_sum = histogram_summary("conv", self.C)
        self.D__sum = histogram_summary("deconv", self.D)
        '''
        self.loss_sum = scalar_summary("loss", self.loss)

        self.t_vars = tf.trainable_variables()
        #self.c_vars = [var for var in t_vars if 'c_' in var.name]
        #self.d_vars = [var for var in t_vars if 'd_' in var.name]

        self.saver = tf.train.Saver()


    def train(self, config):
        optim = tf.train.AdamOptimizer(config.learning_rate,
                                       beta1=config.beta1).minimize(self.loss, var_list=self.t_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # self.all_sum = merge_summary([self.C_sum, self.D__sum, self.loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        could_load, checkpoint = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint
            print(" [*] load success")
        else:
            print(" [!] load failed...")

        for epoch in range(2):
            self.data = glob(os.path.join("./data", "format_0", self.input_fname_patter))
            batch_idxs = min(len(self.data), config.train_size)

            batch_images, label_images = self.stain_data.get_batch_data(1, self.sess)
            feed_dict = {self.inputs_sample_images: batch_images, self.inputs_real_image: label_images}
            _, summary_str = self.sess.run([optim, self.loss_sum], feed_dict=feed_dict)
            self.writer.add_summary(summary_str, counter)
            loss = self.loss.eval(feed_dict)

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f loss:%.8f" %(epoch, 1, 2, time.time() - start_time, loss))

        if np.mod(counter, 500) == 2:
            self.save(config.checkpoint_dir, counter)


    # the shape of image is [batch_size, input_height, input_width, 4]
    def conv_layer(self, image, reuse=False):
        with tf.variable_scope("conv_layer") as scope:
            if reuse:
                scope.reuse_vairables()
            h0 = lrelu(conv2d(image, self.cnn_f_dim, name='h0_conv'))
            h1 = lrelu(self.conv_bn1(conv2d(h0, self.cnn_f_dim * 2, name='h1_conv')))
            h2 = lrelu(self.conv_bn2(conv2d(h1, self.cnn_f_dim * 4, name='h2_conv')))
            h3 = lrelu(self.conv_bn3(conv2d(h2, self.cnn_f_dim * 8, name='h3_conv')))
            h4 = tf.reshape(h3, [self.batch_size, -1])
            return h4

    def deconv_layer(self, feature_map):
        with tf.variable_scope("deconv_layer") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # no noise
            self.z_ = feature_map
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.cnn_f_dim * 8])
            h0 = tf.nn.relu(self.dconv_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                                                     [self.batch_size, s_h8, s_w8, self.cnn_f_dim * 4],
                                                        name="deconv_h1", with_w=True)
            h1 = tf.nn.relu(self.dconv_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.cnn_f_dim * 2],
                                                      name="deconv_h2", with_w=True)
            h2 = tf.nn.relu(self.dconv_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.cnn_f_dim * 1],
                                                name="deconv_h3", with_w=True)
            h3 = tf.nn.relu(self.dconv_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, 3],  name="deconv_h4", with_w=True)
            return tf.nn.tanh(h4)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.gt_dataset_name, self.batch_size, self.output_height, self.output_width)

    def saver(self, checkpoint_dir, step):
        model_name = "LETTERGENERATOE.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

import tensorflow.contrib.slim as slim


def show_all_variables():
     model_vars = tf.trainable_variables()
     slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main():
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        model = LetterGenerator(
            sess, input_height=96, input_width=96,
                 batch_size=100, sample_number=1, output_height=96, output_width=96, cnn_f_dim=64,
                 fc_dim=1024, n_dataset_name='./data/normalization_pic', gt_dataset_name="./data/groundtruth_pic",
                 input_fname_patter='*.jpg', checkpoint_dir="./checkpoint", sample_dir="./samples"
            )
        show_all_variables()
        model.train(FLAGS)


if __name__ == '__main__':
    # tf.app.run()
    main()

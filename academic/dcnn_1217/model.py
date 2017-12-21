from __future__ import division
import sys
sys.path.append('../common')
import os
import time
import math
import tensorflow as tf
import numpy as np
from six.moves import xrange
from ops import *
from utils import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
    def __init__(self, sess,
                 input_height=108,
                 input_width=108,
                 crop=True,
                 batch_size=64,
                 output_height=64,
                 output_width=64,
                 gf_dim=64,
                 df_dim=64,
                 input_c_dim=4,
                 dataset_name='default',
                 input_fname_pattern='*.jpg',
                 checkpoint_dir=None):
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.input_c_dim = input_c_dim
        self.cnn_f_dim = df_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.g_c_bn1 = batch_norm(name="g_c_bn1")
        self.g_c_bn2 = batch_norm(name="g_c_bn2")
        self.g_c_bn3 = batch_norm(name="g_c_bn3")
        self.g_c_bn4 = batch_norm(name="g_c_bn4")
        self.g_c_bn5 = batch_norm(name="g_c_bn5")
        self.g_c_bn6 = batch_norm(name="g_c_bn6")
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name="g_bn4")
        self.g_bn5 = batch_norm(name="g_bn5")
        self.g_bn6 = batch_norm(name="g_bn6")

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.c_dim = 3
        self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_model(self):
        self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        sample_image_dim = [self.input_height, self.input_width, self.input_c_dim]
        self.z = tf.placeholder(tf.float32, [self.batch_size] + sample_image_dim,
                                name='z')

        self.G                  = self.generator(self.z, self.y)
        self.sampler            = self.sampler(self.z, self.y)

        self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.G, inputs), 2.0))
        self.loss_sum = scalar_summary("loss", self.loss)

        self.G_sum = image_summary("G", self.G)
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def sampler(self, z, y=None):
        return self.generator(z, y, train=True, reuse=True)

    def train(self, config, data):
        optim = tf.train.AdamOptimizer(config.learning_rate,
                                       beta1=config.beta1).minimize(self.loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.writer = SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        fp = open("./loss.txt", "w+")
        for epoch in xrange(config.epoch):
            #fp = open(".loss.txt", "a")
            fp.write("----------------epoch:%d---------------\n" %(epoch))
            data.train_shuffle()
            batch_idxs = min(data.get_train_batch_idxs(), config.train_size)
            train_loss = 0
            test_loss = 0
            for idx in xrange(0, batch_idxs):
                first_imgs, standard_imgs, batch_images = data.get_train_data(idx)
                batch_z = self.sess.run(get_4d_pic(first_imgs, standard_imgs, axis=3))
                _, summary_str, loss = self.sess.run([optim, self.loss_sum, self.loss],
                                                     feed_dict={self.inputs: batch_images,
                                                                self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" % (epoch, idx, batch_idxs,
                                                                          time.time() - start_time, loss))
                train_loss += loss
                #if np.mod(counter, 2000) == 0:
                #   self.save(config.checkpoint_dir, counter)
            for test_idx in range(data.get_test_batch_idxs()):
                test_first_imgs, test_standard_imgs, test_batch_images = data.get_test_data(test_idx)
                test_batch_z_ = self.sess.run(get_4d_pic(test_first_imgs, test_standard_imgs, axis=3))
                try:
                    samples, loss = self.sess.run(
                        [self.sampler, self.loss],
                        feed_dict={
                            self.z: test_batch_z_,
                            self.inputs: test_batch_images,
                        },
                    )
                    save_images(test_first_imgs, image_manifold_size(test_first_imgs.shape[0]),
                                './{}/origin_{:02d}_{:04d}.jpg'.format(config.sample_dir, epoch, test_idx))
                    save_images(test_standard_imgs, image_manifold_size(test_standard_imgs.shape[0]),
                                './{}/standard_{:02d}_{:04d}.jpg'.format(config.sample_dir, epoch, test_idx))
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.jpg'.format(config.sample_dir, epoch, test_idx))
                    save_images(test_batch_images, image_manifold_size(test_batch_images.shape[0]),
                                './{}/traget_{:02d}_{:04d}.jpg'.format(config.sample_dir, epoch, test_idx))
                    print("[Sample] loss: %.8f" % (loss))
                    test_loss += loss
                except:
                    print("one pic error!...")
                    return
            print("average train loss: %4.4f, average test loss: %4.4f" %((train_loss / batch_idxs), (test_loss / data.get_test_batch_idxs())))
            fp.write("average train loss %4.4f, average test loss: %4.4f\n\n" %((train_loss / batch_idxs), (test_loss / data.get_test_batch_idxs())))
        fp.close()


    def generator(self, image, y=None, train=False, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            g_c_h0 = lrelu(conv2d(image, self.cnn_f_dim, d_h=1, d_w=1,name='g_h0_conv'))
            g_c_h1 = lrelu(self.g_c_bn1(conv2d(g_c_h0, self.cnn_f_dim * 2, d_h=1, d_w=1, name='g_h1_conv'), train=train))
            g_c_h1 = max_pool_2x2(g_c_h1, name='g_h1_pool')
            g_c_h2 = lrelu(self.g_c_bn2(conv2d(g_c_h1, self.cnn_f_dim * 4, d_h=1, d_w=1, name='g_h2_conv'), train=train))
            g_c_h3 = lrelu(self.g_c_bn3(conv2d(g_c_h2, self.cnn_f_dim * 8, d_h=1, d_w=1, name='g_h3_conv'), train=train))
            g_c_h3 = max_pool_2x2(g_c_h3, name='g_h3_pool')
            #g_c_h4 = lrelu(self.g_c_bn4(conv2d(g_c_h3, self.cnn_f_dim * 16, d_h=1, d_w=1, name='g_h4_conv'), train=train))
            #g_c_h5 = lrelu(self.g_c_bn5(conv2d(g_c_h4, self.cnn_f_dim * 32, d_h=1, d_w=1, name='g_h5_conv'), train=train))
            #g_c_h5 = max_pool_2x2(g_c_h5, name='g_h5_pool')

            g_c_h7 = tf.reshape(g_c_h3, [self.batch_size, -1])




            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            # luxb modify
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)
            s_h128, s_w128 = conv_out_size_same(s_h64, 2), conv_out_size_same(s_w64, 2)

            # project `z` and reshape
            '''
            self.z_, self.h0_w, self.h0_b = linear(
                g_c_h7, self.gf_dim * 64 * s_h128 * s_w128, 'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(
                self.z_, [-1, s_h128, s_w128, self.gf_dim * 64])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=train))
            '''

            self.z_, self.h0_w, self.h0_b = linear(
                g_c_h7, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_h16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=train))

            '''
            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h64, s_w64, self.gf_dim * 32], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1, train=train))

            h2, self.h2_w, self.h2_b = deconv2d(
                h0, [self.batch_size, s_h32, s_w32, self.gf_dim * 16], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2, train=train))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h16, s_w16, self.gf_dim * 8], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3, train=train))
            '''

            h4, self.h4_w, self.h4_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4, train=train))

            h5, self.h5_w, self.h5_b = deconv2d(
                h4, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn5(h5, train=train))

            h6, self.h6_w, self.h6_b = deconv2d(
                h5, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h6', with_w=True)
            h6 = tf.nn.relu(self.g_bn6(h6, train=train))

            h7, self.h7_w, self.h7_b = deconv2d(
                h6, [self.batch_size, s_h, s_w, self.c_dim], name='g_h7', with_w=True)

            return tf.nn.tanh(h7)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
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
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

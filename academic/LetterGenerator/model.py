from __future__ import division
import os
import time
from utils import *
import tensorflow as tf
import numpy as np
import math


class LetterGenerator:
    def __init__(self, sess, input_height=28, input_width=28,
                 output_height=28, output_width=28, cnn_f_dim=64,batch_size=500,
                 fc_dim=1024, input_fname_patter='*.jpg', checkpoint_dir="./checkpoint", generator_dir="./generator"):
        self.sess = sess
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.output_c_dim = 3
        self.input_c_dim = 4

        self.cnn_f_dim = cnn_f_dim
        self.fc_dim = fc_dim
        self.generator_dir = generator_dir

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

        self.input_fname_patter = input_fname_patter

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        sample_image_dim = [self.input_height, self.input_width, self.input_c_dim]
        real_image_dim = [self.input_height, self.input_width, self.output_c_dim]

        # 输入的标准化的灰度letter+样本letter组成的4-d图像
        self.inputs_sample_images = tf.placeholder(tf.float32, [self.batch_size] + sample_image_dim, name='inputs_sample_images')
        self.inputs_real_image = tf.placeholder(tf.float32, [self.batch_size] + real_image_dim, name='real_image')

        inputs_sample_image = self.inputs_sample_images
        inputs_real_image = self.inputs_real_image

        # 用4维图像去做卷积，得到一个feature map
        self.C = self.conv_layer(inputs_sample_image, reuse=False, train=True)
        # 卷积过后得到的feature map，经过deconvolution得到一副生成的rgb图像
        self.D = self.deconv_layer(self.C, reuse=False, train=True)
        generator_C = self.conv_layer(inputs_sample_image, reuse=True, train=False)
        generator_D = self.deconv_layer(generator_C, reuse=True, train=False)
        self.generator = generator_D

        # 描述损失
        self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.D, inputs_real_image), 2.0))

        self.loss_sum = scalar_summary("loss", self.loss)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    # the shape of image is [batch_size, input_height, input_width, 4]
    def conv_layer(self, image, reuse=False, train=True):
        with tf.variable_scope("conv_layer") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, self.cnn_f_dim, name='h0_conv'))
            h1 = lrelu(self.conv_bn1(conv2d(h0, self.cnn_f_dim * 2, name='h1_conv'), train=train))
            h2 = lrelu(self.conv_bn2(conv2d(h1, self.cnn_f_dim * 4, name='h2_conv'), train=train))
            h3 = lrelu(self.conv_bn3(conv2d(h2, self.cnn_f_dim * 8, name='h3_conv'), train=train))
            h4 = tf.reshape(h3, [self.batch_size, -1])
            return h4

    def deconv_layer(self, feature_map, reuse=False, train=True):
        with tf.variable_scope("deconv_layer") as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            # no noise
            self.z_ = feature_map
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.cnn_f_dim * 8])
            h0 = tf.nn.relu(self.dconv_bn0(self.h0, train=train))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                                                     [self.batch_size, s_h8, s_w8, self.cnn_f_dim * 4],
                                                        name="deconv_h1", with_w=True)
            h1 = tf.nn.relu(self.dconv_bn1(self.h1, train=train))

            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.cnn_f_dim * 2],
                                                      name="deconv_h2", with_w=True)
            h2 = tf.nn.relu(self.dconv_bn2(h2, train=train))

            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.cnn_f_dim * 1],
                                                name="deconv_h3", with_w=True)
            h3 = tf.nn.relu(self.dconv_bn3(h3, train=train))

            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, 3],  name="deconv_h4", with_w=True)
            return tf.nn.tanh(h4)

    def train(self, config, train_data):
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

        for epoch in range(config.epoch):
            batch_idxs = min(math.ceil(train_data.get_len()/config.batch_size) - 1, config.train_size)
            for idx in range(batch_idxs):
                batch_images, label_images = train_data.get_batch_data(idx, self.sess)
                feed_dict = {self.inputs_sample_images: batch_images, self.inputs_real_image: label_images}
                _, summary_str = self.sess.run([optim, self.loss_sum], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)
                loss = self.loss.eval(feed_dict)
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f loss:%.8f" %(epoch, idx, batch_idxs, time.time() - start_time, loss))

                if np.mod(counter, 50) == 10:
                    generator, loss = self.sess.run([self.generator, self.loss], feed_dict=feed_dict)
                    save_images(label_images, image_manifold_size(generator.shape[0]),
                              './{}/origin_{:02d}_{:04d}.png'.format(config.generator_dir, epoch, idx))
                    save_images(generator, image_manifold_size(generator.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.generator_dir, epoch, idx))
                    print("[generator] loss: %.8f" % (loss))
                if np.mod(counter, 20) == 2:
                    self.save(config.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format("format_255", self.batch_size, self.output_height, self.output_width)

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

    def save(self, checkpoint_dir, step):
        model_name = "letter_generator.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

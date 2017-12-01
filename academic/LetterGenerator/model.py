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
        self.conv_bn1 = batch_norm(name='g_conv_bn1')
        self.conv_bn2 = batch_norm(name='g_conv_bn2')
        self.conv_bn3 = batch_norm(name='g_conv_bn3')
        self.dconv_bn0 = batch_norm(name='g_dconv_bn0')
        self.dconv_bn1 = batch_norm(name='g_dconv_bn1')
        self.dconv_bn2 = batch_norm(name='g_dconv_bn2')
        self.dconv_bn3 = batch_norm(name='g_dconv_bn3')

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')



        self.input_fname_patter = input_fname_patter

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        sample_image_dim = [self.input_height, self.input_width, self.input_c_dim]
        real_image_dim = [self.input_height, self.input_width, self.output_c_dim]

        self.inputs_sample_images = tf.placeholder(tf.float32, [self.batch_size] + sample_image_dim, name='inputs_sample_images')
        self.inputs_real_image = tf.placeholder(tf.float32, [self.batch_size] + real_image_dim, name='real_image')

        inputs_sample_image = self.inputs_sample_images
        inputs_real_image = self.inputs_real_image

        self.z_sum = histogram_summary("z", self.inputs_sample_images)

        self.G = self.generator(inputs_sample_image, reuse=False, train=True)
        self.D, self.D_logits = self.discriminator(inputs_real_image, reuse=False)
        self.sampler = self.sampler(inputs_sample_image)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def sampler(self, image, reuse=True, train=False):
        return self.generator(image, reuse=reuse, train=train)

    def generator(self, image, reuse=False, train=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            # convolution layer
            g_c_h0 = lrelu(conv2d(image, self.cnn_f_dim, name='g_h0_conv'))
            g_c_h1 = lrelu(self.conv_bn1(conv2d(g_c_h0, self.cnn_f_dim * 2, name='g_h1_conv'), train=train))
            g_c_h2 = lrelu(self.conv_bn2(conv2d(g_c_h1, self.cnn_f_dim * 4, name='g_h2_conv'), train=train))
            g_c_h3 = lrelu(self.conv_bn3(conv2d(g_c_h2, self.cnn_f_dim * 8, name='g_h3_conv'), train=train))
            g_c_h4 = tf.reshape(g_c_h3, [self.batch_size, -1])

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            # todo

            self.z_ = g_c_h4
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.cnn_f_dim * 8])
            g_dc_h0 = tf.nn.relu(self.dconv_bn0(self.h0, train=train))
            self.h1, self.h1_w, self.h1_b = deconv2d(g_dc_h0,
                                                     [self.batch_size, s_h8, s_w8, self.cnn_f_dim * 4],
                                                     name='g_deconv_h1', with_w=True)
            g_dc_h1 = tf.nn.relu(self.dconv_bn1(self.h1, train=train))
            g_dc_h2, self.h2_w, self.h2_b = deconv2d(g_dc_h1, [self.batch_size, s_h4, s_w4, self.cnn_f_dim * 2],
                                                name='g_deconv_h2', with_w=True)
            g_dc_h2 = tf.nn.relu(self.dconv_bn2(g_dc_h2, train=train))
            g_dc_h3, self.h3_w, self.h3_b = deconv2d(g_dc_h2, [self.batch_size, s_h2, s_w2, self.cnn_f_dim * 1],
                                                name='g_deconv_h3', with_w=True)
            g_dc_h3 = tf.nn.relu(self.dconv_bn3(g_dc_h3, train=train))
            g_dc_h4, self.h4_w, self.h4_b = deconv2d(g_dc_h3, [self.batch_size, s_h, s_w, 3], name='g_deconv_h4', with_w=True)
            return tf.nn.tanh(g_dc_h4)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.cnn_f_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.cnn_f_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.cnn_f_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.cnn_f_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
            return tf.nn.sigmoid(h4), h4

    def train(self, config, train_data):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        could_load, checkpoint = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint
            print(" [*] load success")
        else:
            print(" [!] load failed...")

        # todo
        temp = True
        for epoch in range(config.epoch):
            # train_data.shuffle()
            batch_idxs = min(math.ceil(train_data.get_len()/config.batch_size) - 1, config.train_size)
            for idx in range(batch_idxs):
                batch_images, label_images = train_data.get_batch_data(idx, self.sess)
                feed_dict = {self.inputs_sample_images: batch_images, self.inputs_real_image: label_images}
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval(feed_dict)
                errD_real = self.d_loss_real.eval(feed_dict)
                errG = self.g_loss.eval(feed_dict)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    %(epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 20) == 1:
                    generator, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict=feed_dict
                    )
                    try:
                        if temp:
                            temp = False
                            save_images(label_images, image_manifold_size(generator.shape[0]),
                                   './{}/origin_{:02d}_{:04d}.png'.format(config.generator_dir, epoch, idx))
                        save_images(generator, image_manifold_size(generator.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.generator_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except:
                        print("one pic error!...")
                if np.mod(counter, 100) == 2:
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

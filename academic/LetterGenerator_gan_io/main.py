import os
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables
from train_data import Data
from test_data import TestData

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

'''luxb add begin'''
flags.DEFINE_string("samples_path", "../../../../data/m_dcgan/sample/", "samples path")
flags.DEFINE_string("standard_path", "../../../../data/m_dcgan/standard", "standard path")
flags.DEFINE_string("origin_path", "origin", "black_path")
flags.DEFINE_string("black_path", "format_0", "black_path")
flags.DEFINE_string("white_path", "format_255", "black_path")
flags.DEFINE_string("generator_dir", "generator", "generator_dir")
flags.DEFINE_string("config_file", "config.txt", "config_file")
flags.DEFINE_string("train_config_file", "train_config.txt", "config_file")
flags.DEFINE_string("train_config_path", "./runconfig/", "s_dir")
flags.DEFINE_boolean("is_create_train_config", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("train_black", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("train_white", True, "True for training, False for testing [False]")
'''luxb add end'''


FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    '''luxb add begin'''
    if not os.path.exists(FLAGS.samples_path):
        os.makedirs(FLAGS.samples_path)
    if not os.path.exists(FLAGS.standard_path):
        os.makedirs(FLAGS.standard_path)
    if not os.path.exists(FLAGS.generator_dir):
        os.makedirs(FLAGS.generator_dir)
    if not os.path.exists("runconfig"):
        os.makedirs("runconfig")
    if not os.path.exists("test"):
        os.makedirs("test")

    whole_black_path = os.path.join(FLAGS.samples_path, FLAGS.black_path)
    whole_white_path = os.path.join(FLAGS.samples_path, FLAGS.white_path)
    whole_standard_black_path = os.path.join(FLAGS.standard_path, FLAGS.black_path)
    whole_standard_white_path = os.path.join(FLAGS.standard_path, FLAGS.white_path)
    '''luxb add end'''


    # init the train data
    if FLAGS.train_black:
        train_data_black = Data(sampler_images=whole_black_path, standard_pic_dir=whole_standard_black_path,
                                input_fname_pattern=FLAGS.input_fname_pattern,
                                config_file=FLAGS.config_file,
                                train_config_file=os.path.join(FLAGS.train_config_path, "format_0_train_config.txt"),
                                create_frain_config=FLAGS.is_create_train_config, batch=FLAGS.batch_size)

    if FLAGS.train_white:
        train_data_white = Data(sampler_images=whole_white_path, standard_pic_dir=whole_standard_white_path,
                                input_fname_pattern=FLAGS.input_fname_pattern,
                                config_file=FLAGS.config_file,
                                train_config_file=os.path.join(FLAGS.train_config_path, "format_255_train_config.txt"),
                                create_frain_config=FLAGS.is_create_train_config, batch=FLAGS.batch_size)




    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    test_data = TestData(
        test_images="./data/m_dcgan/test/format_255",
        standard_pic_dir=whole_standard_white_path,
        input_fname_pattern=FLAGS.input_fname_pattern,
        test_config_file="./runconfig/format_255_test_config.txt",
        create_frain_config=True,
        batch=FLAGS.batch_size
    )

    with tf.Session(config=run_config) as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=10,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir)
        else:
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir)

        show_all_variables()
        '''luxb delete
        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
        '''

        #dcgan.train(FLAGS, train_data=None)
        dcgan.train(config=FLAGS, train_data=train_data_white, test_data=test_data)
        '''luxb add begin'''
        #if FLAGS.train_black:
            #dcgan.train(config=FLAGS, train_data=train_data_black)
        #if FLAGS.train_white:
            #dcgan.train(config=FLAGS, train_data=train_data_white)
        '''luxb add end'''


        # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
        #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
        #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
        #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
        #                 [dcgan.h4_w, dcgan.h4_b, None])

        # Below is codes for visualization
        OPTION = 1
        #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()

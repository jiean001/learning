from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import pprint
from utils import *
from sample import ImageProcess
from model import *
from standard import Standard
from train_data import Data

flags = tf.app.flags
flags.DEFINE_integer("epoch", 300, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
# todo
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 28, "The size of image to use (will be center cropped). "
                                          "If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 28, "The size of the output images to produce. "
                                           "If None, same value as output_height [None]")
flags.DEFINE_string("samples_path", "../../../../data/m_dcgan/sample/", "samples path")
flags.DEFINE_string("standard_path", "../../../../data/m_dcgan/standard", "standard path")
flags.DEFINE_string("origin_path", "origin", "black_path")
flags.DEFINE_string("black_path", "format_0", "black_path")
flags.DEFINE_string("white_path", "format_255", "black_path")
flags.DEFINE_string("config_file", "config.txt", "config_file")
flags.DEFINE_string("train_config_file", "train_config.txt", "config_file")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("generator_dir", "generator", "generator_dir")
flags.DEFINE_boolean("is_sample_null", False, "True for training, False for testing [False]")
flags.DEFINE_string("gt_dir", "../../../../data/m_dcgan/gt_origin", "gt_dir")
flags.DEFINE_string("s_dir", "../../../../data/m_dcgan/origin/*.jpg", "s_dir")
flags.DEFINE_boolean("is_create_train_config", True, "True for training, False for testing [False]")
flags.DEFINE_string("train_config_path", "./runconfig/", "s_dir")
flags.DEFINE_boolean("train_black", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("train_white", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main():
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_height is None:
        FLAGS.input_height = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_width

    if not os.path.exists(FLAGS.samples_path):
        os.makedirs(FLAGS.samples_path)
    if not os.path.exists(FLAGS.standard_path):
        os.makedirs(FLAGS.standard_path)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.train_config_path):
        os.makedirs(FLAGS.train_config_path)

    whole_black_path = os.path.join(FLAGS.samples_path, FLAGS.black_path)
    whole_white_path = os.path.join(FLAGS.samples_path, FLAGS.white_path)
    whole_standard_black_path = os.path.join(FLAGS.standard_path, FLAGS.black_path)
    whole_standard_white_path = os.path.join(FLAGS.standard_path, FLAGS.white_path)

    # generate format letter and test letter
    if FLAGS.is_sample_null:
        if not os.path.exists(FLAGS.samples_path):
            os.makedirs(FLAGS.samples_path)
        if not os.path.exists(whole_black_path):
            os.makedirs(whole_black_path)
        if not os.path.exists(whole_white_path):
            os.makedirs(whole_white_path)
        output_black_config = os.path.join(FLAGS.black_path, FLAGS.config_file)
        output_white_config = os.path.join(FLAGS.white_path, FLAGS.config_file)
        ImageProcess(gt_dir=FLAGS.gt_dir, s_dir=FLAGS.s_dir, output_dir=FLAGS.samples_path,
                        output_black_config=output_black_config, output_white_config=output_white_config, format_size=[FLAGS.input_height, FLAGS.output_width])

        if not os.path.exists(whole_standard_black_path):
            os.makedirs(whole_standard_black_path)
        if not os.path.exists(whole_standard_white_path):
            os.makedirs(whole_standard_white_path)
        Standard(standard_img_path=FLAGS.standard_path, standard_img_origin_path=FLAGS.origin_path, output_black_path=FLAGS.black_path,
                 output_white_path=FLAGS.white_path, output_height=FLAGS.input_height, output_width=FLAGS.output_width, input_fname_pattern=FLAGS.input_fname_pattern)

    # init the train data
    if FLAGS.train_black:
        train_data_black = Data(sampler_images=whole_black_path, standard_pic_dir=whole_standard_black_path, input_fname_pattern=FLAGS.input_fname_pattern,
                            config_file=FLAGS.config_file,  train_config_file=os.path.join(FLAGS.train_config_path, "format_0_train_config.txt"),
                            create_frain_config=FLAGS.is_create_train_config, batch=FLAGS.batch_size)

    if FLAGS.train_white:
        train_data_white = Data(sampler_images=whole_white_path, standard_pic_dir=whole_standard_white_path,
                                input_fname_pattern=FLAGS.input_fname_pattern,
                                config_file=FLAGS.config_file,
                                train_config_file=os.path.join(FLAGS.train_config_path, "format_255_train_config.txt"),
                                create_frain_config=FLAGS.is_create_train_config, batch=FLAGS.batch_size)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        model = LetterGenerator(
            sess, input_height=FLAGS.input_height, input_width=FLAGS.input_width,
            output_height=FLAGS.output_height, output_width=FLAGS.output_width, cnn_f_dim=64,
            batch_size=FLAGS.batch_size, fc_dim=1024, input_fname_patter=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir, generator_dir=FLAGS.generator_dir)
        show_all_variables()
        if FLAGS.train_black:
            model.train(config=FLAGS, train_data=train_data_black)
        if FLAGS.train_white:
            model.train(config=FLAGS, train_data=train_data_white)

if __name__ == '__main__':
    # tf.app.run()
    main()

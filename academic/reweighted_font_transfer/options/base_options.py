#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-06-09
#
# Author: jiean001
#########################################################


import argparse
import os
from utils.dir_util import mkdirs

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--use_tensorboardX', action='store_true', help='use tensorboardX to visiable')
        self.parser.add_argument('--ftX_comment', type=str, default='classifier_embedding_training', help='use tensorboardX to visiable')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')

        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--log_dir', type=str, default='log',
                                 help='name of the experiment. It decides where to store samples and models')

        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--dataroot', required=True, help='path to images')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--model', type=str, default='classifier',
                                 help='chooses which model to use. cycle_gan, one_direction_test, pix2pix, ...')
        self.parser.add_argument('--input_nc', type=int, default=26, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=26, help='# of output image channels')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--which_model_net_Classifier', type=str, default='basic', help='selects model to use for Classifier')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--classifier', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--config_dir', type=str, default='../config/',
                                 help='selects model to use for Classifier')
        self.parser.add_argument('--reweighted', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--isTrain', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--nThreads', default=6, type=int, help='# threads for loading data')
        self.parser.add_argument('--embedding_freq', default=20, type=int, help='# threads for loading data')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if self.opt.isTrain:
            self.opt.log_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.log_dir)
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        else:
            self.opt.log_dir = os.path.join(self.opt.results_dir, self.opt.log_dir)
            expr_dir = os.path.join(self.opt.results_dir, self.opt.name)

        # save to the disk
        mkdirs(expr_dir)
        mkdirs(self.opt.log_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


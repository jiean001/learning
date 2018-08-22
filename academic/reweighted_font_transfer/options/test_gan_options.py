#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-06-09
#
# Author: jiean001
#########################################################

from .test_options import TestOptions

# the option of Classifier
class TestGANOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument('--style_num', type=int, default=3, help='# of iter at starting learning rate')
        self.parser.add_argument('--constant_cos', type=int, default=2, help='# of iter at starting learning rate')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of iter at starting learning rate')
        self.parser.add_argument('--which_model_netG', type=str, default='reweighted_gan',
                                 help='selects model to use for Classifier')
        self.parser.add_argument('--which_model_preNet', type=str, default='2_layers',
                                 help='selects model to use for Classifier')
        self.parser.add_argument('--which_model_netD', type=str, default='n_layers',
                                 help='selects model to use for Classifier')
        self.parser.add_argument('--n_layers_D', type=int, default=3,
                                 help='selects model to use for Classifier')
        self.parser.add_argument('--use_sigmoid', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--postConv', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--no_lsgan', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--noisy_disc', action='store_true',
                                 help='add noise to the discriminator target labels')
        # 添加预训练的结果 LOADER_CLASSIFIER_EPOCH LOADER_CLASSIFIER_NAME LOADER_CLASSIFIER_NAME
        self.parser.add_argument('--loader_classifier_epoch', type=int, default=60, help='# of iter at starting learning rate')
        self.parser.add_argument('--loader_classifier_name', type=str, default='Classifier', help='# of iter at starting learning rate')
        self.parser.add_argument('--which_net_loader_classifier', type=str, default='0,1', help='# of iter at starting learning rate')
        self.parser.add_argument('--s_c_config', type=str, default='style_classifier.txt', help='# of iter at starting learning rate')
        self.parser.add_argument('--c_c_config', type=str, default='content_classifier.txt', help='# of iter at starting learning rate')
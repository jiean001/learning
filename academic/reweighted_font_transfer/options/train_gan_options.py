#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-06-09
#
# Author: jiean001
#########################################################

from .train_options import TrainOptions

# the option of Classifier
class TrainGANOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
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

        self.isTrain = True
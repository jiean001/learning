#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-09
#
# Author: jiean001
#
# 预训练分类器
#########################################################

from .base_options import BaseOptions

# the option of Classifier
class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--test_type', type=str, default='test_unseen', help='saves results here.')
        self.parser.add_argument('--results_dir', type=str, default='test', help='saves results here.')
        self.parser.add_argument('--which_epoch', type=str, default='15', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False
#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-09
#
# Author: jiean001
#
# 测试预训练分类器
#########################################################
import time

from options.test_options import TestOptions
opt = TestOptions().parse()

from model.models import create_model
from data.data_loader import CreateDataLoader
from utils.visualizer import *
from utils.tensorboard_util import TB_Visualizer

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

tb_v = TB_Visualizer(log_dir=opt.log_dir, comment=opt.ftX_comment, use_tensorboardX=opt.use_tensorboardX)

for i, data in enumerate(dataset['test']):
    iter_start_time = time.time()
    model.set_input(data)
    model.test()
    errors = model.get_current_errors()
    t = (time.time() - iter_start_time)
    print_current_errors(epoch=None, i=i, errors=errors, t=t)

    tb_v.add_loss(errors=errors, scalar_x=i)
    if opt.use_tensorboardX and i % opt.embedding_freq == 0:
        _out, _label, _input = model.get_embedding_para()
        tb_v.add_embedding(_out, _label, _input, i)
tb_v.close()

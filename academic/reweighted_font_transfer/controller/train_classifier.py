#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-06
#
# Author: jiean001
#
# 预训练分类器
#########################################################

import time
import sys
sys.path.append('..')

from options.train_options import TrainOptions
opt = TrainOptions().parse()

from model.models import create_model
from data.data_loader import CreateDataLoader
from utils.visualizer import *
from utils.tensorboard_util import TB_Visualizer

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
total_steps = 0
tb_v = TB_Visualizer(log_dir=opt.log_dir, comment='classifier_embedding_training', use_tensorboardX=opt.use_tensorboardX)

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset['train']):
        iter_start_time = time.time()
        model.set_input(data)
        model.optimize_parameters()

        errors = model.get_current_errors()
        t = (time.time() - iter_start_time)
        print_current_errors(epoch, i, errors, total_steps)

        tb_v.add_loss(errors=errors, scalar_x=total_steps)

        if opt.use_tensorboardX:
            _out, _label, _input = model.get_embedding_para()
            tb_v.add_embedding(_out, _label, _input, total_steps)

        if total_steps == 0:
            tb_v.add_graph(model=model.get_model(), dummy_input=model.get_input())

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
        total_steps += opt.batchSize

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
tb_v.close()
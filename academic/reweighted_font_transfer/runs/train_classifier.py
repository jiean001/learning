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
data_loader = CreateDataLoader(opt)

dataset = data_loader.load_data()

model = create_model(opt)
total_steps = 0


for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset['train']):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            print_current_errors(epoch, i, errors, t)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
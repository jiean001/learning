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
tb_v = TB_Visualizer(log_dir=opt.log_dir, comment=opt.ftX_comment, use_tensorboardX=opt.use_tensorboardX)
start_epoch = 1
# 继续训练
if opt.continue_train:
    start_epoch += int(opt.which_epoch)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset['train']):
        # print(data[1])
        # print('epoch: %d, iter: %d' %(epoch, i))
        iter_start_time = time.time()
        model.set_input(data)
        model.optimize_parameters()

        errors = model.get_current_errors()
        t = (time.time() - iter_start_time)
        print_current_errors(epoch=epoch, i=i, errors=errors, t=t)

        tb_v.add_loss(errors=errors, scalar_x=total_steps)

        # 保存的数据,所以每个epoch只保存一个
        # if opt.use_tensorboardX and i % opt.embedding_freq == 0:
        if opt.use_tensorboardX and i == 0:
            _out, _label, _input = model.get_embedding_para()
            tb_v.add_embedding(_out, _label, _input, total_steps)

        # 因为在多gpu的情况下，保存model会特别慢，所以multi-gpu下不保存模型
        if total_steps == -1:
            if len(opt.gpu_ids) == 1:
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

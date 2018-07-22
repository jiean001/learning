#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-11
#
# Author: jiean001
#
# 预训rew gan
#########################################################

import time

from options.train_gan_options import TrainGANOptions
opt = TrainGANOptions().parse()

from model.models import create_model
from data.data_loader import CreateDataLoader
from utils.visualizer import *
from utils.tensorboard_util import TB_Visualizer
from utils.img_util import *

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
tb_v = TB_Visualizer(log_dir=opt.log_dir, comment=opt.ftX_comment, use_tensorboardX=opt.use_tensorboardX)
start_epoch = 1
# 继续训练
if opt.continue_train:
    start_epoch += int(opt.which_epoch)
total_steps = (start_epoch - 1)*opt.batchSize

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter = epoch*opt.batchSize + i
        # 因为在多gpu的情况下，保存model会特别慢，所以multi-gpu下不保存模型
        if total_steps == -1:
            if len(opt.gpu_ids) == 1:
                tb_v.add_graph(model=model.get_model(), dummy_input=data)

        # print(data[1])
        # print('epoch: %d, iter: %d' %(epoch, i))
        iter_start_time = time.time()
        model.set_input(data)
        model.optimize_parameters()

        errors = model.get_current_errors()
        t = (time.time() - iter_start_time)
        print_current_errors(epoch=epoch, i=i, errors=errors, t=t)

        tb_v.add_loss(errors=errors, scalar_x=iter)
        if total_steps % 20 == 0:
            imgs = get_one_pair_imgs(data, model.get_crt_generate_img().data, model.get_crt_generate_img_b().data)
            tb_v.add_img(img=imgs, iter=iter)
        if total_steps % 30 == 0:
            print_imgs(data, '%s/%04d_%04d.png' % (model.get_save_imgs_dir(), epoch, i),
                              model.get_crt_generate_img().data, model.get_crt_generate_img_b().data)

        # 保存的数据,所以每个epoch只保存一个
        # if opt.use_tensorboardX and i % opt.embedding_freq == 0:
        if opt.use_tensorboardX and i == -1:
            _out, _label, _input = model.get_embedding_para()
            tb_v.add_embedding(_out, _label, _input, iter)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
        total_steps += 1

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
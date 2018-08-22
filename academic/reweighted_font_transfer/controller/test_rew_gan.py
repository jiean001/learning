#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-08-09
#
# Author: jiean001
#
# 测试rew gan
#########################################################

import time

from options.test_gan_options import TestGANOptions
opt = TestGANOptions().parse()

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

for i, data in enumerate(dataset):
    iter_start_time = time.time()
    model.set_input(data)
    model.test()
    errors = model.get_current_errors()
    t = (time.time() - iter_start_time)
    print_current_errors(epoch=0, i=i, errors=errors, t=t)
    tb_v.add_loss(errors=errors, scalar_x=i)
    print_imgs(data, '%s/%04d_%04d.png' % (model.get_save_imgs_dir(), i/13, i%13),
               model.get_crt_generate_img().data, model.get_crt_generate_img_b().data, batch_size=None)
    '''
    # save to tensorboard
    imgs = get_one_pair_imgs(data, model.get_crt_generate_img().data, model.get_crt_generate_img_b().data)[0]
    tb_v.add_img(img=imgs, iter=iter)
    '''
tb_v.close()

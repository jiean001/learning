#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-24
#
# Author: jiean001
#
# 预训rew gan
#########################################################
from utils.dir_util import *
from others.swap import *

from options.train_gan_options import TrainGANOptions
opt = TrainGANOptions().parse()

from model.models import create_model
from data.data_loader import CreateDataLoader
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

for inx, data in enumerate(dataset):
    iter_start_time = time.time()
    model.set_input(data)

    total_layer = 3
    for layer in range(total_layer):
        style_feature_map_rgb_list = model.get_feature_map(input_type=0, layer=layer)
        style_feature_map_b_list = model.get_feature_map(input_type=1, layer=layer)
        content_feature_map = model.get_content_feature_map(layer=layer)
        gt_feature_map = model.get_gt_feature_map(layer=layer)

        crt_path = './feature_map/layer_%d' %(3-layer)
        mkdirs(crt_path)

        print('---------------------------------------')
        j = 0
        for style_feature_map_rgb in style_feature_map_rgb_list:
            # print('style_feature_map_rgb', style_feature_map_rgb.size())
            print_img(style_feature_map_rgb.transpose(0, 1), '%s/style_rgb_%d_%d.png' %(crt_path, j, inx))
            j += 1

        j = 0
        for style_feature_map_b in style_feature_map_b_list:
            # print('style_feature_map_b', style_feature_map_b.size())
            print_img(style_feature_map_b.transpose(0, 1), '%s/style_b_%d_%d.png' % (crt_path, j, inx))
            j += 1
        # print('content_feature_map', content_feature_map.size())
        print_img(content_feature_map.transpose(0, 1), '%s/content_%d.png' % (crt_path, inx))
        # print('gt_feature_map', gt_feature_map.size())
        print_img(gt_feature_map.transpose(0, 1), '%s/gt_%d.png' % (crt_path, inx))

        lens = len(style_feature_map_b_list)

        # standard
        SC_style_feature_map = style_feature_map_b_list[0].unsqueeze(2)
        SC_style_feature_map_rgb = style_feature_map_rgb_list[0].unsqueeze(2)
        for i in range(1, lens):
            SC_style_feature_map = torch.cat((SC_style_feature_map, style_feature_map_b_list[i].unsqueeze(2)), 2)
            SC_style_feature_map_rgb = torch.cat((SC_style_feature_map_rgb, style_feature_map_rgb_list[i].unsqueeze(2)), 2)
        SC_content_feature_map = content_feature_map.unsqueeze(2)
        batch_size, channel, _, H, W = SC_style_feature_map.size()
        print(batch_size, channel, H, W, SC_style_feature_map.size(), SC_style_feature_map_rgb.size(), SC_content_feature_map.size())
        generate_fm = model.mixed_feature_map(SC_style_feature_map, SC_content_feature_map, SC_style_feature_map_rgb,
                                batch_size, channel, H, W)
        print_img(generate_fm.transpose(0, 1), '%s/generate_%d.png' % (crt_path, inx))
        print('%s/generate_%d.png' % (crt_path, inx))

        '''
        SC_style_feature_map = torch.cat((style_feature_map_b_list[0].unsqueeze(2), style_feature_map_b_list[1].unsqueeze(2),
                                          style_feature_map_b_list[2].unsqueeze(2)), 2)
        SC_style_feature_map_rgb = torch.cat((style_feature_map_rgb_list[0].unsqueeze(2), style_feature_map_rgb_list[1].unsqueeze(2),
                                              style_feature_map_rgb_list[2].unsqueeze(2)), 2)
        SC_content_feature_map = content_feature_map.unsqueeze(2)

        batch_size, channel, _, H, W = SC_style_feature_map.size()
        print(SC_style_feature_map.size(), SC_style_feature_map_rgb.size(), SC_content_feature_map.size())

        generate_fm = mixed_reweigted_feature_map(SC_style_feature_map, SC_content_feature_map, SC_style_feature_map_rgb,
                                batch_size, channel, H, W)
        print('generate_fm', generate_fm.size())
        print_img(generate_fm.transpose(0, 1), '%s/generate_L%d.png' % (crt_path, 3 - layer))
        '''

    print('----------------------------------------\n\n\n\n')

tb_v.close()
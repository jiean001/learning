#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-09
#
# Author: jiean001
#########################################################

import torch
import torch.nn as nn
from utils.img_util import *
import time
import torch.tensor as Variable

constant_cos = 2
batch_size = 1
channel = 3
H = 64
W = 64

img_A_RGB = rew_transform(default_img_loader(path=r'./000000/A.png'))
img_B_RGB = rew_transform(default_img_loader(path=r'./000000/B.png'))
img_F_RGB = rew_transform(default_img_loader(path=r'./000000/C.png'))
img_E_RGB = rew_transform(default_img_loader(path=r'./000001/E.png'))  # [0].unsqueeze(0)

img_A_B = get_binary_img(img_A_RGB)
img_B_B = get_binary_img(img_B_RGB)
img_F_B = get_binary_img(img_F_RGB)
img_E_B = get_binary_img(img_E_RGB)

print(type(img_E_RGB), img_E_RGB.size())  # (3, 64, 64)
SC_style_feature_map = torch.cat((img_A_B.unsqueeze(1), img_B_B.unsqueeze(1),
                                  img_F_B.unsqueeze(1)), 1).unsqueeze(0).cuda()
SC_style_feature_map_rgb = torch.cat((img_A_RGB.unsqueeze(1), img_B_RGB.unsqueeze(1),
                                      img_F_RGB.unsqueeze(1)), 1).unsqueeze(0).cuda()
SC_content_feature_map = img_E_B.unsqueeze(1).unsqueeze(0).cuda()

print(SC_style_feature_map.size(), SC_content_feature_map.size())

def mixed_reweigted_feature_map(SC_style_feature_map, SC_content_feature_map, SC_style_feature_map_rgb,
                                batch_size, channel, H, W):
    # print('SC_style_feature_map', SC_style_feature_map.size())  # (4, 576, 3, 16, 16)
    # print('SC_content_feature_map', SC_content_feature_map.size())  # (4, 576, 1, 16, 16)
    # print('SC_style_feature_map_rgb', SC_style_feature_map_rgb.size())  # (4, 576, 3, 16, 16)
    # print(batch_size, channel, H, W)  # (4, 576, 16, 16)

    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=H).cuda()
    softmax = nn.Softmax(dim=1).cuda()
    conv2d.bias.data.fill_(0.0)
    mixed_fm = None

    zeros = torch.zeros(1, H, W).cuda()

    for b in range(batch_size):
        # batch
        sc_s_fm = SC_style_feature_map[b]
        sc_c_fm = SC_content_feature_map[b]
        sc_s1_fm_rgb = SC_style_feature_map_rgb[b]

        mixed_fm_c = None
        for c in range(channel):
            # channel
            input = sc_s_fm[c].unsqueeze(1)
            sc_c_fm_c = sc_c_fm[c].unsqueeze(1)
            sc_s1_fm_rgb_c = sc_s1_fm_rgb[c]

            #  将权重设置为content
            conv2d.weight.data = sc_c_fm_c
            # 计算内积
            output_c = conv2d(input)
            output_c = output_c.squeeze(1).squeeze(1).transpose(0, 1)

            # 计算style的模长
            mode1 = torch.dist(input[0], zeros, 2).view(1, 1)
            mode2 = torch.dist(input[1], zeros, 2).view(1, 1)
            mode3 = torch.dist(input[2], zeros, 2).view(1, 1)
            mode = torch.cat((mode1, mode2, mode3), 1)
            # mode_content = torch.dist(sc_c_fm_c, zeros, 2).view(1, 1)
            # output_c /= mode_content
            # print(mode_content)
            #  乘以constant_cos*content模长的cos距离
            output_c = constant_cos * output_c / mode
            output_c = softmax(output_c)
            print(output_c)

            mixed_fm_c_tmp = sc_s1_fm_rgb_c[0] * output_c[0][0] + \
                             sc_s1_fm_rgb_c[1] * output_c[0][1] + sc_s1_fm_rgb_c[2] * output_c[0][2]
            if c == 0:
                mixed_fm_c = mixed_fm_c_tmp.unsqueeze(0)
            else:
                mixed_fm_c = torch.cat((mixed_fm_c, mixed_fm_c_tmp.unsqueeze(0)))
        mixed_fm_c = mixed_fm_c.unsqueeze(0)
        if b == 0:
            mixed_fm = mixed_fm_c
        else:
            mixed_fm = torch.cat((mixed_fm, mixed_fm_c))
    return mixed_fm

new_img = mixed_reweigted_feature_map(SC_style_feature_map, SC_content_feature_map, SC_style_feature_map_rgb,
                                batch_size, channel, H, W)

print_img(new_img, r'./E_new.png')
print(new_img.size())
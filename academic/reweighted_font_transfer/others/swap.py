#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-09
#
# Author: jiean001
#########################################################

import torch
import torch.nn as nn
import time
import torch.tensor as Variable

batch_size = 256
channel = 576
H = 8
W = 8

#  二值化的feature map
SC_style1_feature_map_b = torch.randn(batch_size, channel, H, W)
SC_style2_feature_map_b = torch.randn(batch_size, channel, H, W)
SC_style3_feature_map_b = torch.ones(batch_size, channel, H, W)
_SC_content_feature_map = torch.randn(batch_size, channel, H, W)

# RGB的feature map
SC_style1_feature_map_rgb = torch.randn(batch_size, channel, H, W)
SC_style2_feature_map_rgb = torch.randn(batch_size, channel, H, W)
SC_style3_feature_map_rgb = torch.ones(batch_size, channel, H, W)
constant_cos = 2

def mixed_reweigted_feature_map():
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=H)
    softmax = nn.Softmax(dim=1)
    conv2d.bias.data.fill_(0)
    mixed_fm = None

    start_time = time.time()
    SC_style_feature_map = torch.cat((SC_style1_feature_map_b.unsqueeze(2),
                                      SC_style2_feature_map_b.unsqueeze(2),
                                      SC_style3_feature_map_b.unsqueeze(2)), 2)
    SC_content_feature_map = _SC_content_feature_map.unsqueeze(2)
    SC_style_feature_map_rgb = torch.cat((SC_style1_feature_map_rgb.unsqueeze(2),
                                          SC_style2_feature_map_rgb.unsqueeze(2),
                                          SC_style3_feature_map_rgb.unsqueeze(2)), 2)

    zeros = torch.zeros(1, H, W)

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
            # print(sc_s1_fm_rgb_c.size())

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
    print(mixed_fm.size(), str(time.time() - start_time))

mixed_reweigted_feature_map()


'''
start_time = time.time()
for b in range(batch_size):
    # batch
    sc_s1_fm = SC_style1_feature_map[b].unsqueeze(1)
    sc_s2_fm = SC_style2_feature_map[b].unsqueeze(1)
    sc_s3_fm = SC_style3_feature_map[b].unsqueeze(1)
    sc_c_fm = SC_content_feature_map[b].unsqueeze(1)
    weight_c = None
    for c in range(channel):
        # channel
        sc_s1_fm_c = sc_s1_fm[c].unsqueeze(1)
        sc_s2_fm_c = sc_s2_fm[c].unsqueeze(1)
        sc_s3_fm_c = sc_s3_fm[c].unsqueeze(1)
        input = torch.cat((sc_s1_fm_c, sc_s2_fm_c, sc_s3_fm_c))
        sc_c_fm_c = sc_c_fm[c].unsqueeze(1)
        #  将权重设置为content
        conv2d.weight.data = sc_c_fm_c
        output_c = conv2d(input)
        output_c = output_c.squeeze(1).squeeze(1).transpose(0, 1)
        if c == 0:
            weight_c = output_c
        else:
            weight_c = torch.cat((weight_c, output_c))
    weight_c = weight_c.unsqueeze(0)
    if b == 0:
        weights = weight_c
    else:
        weights = torch.cat((weights, weight_c))
print(weights.size(), str(time.time() - start_time))
'''


#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-05
#
# Author: jiean001
#########################################################
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


# 初始化权重
def weights_init(m):
    classname = m.__class__.__name__
    print("classname",classname)
    if classname.find('Conv') != -1:
        print("in random conv")
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        print("in random batchnorm")
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# norm的方式
def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        norm_layer = None
        print('normalization layer [%s] is not found' %(norm_type))
    return norm_layer


def conv_norm_relu_module(norm_layer, input_nc, ngf, kernel_size, padding, stride=1, relu='relu'):
    model = [nn.Conv2d(input_nc, ngf, kernel_size=kernel_size, padding=padding,stride=stride)]
    if norm_layer:
        model += [norm_layer(ngf)]

    if relu == 'relu':
        model += [nn.ReLU(True)]
    elif relu == 'Lrelu':
        model += [nn.LeakyReLU(0.2, True)]
    return model


def fc_module(input_nc, output_nc):
    # model = [nn.Linear(input_nc, output_nc), nn.Softmax()]
    model = [nn.Linear(int(input_nc), int(output_nc))]
    return model


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, norm_type='batch'):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, norm_type)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, norm_type):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert (padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm

        conv_block += conv_norm_relu_module(norm_layer, dim, dim, 3, p)
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0.0)]

        if norm_type == 'batch' or norm_type == 'instance':
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                           norm_layer(dim)]
        else:
            assert ("norm not defined")

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Classifier_letter(nn.Module):
    def __init__(self, input_nc=3, output_nc=26, ngf=64, norm='batch', use_dropout=False, gpu_ids=[]):
        if gpu_ids:
            use_gpu = len(gpu_ids) > 0
        else:
            use_gpu = False
        self.norm_layer = get_norm_layer(norm_type=norm)

        if use_gpu:
            assert (torch.cuda.is_available())

        super(Classifier_letter, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_dropout = use_dropout

        self.CNN_layer()
        self.fully_connection_layer()

    # 3 --> 64 --> 64*3 --> 64*9
    def CNN_layer(self):
        factor_ch = 3
        self.layer_1 = nn.Sequential(*conv_norm_relu_module(self.norm_layer, self.input_nc, self.ngf, 7, 3))

        mult = factor_ch ** 0
        self.layer_2 = nn.Sequential(
            *conv_norm_relu_module(self.norm_layer, self.ngf * mult, self.ngf * mult * factor_ch, 3, 1, stride=2))

        mult = factor_ch ** 1
        self.layer_3 = nn.Sequential(
            *conv_norm_relu_module(self.norm_layer, self.ngf * mult, self.ngf * mult * factor_ch, 3, 1, stride=2))

        n_downsampling = 2
        mult = factor_ch ** n_downsampling

        self.layer_4_1 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.layer_4_2 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.layer_4_3 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.layer_4_4 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.layer_4_5 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.layer_4_6 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))

    def fully_connection_layer(self):
        self.fc_layer = nn.Sequential(*fc_module(64*3*3*((64/2/2)**2), self.output_nc))

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            layer_1 = nn.parallel.data_parallel(self.layer_1, input, self.gpu_ids)
            layer_2 = nn.parallel.data_parallel(self.layer_2, layer_1, self.gpu_ids)
            layer_3 = nn.parallel.data_parallel(self.layer_3, layer_2, self.gpu_ids)
            layer_4_res_1 = nn.parallel.data_parallel(self.layer_4_1, layer_3, self.gpu_ids)
            layer_4_res_2 = nn.parallel.data_parallel(self.layer_4_2, layer_4_res_1, self.gpu_ids)
            layer_4_res_3 = nn.parallel.data_parallel(self.layer_4_3, layer_4_res_2, self.gpu_ids)
            layer_4_res_4 = nn.parallel.data_parallel(self.layer_4_4, layer_4_res_3, self.gpu_ids)
            layer_4_res_5 = nn.parallel.data_parallel(self.layer_4_5, layer_4_res_4, self.gpu_ids)
            layer_4_res_6 = nn.parallel.data_parallel(self.layer_4_6, layer_4_res_5, self.gpu_ids)
            fc_input = layer_4_res_6.view(layer_4_res_6.size(0), -1)
            out = nn.parallel.data_parallel(self.fc_layer, fc_input, self.gpu_ids)
        else:
            layer_1 = self.layer_1(input)
            layer_2 = self.layer_2(layer_1)
            layer_3 = self.layer_3(layer_2)
            layer_4_res_1 = self.layer_4_1(layer_3)
            layer_4_res_2 = self.layer_4_2(layer_4_res_1)
            layer_4_res_3 = self.layer_4_3(layer_4_res_2)
            layer_4_res_4 = self.layer_4_4(layer_4_res_3)
            layer_4_res_5 = self.layer_4_5(layer_4_res_4)
            layer_4_res_6 = self.layer_4_6(layer_4_res_5)
            fc_input = layer_4_res_6.view(layer_4_res_6.size(0), -1)
            out = self.fc_layer(fc_input)
        return F.log_softmax(out)
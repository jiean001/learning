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
from utils.img_util import *

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


# 卷积层block
def conv_norm_relu_module(norm_layer, input_nc, ngf, kernel_size, padding, stride=1, relu='relu'):
    model = [nn.Conv2d(input_nc, ngf, kernel_size=kernel_size, padding=padding,stride=stride)]
    if norm_layer:
        model += [norm_layer(ngf)]

    if relu == 'relu':
        model += [nn.ReLU(True)]
    elif relu == 'Lrelu':
        model += [nn.LeakyReLU(0.2, True)]
    return model


# deconv block
def convTranspose_norm_relu_module(norm_layer, input_nc, ngf, kernel_size, padding, stride=1, output_padding=0):
    model = [nn.ConvTranspose2d(input_nc, ngf,
                                kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
             norm_layer(int(ngf)),
             nn.ReLU(True)]
    return model


# 全连接层
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


#  分类网
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


# 生成器
class Generator_Reweighted(nn.Module):
    def __init__(self, input_nc=3, output_nc=4, ngf=64, norm='batch', use_dropout=False,
                 gpu_ids=[], constant_cos=2):

        use_gpu = len(gpu_ids) > 0
        self.norm_layer = get_norm_layer(norm_type=norm)

        if use_gpu:
            assert (torch.cuda.is_available())

        super(Generator_Reweighted, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_dropout = use_dropout
        self.norm = norm
        self.constant_cos = constant_cos

        self.Extract_Style_Feature()
        self.Extract_Content_Feature()
        self.Decode_Feature_Map()

    # 3 --> 64 --> 64*3 --> 64*9
    def Extract_Style_Feature(self):
        factor_ch = 3
        self.S_layer_1 = nn.Sequential(*conv_norm_relu_module(self.norm_layer, self.input_nc, self.ngf, 7, 3))
        mult = factor_ch ** 0
        self.S_layer_2 = nn.Sequential(
            *conv_norm_relu_module(self.norm_layer, self.ngf * mult, self.ngf * mult * factor_ch, 3, 1, stride=2))
        mult = factor_ch ** 1
        self.S_layer_3 = nn.Sequential(
            *conv_norm_relu_module(self.norm_layer, self.ngf * mult, self.ngf * mult * factor_ch, 3, 1, stride=2))
        n_downsampling = 2
        mult = factor_ch ** n_downsampling
        self.S_layer_4_res_1 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.S_layer_4_res_2 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.S_layer_4_res_3 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.S_layer_4_res_4 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.S_layer_4_res_5 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.S_layer_4_res_6 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))

    def Extract_Content_Feature(self):
        factor_ch = 3
        self.C_layer_1 = nn.Sequential(*conv_norm_relu_module(self.norm_layer, self.input_nc, self.ngf, 7, 3))
        mult = factor_ch ** 0
        self.C_layer_2 = nn.Sequential(
            *conv_norm_relu_module(self.norm_layer, self.ngf * mult, self.ngf * mult * factor_ch, 3, 1, stride=2))
        mult = factor_ch ** 1
        self.C_layer_3 = nn.Sequential(
            *conv_norm_relu_module(self.norm_layer, self.ngf * mult, self.ngf * mult * factor_ch, 3, 1, stride=2))
        n_downsampling = 2
        mult = factor_ch ** n_downsampling
        self.C_layer_4_res_1 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.C_layer_4_res_2 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.C_layer_4_res_3 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.C_layer_4_res_4 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.C_layer_4_res_5 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))
        self.C_layer_4_res_6 = nn.Sequential(
            ResnetBlock(self.ngf * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.norm_layer))

    def Decode_Feature_Map(self):
        n_downsampling = 2
        factor_ch = 3
        mult = factor_ch ** (n_downsampling - 0)
        self.SC_layer_5 = nn.Sequential(
            *convTranspose_norm_relu_module(self.norm_layer, self.ngf * mult * 2, int(self.ngf * mult / factor_ch), 3,
                                            1,
                                            stride=2, output_padding=1))
        mult = factor_ch ** (n_downsampling - 1)
        self.SC_layer_6 = nn.Sequential(
            *convTranspose_norm_relu_module(self.norm_layer, self.ngf * mult * 2, int(self.ngf * mult / factor_ch), 3,
                                            1,
                                            stride=2, output_padding=1))
        self.SC_layer_7 = nn.Sequential(nn.Conv2d(self.ngf*2, self.output_nc, kernel_size=7, padding=3), nn.Tanh())

    def Mix_S_C_Feature(self, Style_Feature, Content_feature):
        ret = torch.cat((Content_feature, Style_Feature), dim=1)
        return ret

    def mixed_reweigted_feature_map(self, SC_style_feature_map, SC_content_feature_map, SC_style_feature_map_rgb,
                                    batch_size, channel, H, W):
        conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=H)
        softmax = nn.Softmax(dim=1)
        conv2d.bias.data.fill_(0)
        mixed_fm = None

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
                output_c = self.constant_cos * output_c / mode
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
        return mixed_fm

    # input_style : [batch, channel, H, W]
    def forward_Style(self, input_style):
        if self.gpu_ids and isinstance(input_style.data, torch.cuda.FloatTensor):
            layer_1 = nn.parallel.data_parallel(self.S_layer_1, input_style, self.gpu_ids)  # 6
            layer_2 = nn.parallel.data_parallel(self.S_layer_2, layer_1, self.gpu_ids)  # 5
            layer_3 = nn.parallel.data_parallel(self.S_layer_3, layer_2, self.gpu_ids)
            layer_4_res_1 = nn.parallel.data_parallel(self.S_layer_4_res_1, layer_3, self.gpu_ids)
            layer_4_res_2 = nn.parallel.data_parallel(self.S_layer_4_res_2, layer_4_res_1, self.gpu_ids)
            layer_4_res_3 = nn.parallel.data_parallel(self.S_layer_4_res_3, layer_4_res_2, self.gpu_ids)
            layer_4_res_4 = nn.parallel.data_parallel(self.S_layer_4_res_4, layer_4_res_3, self.gpu_ids)
            layer_4_res_5 = nn.parallel.data_parallel(self.S_layer_4_res_5, layer_4_res_4, self.gpu_ids)
            layer_4_res_6 = nn.parallel.data_parallel(self.S_layer_4_res_6, layer_4_res_5, self.gpu_ids)
        else:
            layer_1 = self.S_layer_1(input_style)
            layer_2 = self.S_layer_2layer_2(layer_1)
            layer_3 = self.S_layer_3layer_3(layer_2)
            layer_4_res_1 = self.S_layer_4_res_1(layer_3)
            layer_4_res_2 = self.S_layer_4_res_2(layer_4_res_1)
            layer_4_res_3 = self.S_layer_4_res_3(layer_4_res_2)
            layer_4_res_4 = self.S_layer_4_res_4(layer_4_res_3)
            layer_4_res_5 = self.S_layer_4_res_5(layer_4_res_4)
            layer_4_res_6 = self.S_layer_4_res_6(layer_4_res_5)
        return layer_4_res_6, layer_2, layer_1

    # input_content : [batch, channel, H, W]
    def forward_Content(self, input_content):
        if self.gpu_ids and isinstance(input_content.data, torch.cuda.FloatTensor):
            layer_1 = nn.parallel.data_parallel(self.C_layer_1, input_content, self.gpu_ids)
            layer_2 = nn.parallel.data_parallel(self.C_layer_2, layer_1, self.gpu_ids)
            layer_3 = nn.parallel.data_parallel(self.C_layer_3, layer_2, self.gpu_ids)
            layer_4_res_1 = nn.parallel.data_parallel(self.C_layer_4_res_1, layer_3, self.gpu_ids)
            layer_4_res_2 = nn.parallel.data_parallel(self.C_layer_4_res_2, layer_4_res_1, self.gpu_ids)
            layer_4_res_3 = nn.parallel.data_parallel(self.C_layer_4_res_3, layer_4_res_2, self.gpu_ids)
            layer_4_res_4 = nn.parallel.data_parallel(self.C_layer_4_res_4, layer_4_res_3, self.gpu_ids)
            layer_4_res_5 = nn.parallel.data_parallel(self.C_layer_4_res_5, layer_4_res_4, self.gpu_ids)
            layer_4_res_6 = nn.parallel.data_parallel(self.C_layer_4_res_6, layer_4_res_5, self.gpu_ids)
        else:
            layer_1 = self.C_layer_1(input_content)
            layer_2 = self.C_layer_2layer_2(layer_1)
            layer_3 = self.C_layer_3layer_3(layer_2)
            layer_4_res_1 = self.C_layer_4_res_1(layer_3)
            layer_4_res_2 = self.C_layer_4_res_2(layer_4_res_1)
            layer_4_res_3 = self.C_layer_4_res_3(layer_4_res_2)
            layer_4_res_4 = self.C_layer_4_res_4(layer_4_res_3)
            layer_4_res_5 = self.C_layer_4_res_5(layer_4_res_4)
            layer_4_res_6 = self.C_layer_4_res_6(layer_4_res_5)
        return layer_4_res_6, layer_2, layer_1

    # input_content : [batch, channel, H, W]
    # input_style : [batch, num, channel, H, W]
    def forward(self, input_style, input_content):
        # style rgb
        input_style = input_style.transpose(0, 1)
        num, batch_size, channel, H, W = input_style.size()
        # todo
        assert num == 3, 'style number != 3'
        SC_style1_feature_map_rgb, _, _ = self.forward_Style(input_style=input_style[0])
        SC_style2_feature_map_rgb, _, _ = self.forward_Style(input_style=input_style[1])
        SC_style3_feature_map_rgb, _, _ = self.forward_Style(input_style=input_style[2])
        # style binary
        input_style_b = get_binary_img(input_style)
        SC_style1_feature_map_b, _, _ = self.forward_Style(input_style=input_style_b[0].detach())
        SC_style2_feature_map_b, _, _ = self.forward_Style(input_style=input_style_b[1].detach())
        SC_style3_feature_map_b, _, _ = self.forward_Style(input_style=input_style_b[2].detach())
        # style_content binary
        input_content = get_binary_img(input_content)
        _SC_content_feature_map, _, _ = self.forward_Style(input_style=input_content.detach())
        # content binary
        C_content_feature_map, C_l2, C_l1 = self.forward_Content(input_style=input_content)

        SC_style_feature_map = torch.cat((SC_style1_feature_map_b.unsqueeze(2),
                                      SC_style2_feature_map_b.unsqueeze(2),
                                      SC_style3_feature_map_b.unsqueeze(2)), 2)
        SC_content_feature_map = _SC_content_feature_map.unsqueeze(2)
        SC_style_feature_map_rgb = torch.cat((SC_style1_feature_map_rgb.unsqueeze(2),
                                              SC_style2_feature_map_rgb.unsqueeze(2),
                                              SC_style3_feature_map_rgb.unsqueeze(2)), 2)

        reweigthed_fm = self.mixed_reweigted_feature_map(SC_style_feature_map=SC_style_feature_map,
                                                         SC_content_feature_map=SC_content_feature_map,
                                                         SC_style_feature_map_rgb=SC_style_feature_map_rgb,
                                                         batch_size=batch_size, channel=channel, H=H, W=W)

        mixed_feature = self.Mix_S_C_Feature(reweigthed_fm, C_content_feature_map)

        if self.gpu_ids and isinstance(input_content.data, torch.cuda.FloatTensor) and isinstance(input_style.data, torch.cuda.FloatTensor):
            layer_5 = nn.parallel.data_parallel(self.layer_5, mixed_feature, self.gpu_ids)
            mixed_feature_2 = self.Mix_S_C_Feature(layer_5, C_l2)
            layer_6 = nn.parallel.data_parallel(self.layer_6, mixed_feature_2, self.gpu_ids)
            mixed_feature_1 = self.Mix_S_C_Feature(layer_6, C_l1)
            layer_7 = nn.parallel.data_parallel(self.layer_7, mixed_feature_1, self.gpu_ids)
        else:
            layer_5 = self.layer_5(mixed_feature)
            mixed_feature_2 = self.Mix_S_C_Feature(layer_5, C_l2)
            layer_6 = self.layer_6(mixed_feature_2)
            mixed_feature_1 = self.Mix_S_C_Feature(layer_6, C_l1)
            layer_7 = self.layer_7(mixed_feature_1)
        out = layer_7.transpose(0, 1)
        generate_rgb = out[0:3].transpose(0, 1)
        generate_b = out[3:].transpose(0, 1)
        return generate_rgb, generate_b


##Apply a transformation on the input and prediction before feeding into the discriminator
## in the conditional case
class InputTransformation(nn.Module):
    def __init__(self, input_nc, nif=32, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(InputTransformation, self).__init__()
        self.gpu_ids = gpu_ids
        use_gpu = len(gpu_ids) > 0
        if use_gpu:
            assert (torch.cuda.is_available())

        sequence = [nn.Conv2d(input_nc, nif, kernel_size=3, stride=2, padding=1),
                    norm_layer(nif),
                    nn.ReLU(True)]
        sequence += [nn.Conv2d(nif, nif, kernel_size=3, stride=2, padding=1),
                     norm_layer(nif),
                     nn.ReLU(True)]
        self.model = nn.Sequential(*sequence)
    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# D pre
def define_preNet(input_nc, nif=32, which_model_preNet='2_layers', norm='batch', gpu_ids=[]):
    preNet = None
    norm_layer = get_norm_layer(norm_type=norm)
    use_gpu = len(gpu_ids) > 0
    if which_model_preNet == '2_layers':
        print("2 layers convolution applied before being fed into the discriminator")
        preNet = InputTransformation(input_nc, nif, norm_layer, gpu_ids)
        if use_gpu:
            assert(torch.cuda.is_available())
            preNet.cuda(device=gpu_ids[0])
        preNet.apply(weights_init)
    return preNet


# D, patchGan
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=15, ndf=64, n_layers=3, is_RGB=True, norm_layer=nn.BatchNorm2d, use_sigmoid=False, norm_type='batch',
                 postConv=True, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.is_RGB = is_RGB

        kw = 5
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += conv_norm_relu_module(norm_type, norm_layer, ndf * nf_mult_prev,
                                              ndf * nf_mult, kw, padw, stride=2, relu='Lrelu')

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += conv_norm_relu_module(norm_type, norm_layer, ndf * nf_mult_prev,
                                          ndf * nf_mult, kw, padw, stride=1, relu='Lrelu')

        if postConv:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

            if use_sigmoid:
                sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    # input:[batch, count, channel, hight, width]
    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

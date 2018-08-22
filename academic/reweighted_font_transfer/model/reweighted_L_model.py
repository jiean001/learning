#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-24
#
# Author: jiean001
#
# 只用L1和MSE,不用GAN
#########################################################

import random
from collections import OrderedDict

import torch
from torch.autograd import Variable

from . import network_controller
from .base_model import BaseModel
from utils.img_util import *
from utils.dir_util import *
import os


class Reweighted_L(BaseModel):
    def name(self):
        return 'Reweighted_L'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.save_imgs_dir = os.path.join(self.save_dir, 'imgs')
        mkdir(self.save_imgs_dir)

        self.isTrain = opt.isTrain

        self.input_style = self.Tensor(opt.batchSize, opt.style_num, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_content = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_gt = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_standard = self.Tensor(opt.batchSize, opt.style_num, opt.input_nc, opt.fineSize, opt.fineSize)
        self.has_standard = False

        self.netG = network_controller.define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf,
                                                norm=opt.norm, use_dropout=opt.use_dropout, gpu_ids=self.gpu_ids,
                                                which_model_netG=opt.which_model_netG, constant_cos=opt.constant_cos)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
        else:
            if int(opt.which_epoch) > 0:
                self.load_network(self.netG, 'G', opt.which_epoch, print_weights=True)
            else:
                # load style
                if opt.which_net_loader_classifier.find('0') != -1:
                    config_file = os.path.join(opt.config_dir, opt.s_c_config)
                    self.load_network(network=self.netG, network_label=opt.loader_classifier_name, epoch_label=opt.loader_classifier_epoch,
                                      config_file=config_file, print_weights=True)

                if opt.which_net_loader_classifier.find('1') != -1:
                    config_file = os.path.join(opt.config_dir, opt.c_c_config)
                    self.load_network(network=self.netG, network_label=opt.loader_classifier_name,
                                      epoch_label=opt.loader_classifier_epoch, config_file=config_file, print_weights=True)

        self.criterionL1 = torch.nn.L1Loss()
        self.MSELoss = torch.nn.MSELoss()
        if self.isTrain:
            self.old_lr = opt.lr
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            # network_controller.print_network(self.netG)
            print('-----------------------------------------------')

    def set_input(self, input):
        input_style = input['style_imgs']
        input_content = input['content_imgs']
        input_gt = input['gt_img']

        self.input_style.resize_(input_style.size()).copy_(input_style)
        self.input_content.resize_(input_content.size()).copy_(input_content)
        self.input_gt.resize_(input_gt.size()).copy_(input_gt)

        if 'standard_imgs' in input.keys():
            self.has_standard = True
            input_standard = input['standard_imgs']
            self.input_standard.resize_(input_standard.size()).copy_(input_standard)

    def forward(self):
        self._input_style = Variable(self.input_style)
        self._input_content = Variable(self.input_content)
        self._input_gt = Variable(self.input_gt, volatile=True)

        if self.has_standard:
            self._input_standard = Variable(self.input_standard, volatile=True)
            self.generate_letter, self.generate_letter_b = self.netG.forward(self._input_style, self._input_content, self._input_standard)
        else:
            self.generate_letter, self.generate_letter_b = self.netG.forward(self._input_style, self._input_content)

    # no backprop gradients
    def test(self):
        self._input_style = Variable(self.input_style, volatile=True)
        self._input_content = Variable(self.input_content, volatile=True)
        self._input_gt = Variable(self.input_gt, volatile=True)

        if self.has_standard:
            self._input_standard = Variable(self.input_standard, volatile=True)
            self.generate_letter, self.generate_letter_b = self.netG.forward(self._input_style, self._input_content, self._input_standard)
        else:
            self.generate_letter, self.generate_letter_b = self.netG.forward(self._input_style, self._input_content)

        self.loss_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
        self.loss_G_MSE = self.MSELoss(self.generate_letter, self._input_gt)

        self.loss_G_B_L1 = self.criterionL1(self.generate_letter_b,
                                            get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(0).transpose(0,
                                                                                                                     1))
        self.loss_G_B_MSE = self.MSELoss(self.generate_letter_b,
                                         get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(0).transpose(0, 1))

    def backward_G(self):
        self.loss_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
        self.loss_G_MSE = self.MSELoss(self.generate_letter, self._input_gt)

        self.loss_G_B_L1 = self.criterionL1(self.generate_letter_b, get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(0).transpose(0, 1))
        self.loss_G_B_MSE = self.MSELoss(self.generate_letter_b, get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(0).transpose(0, 1))

        # print(self.ge)
        self.loss_G = 1.0 * (self.loss_G_B_L1 + self.loss_G_B_MSE) + 1.0 * (self.loss_G_L1 + self.loss_G_MSE)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('loss_G_L1', self.loss_G_L1.data.item()),
                            ('loss_G_MSE', self.loss_G_MSE.data.item()),
                            ('loss_G_B_MSE', self.loss_G_B_MSE.data.item()),
                            ('loss_G_B_L1', self.loss_G_B_L1.data.item())
        ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, gpu_ids=self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def get_crt_generate_img(self):
        return self.generate_letter

    def get_crt_generate_img_b(self):
        return self.generate_letter_b

    def get_model(self):
        return self.netG

    def get_save_imgs_dir(self):
        return self.save_imgs_dir

    # 可视化feature map
    def get_feature_map(self, input_type, layer=1):
        if input_type == 0:  # style
            self._input = Variable(self.input_style, volatile=True)
        elif input_type == 1:  # standard
            self._input = Variable(self.input_standard, volatile=True)
        _input = self._input.transpose(0, 1)
        num, _, _, _, _ = _input.size()
        fm = [self.netG.forward_Style(input_style=_input[0])[layer]]
        for i in range(1, num):
            fm.append(self.netG.forward_Style(input_style=_input[i])[layer])
        return fm

    def get_content_feature_map(self, layer=0):
        self._input_content = Variable(self.input_content, volatile=True)
        input_content = get_binary_img(self._input_content)
        content_feture_map = self.netG.forward_Style(input_style=input_content.detach())
        return content_feture_map[layer]

    def get_gt_feature_map(self, layer=0):
        self._input_content = Variable(self.input_gt, volatile=True)
        input_content = get_binary_img(self._input_content)
        content_feture_map = self.netG.forward_Style(input_style=input_content.detach())
        return content_feture_map[layer]

    def mixed_feature_map(self, SC_style_feature_map, SC_content_feature_map, SC_style_feature_map_rgb,
                                    batch_size, channel, H, W):
        return self.netG.mixed_reweigted_feature_map(SC_style_feature_map, SC_content_feature_map, SC_style_feature_map_rgb,
                                    batch_size, channel, H, W)
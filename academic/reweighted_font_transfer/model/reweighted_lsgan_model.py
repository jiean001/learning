#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-08-10
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


class Reweighted_LSGAN(BaseModel):
    def name(self):
        return 'Reweighted_LSGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.save_imgs_dir = os.path.join(self.save_dir, 'imgs')
        mkdir(self.save_imgs_dir)

        self.isTrain = opt.isTrain
        # 控制loss
        self.use_gan = opt.use_gan
        self.D_B = opt.D_B
        self.D_RGB = opt.D_RGB

        self.input_style = self.Tensor(opt.batchSize, opt.style_num, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_content = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_gt = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_standard = self.Tensor(opt.batchSize, opt.style_num, opt.input_nc, opt.fineSize, opt.fineSize)
        self.has_standard = False

        self.netG = network_controller.define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf,
                                                norm=opt.norm, use_dropout=opt.use_dropout, gpu_ids=self.gpu_ids,
                                                which_model_netG=opt.which_model_netG, constant_cos=opt.constant_cos)
        if self.use_gan:
            D_input_channel = (int(opt.style_img_num) + 1 + 1) * 3
            if self.D_RGB:
                # RGB
                self.preNet_RGB = network_controller.define_preNet(D_input_channel, D_input_channel, gpu_ids=self.gpu_ids)
                self.netD_RGB = network_controller.define_D(input_nc=D_input_channel, ndf=opt.ndf,
                                                                which_model_netD=opt.which_model_netD,
                                                                is_RGB=True, n_layers_D=opt.n_layers_D,
                                                                use_sigmoid=opt.use_sigmoid, postConv=True, gpu_ids=self.gpu_ids)
            if self.D_B:
                # BINARY
                self.preNet_Binary = network_controller.define_preNet(D_input_channel, D_input_channel, gpu_ids=self.gpu_ids)
                self.netD_Binary = network_controller.define_D(input_nc=D_input_channel, ndf=opt.ndf,
                                                               which_model_netD=opt.which_model_netD,
                                                               is_RGB=False, n_layers_D=opt.n_layers_D,
                                                               use_sigmoid=opt.use_sigmoid, postConv=True, gpu_ids=self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
        else:
            if int(opt.which_epoch_D) > 0:
                self.load_network(self.netG, 'G', opt.which_epoch_D, print_weights=False)
                if self.D_B:
                    self.load_network(self.netD_Binary, 'D_B', opt.which_epoch_D)
                    self.load_network(self.preNet_Binary, 'PRE_Binary', opt.which_epoch_D)
                if self.D_RGB:
                    self.load_network(self.preNet_RGB, 'PRE_RGB', opt.which_epoch_D)
                    self.load_network(self.netD_RGB, 'D_RGB', opt.which_epoch_D)

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
        self.criterionGAN = network_controller.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

        if self.isTrain:
            self.old_lr = opt.lr
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            network_controller.print_network(self.netG)
            if self.D_B:
                self.optimizer_pre_Binary = torch.optim.Adam(self.preNet_Binary.parameters(), lr=opt.lr,
                                                             betas=(opt.beta1, 0.999))
                self.optimizer_D_Binary = torch.optim.Adam(self.netD_Binary.parameters(), lr=opt.lr,
                                                           betas=(opt.beta1, 0.999))
                network_controller.print_network(self.preNet_Binary)
                network_controller.print_network(self.netD_Binary)
            if self.D_RGB:
                self.optimizer_D_RGB = torch.optim.Adam(self.netD_RGB.parameters(), lr=opt.lr,
                                                        betas=(opt.beta1, 0.999))
                self.optimizer_pre_RGB = torch.optim.Adam(self.preNet_RGB.parameters(), lr=opt.lr,
                                                          betas=(opt.beta1, 0.999))
                network_controller.print_network(self.preNet_RGB)
                network_controller.print_network(self.netD_RGB)

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

        if self.use_gan:
            self.loss_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
            self.loss_G_B_L1 = self.criterionL1(self.generate_letter_b,
                                                get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(
                                                    0).transpose(0, 1))

            # 假的当真的
            if self.D_RGB:
                fake_AB = self.get_D_input(self.generate_letter, is_Binary=False)
                pred_fake_patch = self.netD_RGB.forward(fake_AB)
                self.loss_G_GAN_RGB = self.criterionGAN(pred_fake_patch, True)
                transformed_A = self.preNet_RGB.forward(fake_AB)
                pred_fake = self.netD_RGB.forward(transformed_A)
                self.loss_G_GAN_RGB += self.criterionGAN(pred_fake, True)

            if self.D_B:
                fake_AB_2 = self.get_D_input(self.generate_letter_b, is_Binary=True)
                pred_fake_patch_2 = self.netD_Binary.forward(fake_AB_2)
                self.loss_G_GAN_BINARY = self.criterionGAN(pred_fake_patch_2, True)
                transformed_A_2 = self.preNet_RGB.forward(fake_AB_2)
                pred_fake_2 = self.netD_Binary.forward(transformed_A_2)
                self.loss_G_GAN_BINARY += self.criterionGAN(pred_fake_2, True)

            if self.D_RGB and self.D_B:
                self.loss_G = (self.loss_G_L1 + self.loss_G_B_L1) * 0.5 + (self.loss_G_GAN_RGB+self.loss_G_GAN_BINARY)*0.5
            elif self.D_B:
                self.loss_G = (self.loss_G_L1 + self.loss_G_B_L1) * 0.5 + (
                            0 + self.loss_G_GAN_BINARY) * 0.5
            else:  # only self.D_RGB
                self.loss_G = (self.loss_G_L1 + self.loss_G_B_L1) * 0.5 + (
                            self.loss_G_GAN_RGB + 0) * 0.5
        else:
            self.loss_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
            self.loss_G_MSE = self.MSELoss(self.generate_letter, self._input_gt)

            self.loss_G_B_L1 = self.criterionL1(self.generate_letter_b, get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(0).transpose(0, 1))
            self.loss_G_B_MSE = self.MSELoss(self.generate_letter_b, get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(0).transpose(0, 1))
            self.loss_G = 1.0 * (self.loss_G_B_L1 + self.loss_G_B_MSE) + 1.0 * (self.loss_G_L1 + self.loss_G_MSE)

    def backward_G(self):
        if self.use_gan:
            self.loss_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
            self.loss_G_B_L1 = self.criterionL1(self.generate_letter_b,
                                                get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(
                                                    0).transpose(0, 1))
            # 假的当真的
            if self.D_RGB:
                fake_AB = self.get_D_input(self.generate_letter, is_Binary=False)
                pred_fake_patch = self.netD_RGB.forward(fake_AB)
                self.loss_G_GAN_RGB = self.criterionGAN(pred_fake_patch, True) * 0.5
                transformed_A = self.preNet_RGB.forward(fake_AB)
                pred_fake = self.netD_RGB.forward(transformed_A)
                self.loss_G_GAN_RGB += self.criterionGAN(pred_fake, True) * 0.5

            if self.D_B:
                fake_AB_2 = self.get_D_input(self.generate_letter_b, is_Binary=True)
                # print(fake_AB_2.size(), fake_AB.size())
                pred_fake_patch_2 = self.netD_Binary.forward(fake_AB_2)
                self.loss_G_GAN_BINARY = self.criterionGAN(pred_fake_patch_2, True) * 0.5
                transformed_A_2 = self.preNet_RGB.forward(fake_AB_2)
                pred_fake_2 = self.netD_Binary.forward(transformed_A_2)
                self.loss_G_GAN_BINARY += self.criterionGAN(pred_fake_2, True) * 0.5

            if self.D_RGB and self.D_B:
                self.loss_G = ((self.loss_G_L1 + self.loss_G_B_L1) * 0.5 + (self.loss_G_GAN_RGB + self.loss_G_GAN_BINARY) * 0.5) * 0.5
            elif self.D_B:
                self.loss_G = ((self.loss_G_L1 + self.loss_G_B_L1) * 0.5 + (
                            0 + self.loss_G_GAN_BINARY) * 1) * 0.5
            else:  # only self.D_RGB
                self.loss_G = ((self.loss_G_L1 + self.loss_G_B_L1) * 0.5 + (
                            self.loss_G_GAN_RGB + 0) * 1) * 0.5
            self.loss_G.backward()
        else:
            self.loss_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
            self.loss_G_MSE = self.MSELoss(self.generate_letter, self._input_gt)

            self.loss_G_B_L1 = self.criterionL1(self.generate_letter_b, get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(0).transpose(0, 1))
            self.loss_G_B_MSE = self.MSELoss(self.generate_letter_b, get_binary_img(self._input_gt).transpose(0, 1)[0].unsqueeze(0).transpose(0, 1))
            self.loss_G = (0.5 * (self.loss_G_B_L1 + self.loss_G_B_MSE) + 0.5 * (self.loss_G_L1 + self.loss_G_MSE)) * 0.5
            self.loss_G.backward()

    def add_noise_disc(self, real):
        if self.opt.noisy_disc:
            rand_lbl = random.random()
            if rand_lbl < 0.6:
                label = (not real)
            else:
                label = (real)
        else:
            label = (real)
        return label

    def get_D_input(self, input_dis_img, is_Binary=False):
        style_count = self.input_style.size(1)
        input_style = self.input_style.transpose(0, 1)
        if is_Binary:
            input = torch.cat((input_dis_img, input_dis_img, input_dis_img), 1)
            input = torch.cat((input, self.input_content), 1)
            # input = torch.cat((self.get_binary_img(input_dis_img), self.input_content), 1)
        else:
            input = torch.cat((input_dis_img, self.input_content), 1)
        for i in range(style_count):
            input = torch.cat((input, input_style[i]), 1)
        return input

    def backward_D_Binary(self):
        # 把假的当做假的
        label_fake = self.add_noise_disc(False)
        fake_AB_2 = self.get_D_input(self.generate_letter_b, is_Binary=True)
        self.pred_fake_patch_2 = self.netD_Binary.forward(fake_AB_2.detach())
        self.loss_D_Binary_fake = self.criterionGAN(self.pred_fake_patch_2, label_fake) * 0.5
        transformed_AB_2 = self.preNet_Binary.forward(fake_AB_2.detach())
        self.pred_fake_2 = self.netD_Binary.forward(transformed_AB_2)
        self.loss_D_Binary_fake += self.criterionGAN(self.pred_fake_2, label_fake) * 0.5

        #　把真的当真的
        label_real = self.add_noise_disc(True)
        real_AB_2 = self.get_D_input(get_binary_img(self._input_gt), is_Binary=False)
        self.pred_real_patch_2 = self.netD_Binary.forward(real_AB_2)
        self.loss_D_Binary_real = self.criterionGAN(self.pred_real_patch_2, label_real) * 0.5
        ransformed_A_real_2 = self.preNet_Binary.forward(real_AB_2)
        self.pred_real_2 = self.netD_Binary.forward(ransformed_A_real_2)
        self.loss_D_Binary_real += self.criterionGAN(self.pred_real_2, label_real) * 0.5

        self.loss_D_Binary = (self.loss_D_Binary_fake + self.loss_D_Binary_real) * 0.5
        self.loss_D_Binary.backward()

    def backward_D_RGB(self):
        label_fake = self.add_noise_disc(False)
        fake_AB = self.get_D_input(self.generate_letter, is_Binary=False)
        self.pred_fake_patch = self.netD_RGB.forward(fake_AB.detach())
        self.loss_D_RGB_fake = self.criterionGAN(self.pred_fake_patch, label_fake) * 0.5
        transformed_AB = self.preNet_RGB.forward(fake_AB.detach())
        self.pred_fake = self.netD_RGB.forward(transformed_AB)
        self.loss_D_RGB_fake += self.criterionGAN(self.pred_fake, label_fake) * 0.5

        label_real = self.add_noise_disc(True)
        real_AB = self.get_D_input(self._input_gt, is_Binary=False)
        self.pred_real_patch = self.netD_RGB.forward(real_AB)
        self.loss_D_RGB_real = self.criterionGAN(self.pred_real_patch, label_real) * 0.5
        ransformed_A_real = self.preNet_RGB.forward(real_AB)
        self.pred_real = self.netD_RGB.forward(ransformed_A_real)
        self.loss_D_RGB_real += self.criterionGAN(self.pred_real, label_real) * 0.5

        self.loss_D_RGB = (self.loss_D_RGB_fake + self.loss_D_RGB_real) * 0.5
        self.loss_D_RGB.backward()

    def optimize_parameters(self):
        self.forward()
        if self.D_B:
            self.optimizer_D_Binary.zero_grad()
            self.optimizer_pre_Binary.zero_grad()
            self.backward_D_Binary()
            self.optimizer_D_Binary.step()
            self.optimizer_pre_Binary.step()
        if self.D_RGB:
            self.optimizer_D_RGB.zero_grad()
            self.optimizer_pre_RGB.zero_grad()
            self.backward_D_RGB()
            self.optimizer_D_RGB.step()
            self.optimizer_pre_RGB.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.use_gan:
            if self.isTrain:
                if self.D_RGB and self.D_B:
                    return OrderedDict([('loss_G_L1', self.loss_G_L1.data.item()),
                                        ('loss_G_B_L1', self.loss_G_B_L1.data.item()),
                                        ('loss_G', self.loss_G.data.item()),
                                        ('loss_D_RGB', self.loss_D_RGB.data.item()),
                                        ('loss_D_Binary', self.loss_D_Binary.data.item()),

                                        ])
                elif self.D_B:
                    return OrderedDict([('loss_G_L1', self.loss_G_L1.data.item()),
                                        ('loss_G_B_L1', self.loss_G_B_L1.data.item()),
                                        ('loss_G', self.loss_G.data.item()),
                                        ('loss_D_Binary', self.loss_D_Binary.data.item()),
                                        ])
                else:
                    return OrderedDict([('loss_G_L1', self.loss_G_L1.data.item()),
                                        ('loss_G_B_L1', self.loss_G_B_L1.data.item()),
                                        ('loss_G', self.loss_G.data.item()),
                                        ('loss_D_RGB', self.loss_D_RGB.data.item())
                                        ])
            else:  # test
                return OrderedDict([('loss_G_L1', self.loss_G_L1.data.item()),
                                    ('loss_G_B_L1', self.loss_G_B_L1.data.item()),
                                    ('loss_G', self.loss_G.data.item()),
                                    ])

        else:
            return OrderedDict([('loss_G_L1', self.loss_G_L1.data.item()),
                                ('loss_G_MSE', self.loss_G_MSE.data.item()),
                                ('loss_G_B_MSE', self.loss_G_B_MSE.data.item()),
                                ('loss_G_B_L1', self.loss_G_B_L1.data.item())
            ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, gpu_ids=self.gpu_ids)
        if self.D_B:
            self.save_network(self.netD_Binary, 'D_B', label, gpu_ids=self.gpu_ids)
            self.save_network(self.preNet_Binary, 'PRE_Binary', label, gpu_ids=self.gpu_ids)
        if self.D_RGB:
            self.save_network(self.netD_RGB, 'D_RGB', label, gpu_ids=self.gpu_ids)
            self.save_network(self.preNet_RGB, 'PRE_RGB', label, gpu_ids=self.gpu_ids)

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

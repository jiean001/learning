import random
from collections import OrderedDict

import torch
from torch.autograd import Variable

from . import network_controller
from .base_model import BaseModel
from utils.img_util import *
from utils.dir_util import *
import os


class Reweighted_GAN(BaseModel):
    def name(self):
        return 'Reweighted_GAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.save_imgs_dir = os.path.join(self.save_dir, 'imgs')
        mkdir(self.save_imgs_dir)

        self.isTrain = opt.isTrain

        self.input_style = self.Tensor(opt.batchSize, opt.style_num, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_content = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_gt = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = network_controller.define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf,
                                                norm=opt.norm, use_dropout=opt.use_dropout, gpu_ids=self.gpu_ids,
                                                which_model_netG=opt.which_model_netG, constant_cos=opt.constant_cos)
        self.preNet_RGB = network_controller.define_preNet(9 + 3 + 3, 9 + 3 + 3, gpu_ids=self.gpu_ids)
        self.preNet_Binary = network_controller.define_preNet(9 + 3 + 3, 9 + 3 + 3, gpu_ids=self.gpu_ids)
        self.netD_RGB = network_controller.define_D(input_nc=9 + 3 + 3, ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                                    is_RGB=True, n_layers_D=opt.n_layers_D,
                                                    use_sigmoid=opt.use_sigmoid, postConv = True, gpu_ids=self.gpu_ids)
        self.netD_Binary = network_controller.define_D(input_nc=9 + 3 + 3, ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                                    is_RGB=False, n_layers_D=opt.n_layers_D,
                                                    use_sigmoid=opt.use_sigmoid, postConv = True, gpu_ids=self.gpu_ids)


        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.preNet_RGB, 'PRE_RGB', opt.which_epoch)
            self.load_network(self.preNet_Binary, 'PRE_Binary', opt.which_epoch)
            self.load_network(self.netD_RGB, 'D_RGB', opt.which_epoch)
            self.load_network(self.netD_Binary, 'D_Binary', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.criterionL1 = torch.nn.L1Loss()
            self.MSELoss = torch.nn.MSELoss()
            self.criterionGAN = network_controller.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_pre_RGB = torch.optim.Adam(self.preNet_RGB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_pre_Binary = torch.optim.Adam(self.preNet_Binary.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_RGB = torch.optim.Adam(self.netD_RGB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_Binary = torch.optim.Adam(self.netD_Binary.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            network_controller.print_network(self.netG)
            network_controller.print_network(self.preNet_RGB)
            network_controller.print_network(self.preNet_Binary)
            network_controller.print_network(self.netD_RGB)
            network_controller.print_network(self.netD_Binary)
            print('-----------------------------------------------')

    def set_input(self, input):
        input_style = input['style_imgs']
        input_content = input['content_imgs']
        input_gt = input['gt_img']

        self.input_style.resize_(input_style.size()).copy_(input_style)
        self.input_content.resize_(input_content.size()).copy_(input_content)
        self.input_gt.resize_(input_gt.size()).copy_(input_gt)

    def forward(self):
        self._input_style = Variable(self.input_style)
        self._input_content = Variable(self.input_content)
        self._input_gt = Variable(self.input_gt, volatile=True)

        self.generate_letter, self.generate_letter_b = self.netG.forward(self._input_style, self._input_content)

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
    
    # no backprop gradients
    def test(self):
        self._input_style = Variable(self.input_style, volatile=True)
        self._input_content = Variable(self.input_content, volatile=True)
        self._input_gt = Variable(self.input_gt, volatile=True)

        self.generate_letter, self.generate_letter_b = self.netG.forward(self._input_style, self._input_content)

    def backward_D_Binary(self):
        label_fake = self.add_noise_disc(False)
        fake_AB_2 = self.get_D_input(self.generate_letter_b, is_Binary=True)
        # self.pred_fake_patch_2 = self.netD_Binary.forward(self._input_content)
        self.pred_fake_patch_2 = self.netD_Binary.forward(fake_AB_2.detach())
        self.loss_D_Binary_fake = self.criterionGAN(self.pred_fake_patch_2, label_fake)
        transformed_AB_2 = self.preNet_Binary.forward(fake_AB_2.detach())
        self.pred_fake_2 = self.netD_Binary.forward(transformed_AB_2)
        self.loss_D_Binary_fake += self.criterionGAN(self.pred_fake_2, label_fake)

        label_real = self.add_noise_disc(True)
        real_AB_2 = self.get_D_input(get_binary_img(self._input_gt), is_Binary=False)
        self.pred_real_patch_2 = self.netD_Binary.forward(real_AB_2)
        self.loss_D_Binary_real = self.criterionGAN(self.pred_real_patch_2, label_real)
        ransformed_A_real_2 = self.preNet_Binary.forward(real_AB_2)
        self.pred_real_2 = self.netD_Binary.forward(ransformed_A_real_2)
        self.loss_D_Binary_real += self.criterionGAN(self.pred_real_2, label_real)

        self.loss_D_Binary = (self.loss_D_Binary_fake + self.loss_D_Binary_real) * 0.5
        self.loss_D_Binary.backward()

    def backward_D_RGB(self):
        label_fake = self.add_noise_disc(False)
        fake_AB = self.get_D_input(self.generate_letter, is_Binary=False)
        self.pred_fake_patch = self.netD_RGB.forward(fake_AB.detach())
        self.loss_D_RGB_fake = self.criterionGAN(self.pred_fake_patch, label_fake)
        transformed_AB = self.preNet_RGB.forward(fake_AB.detach())
        self.pred_fake = self.netD_RGB.forward(transformed_AB)
        self.loss_D_RGB_fake += self.criterionGAN(self.pred_fake, label_fake)

        label_real = self.add_noise_disc(True)
        real_AB = self.get_D_input(self._input_gt, is_Binary=False)
        self.pred_real_patch = self.netD_RGB.forward(real_AB)
        self.loss_D_RGB_real = self.criterionGAN(self.pred_real_patch, label_real)
        ransformed_A_real = self.preNet_RGB.forward(real_AB)
        self.pred_real = self.netD_RGB.forward(ransformed_A_real)
        self.loss_D_RGB_real += self.criterionGAN(self.pred_real, label_real)

        self.loss_D_RGB = (self.loss_D_RGB_fake + self.loss_D_RGB_real) * 0.5
        self.loss_D_RGB.backward()

    def backward_G(self):
        fake_AB = self.get_D_input(self.generate_letter, is_Binary=False)
        pred_fake_patch = self.netD_RGB.forward(fake_AB)
        self.loss_G_GAN_RGB = self.criterionGAN(pred_fake_patch, True)
        transformed_A = self.preNet_RGB.forward(fake_AB)
        pred_fake = self.netD_RGB.forward(transformed_A)
        self.loss_G_GAN_RGB += self.criterionGAN(pred_fake, True)

        fake_AB_2 = self.get_D_input(self.generate_letter_b, is_Binary=True)
        pred_fake_patch_2 = self.netD_Binary.forward(fake_AB_2)
        self.loss_G_GAN_BINARY = self.criterionGAN(pred_fake_patch_2, True)
        transformed_A_2 = self.preNet_RGB.forward(fake_AB_2)
        pred_fake_2 = self.netD_Binary.forward(transformed_A_2)
        self.loss_G_GAN_BINARY += self.criterionGAN(pred_fake_2, True)

        self.loss_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
        self.loss_G_MSE = self.MSELoss(self.generate_letter, self._input_gt)

        self.loss_G = self.loss_G_GAN_RGB + self.loss_G_GAN_BINARY + self.loss_G_L1 + self.loss_G_MSE
        # self.loss_G = self.loss_G_GAN_BINARY
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D_Binary.zero_grad()
        self.optimizer_pre_Binary.zero_grad()
        self.backward_D_Binary()
        self.optimizer_D_Binary.step()
        self.optimizer_pre_Binary.step()

        self.optimizer_D_RGB.zero_grad()
        self.optimizer_pre_RGB.zero_grad()
        self.backward_D_RGB()
        self.optimizer_D_RGB.step()
        self.optimizer_pre_RGB.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('loss_G_GAN_RGB', self.loss_G_GAN_RGB.data.item()),
                            ('loss_G_GAN_BINARY', self.loss_G_GAN_BINARY.data.item()),
                            ('loss_G_L1', self.loss_G_L1.data.item()),
                            ('loss_G_MSE', self.loss_G_MSE.data.item()),
                            ('loss_D_RGB', self.loss_D_RGB.data.item()),
                            ('loss_D_Binary', self.loss_D_Binary.data.item())
        ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, gpu_ids=self.gpu_ids)
        self.save_network(self.preNet_RGB, 'PRE_RGB', label, gpu_ids=self.gpu_ids)
        self.save_network(self.preNet_Binary, 'PRE_Binary', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netD_RGB, 'D_RGB', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netD_Binary, 'D_Binary', label, gpu_ids=self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_Binary.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_RGB.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_pre_Binary.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_pre_RGB.param_groups:
            param_group['lr'] = lr
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
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

        self.netG = network_controller.define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf,
                                                norm=opt.norm, use_dropout=opt.use_dropout, gpu_ids=self.gpu_ids,
                                                which_model_netG=opt.which_model_netG, constant_cos=opt.constant_cos)


        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.criterionL1 = torch.nn.L1Loss()
            self.MSELoss = torch.nn.MSELoss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            network_controller.print_network(self.netG)
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


    # no backprop gradients
    def test(self):
        self._input_style = Variable(self.input_style, volatile=True)
        self._input_content = Variable(self.input_content, volatile=True)
        self._input_gt = Variable(self.input_gt, volatile=True)

        self.generate_letter, self.generate_letter_b = self.netG.forward(self._input_style, self._input_content)

    def backward_G(self):
        self.loss_G_L1 = self.criterionL1(self.generate_letter, self._input_gt)
        self.loss_G_MSE = self.MSELoss(self.generate_letter, self._input_gt)
        self.loss_G_B_L1 = self.criterionL1(self.generate_letter, get_binary_img(self._input_gt))
        self.loss_G_B_MSE = self.MSELoss(self.generate_letter, get_binary_img(self._input_gt))

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
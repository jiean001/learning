#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-05
#
# Author: jiean001
#########################################################
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import network_controller
from .base_model import BaseModel
from collections import OrderedDict
import torch.nn.functional as F

class Classifier_Model(BaseModel):
    def name(self):
        return 'Classifier_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # 输入
        self.input = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        # 输出
        self.label = self.Tensor(opt.batchSize, 1)
        # 模型
        self.Classifier = network_controller.define_Classifier(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, norm=opt.norm, use_dropout=opt.use_dropout, gpu_ids=self.gpu_ids, which_model_net_Classifier=opt.which_model_net_Classifier)
        # 加载模型
        if not self.isTrain or opt.continue_train:
            self.load_network(self.Classifier, 'Classifier', opt.which_epoch)
        if self.isTrain:
            # 优化设置
            self.old_lr = opt.lr
            self.criterion = F.nll_loss # nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # 打印模型
        print('---------- Networks initialized -------------')
        network_controller.print_network(self.Classifier)
        print('-----------------------------------------------')

    def set_input(self, input):
        data = input[0]  # input = input['data']
        label = input[1]  # label = input['label']

        self.input.resize_(data.size()).copy_(data)
        self.label.resize_(label.size()).copy_(label)

    def forward(self):
        self._input = Variable(self.input, requires_grad=True)
        self._label = Variable(self.label, requires_grad=False)

        self.optimizer.zero_grad()
        self.output = self.Classifier.forward(self._input)

        self.loss = self.criterion(self.output, self._label.long())

        self.loss.backward()
        self.optimizer.step()

    def test(self, is_print):
        self._input = Variable(self.input, volatile=True)
        self._label = Variable(self.label, volatile=True)

        self.output = self.Classifier.forward(self._input)
        self.test_loss = F.nll_loss(self.output, self._label, size_average=False).data[0]
        pred = self.output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        self.correct = pred.eq(self._label.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

        if is_print:
            num = len(self._label)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                self.test_loss / num , self.correct, len(num),
                100. * self.correct / len(num)))

    def optimize_parameters(self):
        self.forward()

    def get_current_errors(self):
        return OrderedDict([('loss', self.loss.item())])

    def save(self, label):
        self.save_network(self.Classifier, 'Classifier', label, gpu_ids=self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.save_network.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def get_embedding_para(self):
        return self.output, self._label, self._input

    def get_model(self):
        return self.Classifier

    def get_input(self):
        return self._input
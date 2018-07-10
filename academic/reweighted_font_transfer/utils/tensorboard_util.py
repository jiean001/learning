#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-07
#
# Author: jiean001
#
# tensorboard可视化
#########################################################

import torch
from tensorboardX import SummaryWriter

class TB_Visualizer:
    def __init__(self, log_dir=r'./runs',comment='tensorboardX_name', use_tensorboardX=False, use_gou=True):
        print('log_dir', log_dir)
        self.use_tensorboardX = use_tensorboardX
        if self.use_tensorboardX:
            self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
            self.use_gpu = use_gou

    def close(self):
        if self.use_tensorboardX:
            self.writer.close()

    def add_loss(self, errors, scalar_x):
        if self.use_tensorboardX:
            for k, v in errors.items():
                self.add_scalar(scalar_name=k, scalar_value=v, scalar_x=scalar_x)

    def add_scalar(self, scalar_name, scalar_value, scalar_x):
        if self.use_tensorboardX:
            self.writer.add_scalar(scalar_name, scalar_value, scalar_x)

    def add_embedding(self, out, label, data, scalar_x):
        # print(label.data[:10])
        if self.use_gpu:
            out = torch.cat((out.data, torch.ones(len(out), 1).cuda()), 1)
        else:
            out = torch.cat((out.data, torch.ones(len(out), 1)), 1)
        self.writer.add_embedding(out, metadata=label.data, label_img=data.data, global_step=scalar_x)

    def add_graph(self, model, dummy_input):
        if self.use_tensorboardX:
            print('add graph')
            self.writer.add_graph(model, (dummy_input,))



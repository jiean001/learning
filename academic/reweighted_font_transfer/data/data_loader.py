#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-06-09
#
# Author: jiean001
#
# 预训练分类网络的数据组织形式
#########################################################

import torch
import torchvision.datasets as dt
import torchvision.transforms as transforms
import os
from .base_data_loader import BaseDataLoader


def CreateDataLoader(opt):
    data_loader = None
    if opt.classifier:
        data_loader = Classifier_DataLoader()
    data_loader.initialize(opt)
    return data_loader


class Classifier_DataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize

        transform = {
            'train': transforms.Compose([
                transforms.Resize(opt.loadSize, opt.loadSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                         (0.5, 0.5, 0.5))
            ]),
            'val': transforms.Compose([
                transforms.Resize(opt.loadSize, opt.loadSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        }

        dataset = {x: dt.ImageFolder(os.path.join(opt.dataroot, x), transform[x]) for x in ['train', 'val']}

        data_loader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=self.opt.batchSize, shuffle=not self.opt.serial_batches, num_workers=int(self.opt.nThreads))
                              for x in ['train', 'val']}

        self.dataset = dataset
        self._data = data_loader

    def name(self):
        return 'Classifier_DataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return 9
        # return min(len(self.dataset), self.opt.max_dataset_size)
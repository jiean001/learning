#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-09
#
# Author: jiean001
#
# 自定义DataLoader
#########################################################

import torch
import torchvision.datasets as dt
import torchvision.transforms as transforms
import os
from .base_data_loader import BaseDataLoader
from .image_folder import ImageFolder


def CreateDataLoader(opt):
    data_loader = None
    if opt.classifier:
        data_loader = Classifier_DataLoader()
    elif opt.reweighted:
        data_loader = ReWeighted_DataLoader()
    data_loader.initialize(opt)
    return data_loader


class ReWeighted_DataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize

        transform = transforms.Compose([
            transforms.Resize(opt.loadSize, opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        if opt.isTrain:
            root = os.path.join(opt.dataroot, 'train')
            config_dir = os.path.join(opt.config_dir, 'train')
        else:
            if opt.test_type == 'test_seen':
                root = os.path.join(opt.dataroot, 'train')
            elif opt.test_type.startswith('test_unseen'):
                root = os.path.join(opt.dataroot, 'test')
            config_dir = os.path.join(opt.config_dir, opt.test_type)

        _, dataset_name = os.path.split(opt.dataroot)

        dataset = ImageFolder(root=root, config_dir=config_dir, dataset_name=dataset_name,
                              transform=transform, fineSize=opt.fineSize, no_permutation=not opt.isTrain)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=opt.isTrain,
            num_workers=int(self.opt.nThreads))

        self.dataset = dataset
        self._data = ReWeighted_Data(data_loader, opt.max_dataset_size)

    def name(self):
        return 'ReWeighted_DataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


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
            'test': transforms.Compose([
                transforms.Resize(opt.loadSize, opt.loadSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        }

        dataset = {x: dt.ImageFolder(os.path.join(opt.dataroot, x), transform[x]) for x in ['train', 'val', 'test']}
        data_loader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=self.opt.batchSize, shuffle=not self.opt.serial_batches, num_workers=int(self.opt.nThreads))
                              for x in ['train', 'val', 'test']}

        self.dataset = dataset
        self._data = data_loader

    def name(self):
        return 'Classifier_DataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset['train']), self.opt.max_dataset_size)


class ReWeighted_Data(object):
    def __init__(self, data_loader, max_dataset_size):
        self.data_loader = data_loader
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def next(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        try:
            style_imgs, content_imgs, gt_img, standard_imgs = next(self.data_loader_iter)
            return {'style_imgs': style_imgs, 'content_imgs': content_imgs, 'gt_img': gt_img, 'standard_imgs': standard_imgs}
        except:
            style_imgs, content_imgs, gt_img = next(self.data_loader_iter)
            return {'style_imgs': style_imgs, 'content_imgs': content_imgs, 'gt_img': gt_img}

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        try:
            style_imgs, content_imgs, gt_img, standard_imgs = next(self.data_loader_iter)
            return {'style_imgs': style_imgs, 'content_imgs': content_imgs, 'gt_img': gt_img, 'standard_imgs': standard_imgs}
        except:
            style_imgs, content_imgs, gt_img = next(self.data_loader_iter)
            return {'style_imgs': style_imgs, 'content_imgs': content_imgs, 'gt_img': gt_img}
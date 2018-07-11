#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-10
#
# Author: jiean001
#
# 自定义ImageFolder
#########################################################

from utils.img_util import *
import torch.utils.data as data
import os
import os.path
import numpy as np
import torch
import json
import glob


# 将dataset保存为list
def make_dataset(config_file_dir, dataset_name):
    config_file_list = glob.glob(r'%s/%s*.json' %(config_file_dir, dataset_name))
    is_first = True
    for config_file in config_file_list:
        f = open(config_file, 'r')
        _images = json.load(f)
        if is_first:
            is_first = False
            images = _images
        else:
            images.append(_images)
        f.close()
    np.random.shuffle(images)
    return images


# [batch, C, H, W]
def get_gt_tensor(root, path_list, loader=default_img_loader, transform=None, fineSize=64):
    is_first = True
    for path in path_list:
        _path = os.path.join(root, path)
        img = loader(_path, fineSize, fineSize)
        if transform is not None:
            img = transform(img)
        if is_first:
            is_first = False
            ims = img
        else:
            ims = torch.cat((ims, img))
    return ims


# [batch, C(1,2,3), H, W]
def get_content_tensor(root, path_list, loader=default_img_loader, transform=None, fineSize=64):
    is_first = True
    for path in path_list:
        _path = os.path.join(root, path)
        img = loader(_path, fineSize, fineSize)
        if transform is not None:
            img = transform(img)
        if is_first:
            is_first = False
            ims = img[0].unsqueeze(0)
        else:
            ims = torch.cat((ims, img[0].unsqueeze(0)))
    return get_binary_img(ims)


# [batch, num, C, H, W]
def get_style_tensor(root, path_list, loader=default_img_loader, transform=None, fineSize=64):
    is_first = True
    for path in path_list:
        _path = os.path.join(root, path)
        img = loader(_path, fineSize, fineSize)
        if transform is not None:
            img = transform(img)
        if is_first:
            is_first = False
            ims = img.unsqueeze(0)
        else:
            ims = torch.cat((ims, img.unsqueeze(0)))
    return ims


class ImageFolder(data.Dataset):

    def __init__(self, root, config_dir, dataset_name, transform=None,
                 loader=default_img_loader, fineSize=0,
                 no_permutation=False):
        print(config_dir)
        imgs = make_dataset(config_dir, dataset_name)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + ', ' + config_dir + "\n"
                                "Supported image extensions are: " + ",".join('.png')))

        self.root = root
        self.imgs = imgs
        if no_permutation:
            self.imgs = sorted(self.imgs)

        self.transform = transform
        self.loader = loader
        self.fineSize = fineSize

    def __getitem__(self, index):
        style_content_gt = self.imgs[index]
        style_list = style_content_gt[0]
        content_list = style_content_gt[1]
        gt_img_ = style_content_gt[2]

        style_imgs = get_style_tensor(self.root, style_list, self.loader, self.transform, fineSize=self.fineSize)
        content_imgs = get_content_tensor(self.root, content_list, self.loader, self.transform, fineSize=self.fineSize)
        gt_img = get_gt_tensor(self.root, gt_img_, self.loader, self.transform, fineSize=self.fineSize)
        return style_imgs, content_imgs, gt_img

    def __len__(self):
        return len(self.imgs)

#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-06
#
# Author: jiean001
#
# 图片基本操作
#########################################################
from PIL import Image


# 图片的默认打开方式
def default_img_loader(path, width=64, height=64):
    if width and height:
        return Image.open(path).convert('RGB').resize((width, height))
    return Image.open(path).convert('RGB')


# 判断是否为图片
def is_image(input_path):
    return input_path.endswith('.png') or input_path.endswith('.jpg')


# 非精确获得二值化的图,输出和输入的size一样
def get_binary_img(imgs, is_print=False):
    delta = 0.000001
    tmp = imgs
    tmp = (tmp + delta).int()
    tmp = (tmp.float() - 0.5) / 0.5
    if is_print:
        print(tmp[0][0][0])
    return tmp.float()

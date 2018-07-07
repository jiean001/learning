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


def is_image(input_path):
    return input_path.endswith('.png') or input_path.endswith('.jpg')

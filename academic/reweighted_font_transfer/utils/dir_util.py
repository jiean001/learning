#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-06
#
# Author: jiean001
#
# 文件夹基本操作
#########################################################

import os


# 遍历rootdir文件夹下所有文件和文件夹
# 按照deal_dir和deal_file的方式处理
def iterate_dir(rootdir, deal_dir=None, deal_file=None):
    for dir_or_file in os.listdir(rootdir):
        path = os.path.join(rootdir, dir_or_file)
        if os.path.isfile(path):
            if deal_file == 'pass':
                pass
            elif deal_file:
                deal_file(path)
            else:
                print(path)
        if os.path.isdir(path):
            if deal_dir == 'pass':
                pass
            elif deal_dir:
                deal_dir(path)
            else:
                print(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

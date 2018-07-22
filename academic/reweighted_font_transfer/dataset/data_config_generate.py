#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-10
#
# Author: jiean001
#
# 生成数据集的配置文件
# style_content_gt:
#   -- list
#      -- style_list
#      -- content_list
#      -- gt_list
#  -- list
#     ...
#########################################################

from utils.dir_util import *
import json
import numpy as np

# 　style和content的关系表
# 这个表应该是随机生成的，当前先固定下来
style_content_related_map = {'style_name': '[content_name1, content_name2, ..., content_nameN]'}
content_file_lists = ['000004', '000005', '000009']


# 随机生成一个字符
def generate_one_char(file_name=None):
    if file_name:
        return '%s/%s.png' % (file_name, chr(np.random.randint(0, 26) + ord('A')))
    return '%s.png' % (chr(np.random.randint(0, 26) + ord('A')))


# 随机生成一组style_num个style字符
def generate_style_pair(style_num=3, file_name=None):
    style_pair = [generate_one_char(file_name)]
    for i in range(1, style_num):
        new_char = generate_one_char(file_name)
        while new_char in style_pair:
            new_char = generate_one_char(file_name)
        style_pair.append(new_char)
    return style_pair


# 随机生成ground truth
def generate_gt(file_name=None):
    return [generate_one_char(file_name)]


# 随机生成content数据
def generate_content_pair(gt_name):
    gt_name = gt_name[0]
    style_file_name, gt = os.path.split(gt_name)
    content_pair = []
    # 在下一个version进行更改
    # for content_file_name in style_content_related_map[style_file_name]:
    for content_file_name in content_file_lists:
        atom = gt_name.replace(style_file_name, content_file_name)
        content_pair.append(atom)
    return content_pair


# 生成一行数据
def generate_one_row(style_num=3, file_name=None):
    one_row = []
    style_pair = generate_style_pair(style_num=style_num, file_name=file_name)
    gt = generate_gt(file_name=file_name)
    content_pair = generate_content_pair(gt)
    one_row.append(style_pair)
    one_row.append(content_pair)
    one_row.append(gt)
    return one_row


class Data_Config_Generate:
    def __init__(self, each_style_num=4, each_config_num=1024,
                 dataset_name = r'Capitals_colorGrad64',
                 dataset_dir=r'/home/xiongbo/datasets/SEPARATE/Capitals_colorGrad64/train/',
                 config_dir=r'../config/train/',
                 iterate_dir=iterate_dir, generate_one_row=generate_one_row):
        # 每个配置文件有 each_style_num * each_config_num条数据
        # 每个style有几个字符 Capitals_colorGrad64
        self.each_style_num = each_style_num
        # 每个config文件有多少组风格
        self.each_config_num = each_config_num
        # 数据集合的路径
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        # 配置文件的存放地
        self.config_dir = config_dir
        # 当前配置文件存放了多少条数据
        self.crt_config_num = 0
        # 当前配置文件的索引
        self.crt_config_file_index = 0

        self.generate_one_row = generate_one_row

        self.create_new_cofig_file()
        # 遍历文件夹
        self.iterate_dir = iterate_dir
        self.iterate_dir(self.dataset_dir, deal_dir=self.deal_dir, deal_file=self.deal_file)
        self.save_to_json()

    def create_new_cofig_file(self):
        self.config_file = os.path.join(self.config_dir, '%s_%04d.json' %(self.dataset_name, self.crt_config_file_index))
        self.crt_config_file_index += 1
        self.crt_config_num = 0
        self.style_content_gt_list = []

    # 　生成每个font的训练数据
    def deal_one_font(self, style_num=3, file_name=None):
        _style_content_gt_list = self.generate_one_row(style_num=style_num, file_name=file_name)
        for i in range(1, self.each_style_num):
            s_c_g = self.generate_one_row(style_num=style_num, file_name=file_name)
            while s_c_g in _style_content_gt_list:
                s_c_g = self.generate_one_row(style_num=style_num, file_name=file_name)
                _style_content_gt_list.append(s_c_g)
        return _style_content_gt_list

    #  保存为json文件
    def save_to_json(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.style_content_gt_list, f)
            print('save %s okay' % (self.config_file))
        self.create_new_cofig_file()

    def deal_dir(self, path):
        _, style_name = os.path.split(path)
        self.crt_config_num += 1
        self.style_content_gt_list.append(self.deal_one_font(file_name=style_name))
        if self.crt_config_num == self.each_config_num:
            self.save_to_json()

    def deal_file(self, image_path):
        return 'pass'


if __name__ == '__main__':
    # pc
    # tmp = Data_Config_Generate()
    # 93
    dataset_dir = r'/home/share/dataset/MCGAN/SEPARATE/Capitals_colorGrad64/train/'
    tmp_93 = Data_Config_Generate(dataset_dir=dataset_dir)


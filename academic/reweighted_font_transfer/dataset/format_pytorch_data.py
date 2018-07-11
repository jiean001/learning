#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-09
#
# Author: jiean001
#
# 格式化pytorch形式的数据输入,将同一个class的图片放到同一个label下
#########################################################

from utils.dir_util import *

# 将mcgan的数据进行分割，并且保存两份
class Format_Pytorch_DataSet_T:
    def __init__(self, source_dir=r'/home/share/dataset/MCGAN/TOGETHER/public_web_fonts_t', iterate_dir=iterate_dir, cmd=r'./mv_label_dir.sh'):
        self.source_dir = source_dir
        self.iterate_dir = iterate_dir
        self.cmd = cmd

        self.deal_dir(self.source_dir)
        # self.iterate_dir(self.source_dir, deal_dir=self.deal_dir, deal_file='pass')

    def deal_dir(self, path):
        _, current_dir = os.path.split(path)
        print(current_dir)
        if current_dir == 'train' or current_dir == 'test' or current_dir == 'val':
            for i in range(26):
                cmd = '%s %s %s' %(self.cmd, path, chr(ord('A') + i))
                print(cmd)
                os.system(cmd)
            return
        elif current_dir == 'BASE':
            return
        else:
            self.iterate_dir(path, deal_dir=self.deal_dir, deal_file='pass')

# 将mcgan的数据进行分割，并且保存两份
class Format_Pytorch_DataSet_S:
    def __init__(self, source_dir=r'/home/share/dataset/MCGAN/TOGETHER/public_web_fonts_t', iterate_dir=iterate_dir):
        self.source_dir = source_dir
        self.iterate_dir = iterate_dir

        # self.deal_dir(self.source_dir)
        self.iterate_dir(self.source_dir, deal_dir=self.deal_dir, deal_file='pass')

    def deal_dir(self, path):
        _, current_dir = os.path.split(path)
        print(current_dir)
        if current_dir == 'train' or current_dir == 'test' or current_dir == 'val':
            crt_index = 0
            for style_name in os.listdir(path):
                old_path = os.path.join(path, style_name)
                new_path = os.path.join(path, '%06d' %(crt_index))
                crt_index += 1
                cmd = 'mv "%s" %s' %(old_path, new_path)
                os.system(cmd)
            return
        elif current_dir == 'BASE':
            return
        else:
            self.iterate_dir(path, deal_dir=self.deal_dir, deal_file='pass')

if __name__ == '__main__':
    # TOGETHER...
    # pc
    # source_dir = r'/home/xiongbo/datasets/TOGETHER'
    # 218
    # source_dir = r'/home/share/dataset/MCGAN/TOGETHER'
    # mcgan = Format_Pytorch_DataSet_T(source_dir=source_dir)

    # Separate...
    # pc
    source_dir = r'/home/xiongbo/datasets/SEPARATE/'
    Format_Pytorch_DataSet_S(source_dir=source_dir)
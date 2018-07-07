#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-06-09
#
# Author: jiean001
#
# 格式化pytorch形式的数据输入,将同一个class的图片放到同一个label下
#########################################################

from utils.dir_util import *

# 将mcgan的数据进行分割，并且保存两份
class Format_Pytorch_DataSet:
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

if __name__ == '__main__':
    mcgan = Format_Pytorch_DataSet(source_dir=r'/home/share/dataset/MCGAN/TOGETHER/Capitals_colorGrad64/train')
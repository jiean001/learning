#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-05
#
# Author: jiean001
#
# 将stack的数据【mcgan】分成一个一个的字符,并且分开两份保存
# 一份作classifier
# 另一份作为reweigted net的数据输入
#########################################################


from utils.img_util import *
from utils.dir_util import *


# 切割图片
def cut_image(image_path, _width=64*26, _height=64, out_together=None, out_separate=None):
    image = default_img_loader(image_path, None, None)
    width, height = image.size
    if not (width == _width and height == _height):
        return
    assert width == _width and height == _height, 'error %s' %(image_path)
    for i in range(width // height):
        box=(i*height, 0, (i+1)*height, height)
        new_img = image.crop(box)
        # 将全部截取的图片放在一起
        if out_together:
            base_name = os.path.basename(image_path).replace('.png', '').replace('.jpg', '')
            img_together_name = '%s_%s.png' %(base_name, chr(ord('A') + i))
            out_together_path = os.path.join(out_together, img_together_name)
            new_img.save(out_together_path)
        # 从一幅图里面截取到的图片放到一起
        if out_separate:
            img_separate_name = '%s.png' %(chr(ord('A') + i))
            out_separate_path = os.path.join(out_separate, img_separate_name)
            new_img.save(out_separate_path)


# 将mcgan的数据进行分割，并且保存两份
class MCGAN_DATA:
    def __init__(self, source_dir=r'/home/share/dataset/MCGAN/ORIGIN', target_together_dir=r'/home/share/dataset/MCGAN/TOGETHER', target_separate_dir=r'/home/share/dataset/MCGAN/SEPARATE', iterate_dir=iterate_dir):
        self.source_dir = source_dir
        self.target_separate_dir = target_separate_dir
        self.target_together_dir = target_together_dir
        self.iterate_dir = iterate_dir
        self.iterate_dir(self.source_dir, deal_dir=self.deal_dir, deal_file=self.deal_file)

    def deal_dir(self, path):
        self.iterate_dir(path, deal_dir=self.deal_dir, deal_file=self.deal_file)

    def deal_file(self, image_path):
        dirname, filename = os.path.split(image_path)
        if not is_image(filename):
            return
        together_dir = dirname.replace(self.source_dir, self.target_together_dir)
        separete_dir = '%s/%s' %(dirname.replace(self.source_dir, self.target_separate_dir), filename.replace('.png', '').replace('.jpg', ''))
        try:
            cut_image(image_path, out_together=together_dir, out_separate=separete_dir)
        except IOError:  # python 2.7
        # except FileNotFoundError:  # python 3.6
            if not os.path.exists(together_dir):
                os.makedirs(together_dir)
                print('create dir: %s' %(together_dir))
            if not os.path.exists(separete_dir):
                os.makedirs(separete_dir)
                print('create dir: %s' %(separete_dir))
            cut_image(image_path, out_together=together_dir, out_separate=separete_dir)


if __name__ == '__main__':
    # pc
    source_dir = r'/home/xiongbo/datasets/'
    target_together_dir = r'/home/xiongbo/datasets/TOGETHER'
    target_separate_dir = r'/home/xiongbo/datasets/separate'
    mcgan = MCGAN_DATA(source_dir=source_dir, target_together_dir=target_together_dir, target_separate_dir=target_separate_dir)

    # mcgan = MCGAN_DATA()  # (source_dir=r'.', target_together_dir=r'./together', target_separate_dir=r'./separate')

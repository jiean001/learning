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
import torchvision.utils as vutils
import torch
import torchvision.transforms as transforms

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

#  一个batch的训练图
def get_one_pair_imgs(data_dict, generate_imgs=None, generate_imgs_b=None):
    style_imgs = data_dict['style_imgs'].cuda()  # [b,n,c,h,w]
    content_imgs = data_dict['content_imgs'].cuda()  # [b,c,h,w]
    gt_img = data_dict['gt_img'].cuda()  # [b,c,h,w]
    gt_img_b = get_binary_img(gt_img)

    generate_imgs_b = torch.cat((generate_imgs_b, generate_imgs_b, generate_imgs_b), 1)

    content_imgs = content_imgs.view(content_imgs.size(0), content_imgs.size(1), 1, content_imgs.size(2),
                                     content_imgs.size(3))

    is_First = True
    for style_index in range(content_imgs.size(0)):  # batch
        if generate_imgs is not None:
            if is_First:
                is_First = False
                imgs = generate_imgs[style_index].unsqueeze(0)
            else:
                imgs = torch.cat((imgs, generate_imgs[style_index].unsqueeze(0)))

        if is_First:
            is_First = False
            imgs = gt_img[style_index].unsqueeze(0)
        else:
            imgs = torch.cat((imgs, gt_img[style_index].unsqueeze(0)))

        if generate_imgs_b is not None:
            imgs = torch.cat((imgs, generate_imgs_b[style_index].unsqueeze(0)))
        crt_gt_b = gt_img_b[style_index].unsqueeze(0)
        imgs = torch.cat((imgs, crt_gt_b))

        for number_index in range(style_imgs.size(1)):
            imgs = torch.cat((imgs, style_imgs[style_index][number_index].unsqueeze(0)))

        for number_index in range(content_imgs.size(1)):
            crt_content_c1 = content_imgs[style_index][number_index]
            crt_content = torch.cat((crt_content_c1, crt_content_c1, crt_content_c1))
            imgs = torch.cat((imgs, crt_content.unsqueeze(0)))
    return imgs, content_imgs.size(0)


#  打印图片
def print_imgs(data_dict, out_name, generate_imgs=None, generate_imgs_b=None, batch_size=None):
    # size: (batch, channel*number, height, width)
    imgs, _batch_size = get_one_pair_imgs(data_dict, generate_imgs, generate_imgs_b)
    if batch_size:
        pass
    else:
        batch_size = _batch_size
    img_num = imgs.size(0)
    row_num = img_num / batch_size
    vutils.save_image(imgs, out_name, nrow=row_num)


def print_img(img, out_name):
    vutils.save_image(img, out_name)


rew_transform = transforms.Compose([
            transforms.Resize(64, 64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
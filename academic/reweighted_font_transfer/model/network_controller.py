#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-05
#
# Author: jiean001
#########################################################

from .networks import *


def define_Classifier(input_nc=3, output_nc=26, ngf=64, norm='batch', use_dropout=False, gpu_ids=[], which_model_net_Classifier='Classifier_letter'):
    net_Classifier = None
    if gpu_ids:
        use_gpu = len(gpu_ids) > 0
    else:
        use_gpu = False

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_net_Classifier == 'Classifier_letter':
        net_Classifier = Classifier_letter(input_nc=input_nc, output_nc=output_nc, ngf=ngf, norm=norm, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print('Classifier model name [%s] is not recognized' %which_model_net_Classifier)

    if use_gpu:
        net_Classifier.cuda(device=gpu_ids[0])
    net_Classifier.apply(weights_init)
    return net_Classifier


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)

    print('Total number of parameters: %d' % num_params)
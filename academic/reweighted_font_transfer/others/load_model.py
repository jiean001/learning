#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-23
#
# Author: jiean001
#########################################################

import json
from utils.img_util import *

S_C_config = r'../config/style_classifier.txt'
C_C_config = r'../config/content_classifier.txt'


def load_base_network(network, weights, print_weights=False, ignore_BN=False, is_visiable_filter=False):
    if ignore_BN:
        for key in weights.keys():
            if key.endswith('running_mean'):
                weights[key].zero_()
            elif key.endswith('running_var'):
                weights[key].fill_(1.0)
    if print_weights:
        for key in weights.keys():
            print(key, 'pretrained, mean,std:', torch.mean(weights[key]), torch.std(weights[key]))

    if is_visiable_filter:
        haha = 0
        for key in weights.keys():
            if len(weights[key].size()) == 4:
                filter = weights[key][0]
                filter = (filter+1).int()
                # print(filter)
                # print((filter+1).int())
                for i in range(1, weights[key].size(0)):
                    tmp = weights[key][i]
                    tmp = (tmp+1).int()
                    # 26 * 26 * 3
                    # 64 7
                    # 64 * 3 * 7 * 7
                    filter = torch.cat((filter, tmp), 0)
                    '''
                    if haha == 0:
                        if i > 4:
                            break
                    else:
                        break
                    '''
                haha = 1
                print(key, type(weights[key]), weights[key].size(), filter.size())
                print_img(filter.unsqueeze(1), '%s.png' % (key))
                # break

    if network:
        network.load_state_dict(weights)


def load_network(network, save_path=r'./saved_model/60_net_Classifier.pth', print_weights=False, ignore_BN=False, is_visiable_filter=False):
    weights = torch.load(save_path)
    load_base_network(network, weights, print_weights, ignore_BN, is_visiable_filter)


def load_section_network(network, config_file=S_C_config, save_path=r'./saved_model/60_net_Classifier.pth'):
    classifier_weights = torch.load(save_path)
    weights = {}
    f = open(config_file, 'r')
    S_C_dict = json.load(f)
    for s_key in S_C_dict.keys():
        weights[s_key] = classifier_weights[S_C_dict[s_key]]
    load_base_network(network, weights, print_weights=True)

load_network(None, is_visiable_filter=True)
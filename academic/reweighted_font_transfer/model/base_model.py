#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-05
#
# Author: jiean001
#########################################################

import os
import torch
import json

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        if self.isTrain:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        else:
            self.save_dir = os.path.join(opt.results_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    def load_combo_network(self, network1, network2, network_label, epoch_label, print_weights=False, ignore_BN=False):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        weights = torch.load(save_path)
        # print weights
        if ignore_BN:
            for key in weights.keys():
                if key.endswith('running_mean'):
                    weights[key].zero_()
                elif key.endswith('running_var'):
                    weights[key].fill_(1.0)
        if print_weights:
            for key in weights.keys():
                print(key, 'pretrained, mean,std:', torch.mean(weights[key]), torch.std(weights[key]))

        keys1 = network1.state_dict().keys()
        weights1 = {}
        for key in keys1:
            weights1[key] = weights[key]
        network1.load_state_dict(weights1)
        weights2 = {}

        keys2 = network2.state_dict().keys()
        keys2_in_weights = list(set(weights.keys()) - set(keys1))
        keys1_last_lyr_number = max([int(key.split(".")[1]) for key in keys1])
        for old_key in keys2_in_weights:
            old_key_i = old_key.split(".")
            lyr_num = str(int(old_key_i[1]) - keys1_last_lyr_number - 1)
            old_key_p2 = old_key.split(''.join([old_key_i[0], '.', old_key_i[1]]))[1]
            new_key = ''.join([old_key_i[0], '.', lyr_num])
            new_key = ''.join([new_key, old_key_p2])
            weights2[new_key] = weights[old_key]

        network2.load_state_dict(weights2)

    def load_base_network(self, network, weights, print_weights=False, ignore_BN=False):
        if ignore_BN:
            for key in weights.keys():
                if key.endswith('running_mean'):
                    weights[key].zero_()
                elif key.endswith('running_var'):
                    weights[key].fill_(1.0)
        if print_weights:
            for key in weights.keys():
                print(key, 'pretrained, mean,std:', torch.mean(weights[key]), torch.std(weights[key]))
        if network:
            network.load_state_dict(weights)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, config_file=None, print_weights=False, ignore_BN=False):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        weights = torch.load(save_path)
        if not config_file:
            self.load_base_network(network, weights, print_weights, ignore_BN)
        else:
            model_dict = network.state_dict()
            # new_weights = {}
            f = open(config_file, 'r')
            S_C_dict = json.load(f)
            for s_key in S_C_dict.keys():
                model_dict[s_key] = weights[S_C_dict[s_key]]
            # model_dict.update(new_weights)
            self.load_base_network(network, model_dict, print_weights, ignore_BN)

    def update_learning_rate():
        pass
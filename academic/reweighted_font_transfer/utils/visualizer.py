#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-06
#
# Author: jiean001
#########################################################

def print_current_errors(epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)

    print(message)
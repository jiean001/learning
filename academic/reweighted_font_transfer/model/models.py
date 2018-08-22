#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-08-10
#
# Author: jiean001
#########################################################

def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'classifier':
        from .classifier_model import Classifier_Model
        model = Classifier_Model()
    elif opt.model == 'reweighted_gan':
        from .reweighted_gan_model import Reweighted_GAN
        model = Reweighted_GAN()
    elif opt.model == 'reweighted_l':
        from .reweighted_L_model import Reweighted_L
        model = Reweighted_L()
    elif opt.model == 'reweighted_lsgan':  # 包含了l和gan的case
        from .reweighted_lsgan_model import Reweighted_LSGAN
        model = Reweighted_LSGAN()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

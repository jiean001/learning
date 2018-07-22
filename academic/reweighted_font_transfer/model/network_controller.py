#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-05
#
# Author: jiean001
#########################################################

from .networks import *
from torch.autograd import Variable

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


def define_G(input_nc=3, output_nc=26, ngf=64, norm='batch', use_dropout=False, gpu_ids=[],
             which_model_netG='reweighted_gan', constant_cos=2):
    net_G = None
    if gpu_ids:
        use_gpu = len(gpu_ids) > 0
    else:
        use_gpu = False

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'reweighted_gan':
        net_G = Generator_Reweighted(input_nc=input_nc, output_nc=output_nc, ngf=ngf, norm=norm, use_dropout=use_dropout, gpu_ids=gpu_ids, constant_cos=constant_cos)
    else:
        print('GAN G model name [%s] is not recognized' %which_model_netG)

    if use_gpu:
        net_G.cuda(device=gpu_ids[0])
    net_G.apply(weights_init)
    return net_G

def define_preNet(input_nc, nif=32, which_model_preNet='2_layers', norm='batch', gpu_ids=[]):
    preNet = None
    norm_layer = get_norm_layer(norm_type=norm)
    use_gpu = len(gpu_ids) > 0
    if which_model_preNet == '2_layers':
        print("2 layers convolution applied before being fed into the discriminator")
        preNet = InputTransformation(input_nc, nif, norm_layer, gpu_ids)
        if use_gpu:
            assert(torch.cuda.is_available())
            preNet.cuda(device=gpu_ids[0])
        preNet.apply(weights_init)
    return preNet


def define_D(input_nc, ndf, which_model_netD, is_RGB,
             n_layers_D=3, norm='batch', use_sigmoid=False, postConv=True, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, is_RGB=is_RGB, n_layers=3, use_sigmoid=use_sigmoid,norm_layer=norm_layer, norm_type=norm, postConv=postConv, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, is_RGB=is_RGB, use_sigmoid=use_sigmoid,norm_layer=norm_layer, norm_type=norm, postConv=postConv, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
    netD.apply(weights_init)
    return netD


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)

    print('Total number of parameters: %d' % num_params)
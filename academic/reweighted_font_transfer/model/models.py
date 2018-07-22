################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################


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
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

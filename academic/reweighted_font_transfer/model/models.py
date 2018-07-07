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
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

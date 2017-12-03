from glob import glob
import os
from utils import *
import numpy as np
import random

DEBUG = False
class Data:
    def __init__(self,
                 sampler_images="../../../../data/m_dcgan/sample/format_255",
                 standard_pic_dir="../../../../data/m_dcgan/standard/format_255",
                 input_fname_pattern="*.jpg",
                 config_file = "config.txt",
                 train_config_file="./runconfig/format_255_train_config.txt",
                 create_frain_config=True,
                 batch=500):
        """
        :param sampler_images:
        :param standard_pic_dir:
        :param input_fname_pattern:
        :param config_file:
        :param train_config_file:
        :param create_frain_config:
        :param batch:
        """
        self.sampler_images = sampler_images
        self.standard_pic_dir = standard_pic_dir
        self.input_fname_pattern = input_fname_pattern
        self.config_file = config_file
        self.train_config_file = train_config_file
        self.data = glob(os.path.join(self.sampler_images, self.input_fname_pattern))
        self.data.sort()
        self.batch = batch

        self.style_dic = {}
        self.get_style_dictionary()
        self.img_data = []

        if create_frain_config:
            self.fp = open(self.train_config_file, "w+")
            self.save_train_config()
            self.fp.close()
        self.init_img_data()

    def create_files(self):
        i = 0
        for line in self.img_data:
            i += 1
            oldname = os.path.join(self.sampler_images, line[2])
            newname = ("./data/temp/pic%d.jpg" %i)
            shutil.copyfile(oldname, newname)

    # inital the dictionary
    def get_style_dictionary(self):
        bf = open(os.path.join(self.sampler_images, self.config_file)).read().encode("utf-8").splitlines()
        value_ = []
        for idx in bf:
            idx = idx.decode("utf-8")
            spt = idx.split(":")
            assert spt.__len__() == 2
            if spt[0].strip() == "value":
                value_.append(spt[1].strip())
            elif spt[0].strip() == "key":
                self.style_dic[spt[1].strip()] = value_
                value_ = []
            else:
                assert 1 != 1

    def save_train_config(self):
        for key, value in self.style_dic.items():
            assert type(value) == list and value.__len__() > 1
            value_len = len(value)
            for i in range(value_len):
                for j in range(i + 1, value_len):
                    sample_img = value[i]
                    label_img = value[j]
                    self.fp.write("%s : %s : %s\n" % (sample_img, self.get_standard_img_name(label_img), label_img))
                    self.fp.write("%s : %s : %s\n" % (label_img, self.get_standard_img_name(sample_img), sample_img))

    def get_standard_img_name(self, label_img):
        spt = label_img.split("_")
        assert spt.__len__() == 3
        return spt[1] + ".jpg"

    def init_img_data(self):
        bf = open(self.train_config_file).read().encode("utf-8").splitlines()
        for idx in bf:
            idx = idx.decode("utf-8")
            spt = idx.split(":")
            assert spt.__len__() == 3
            line = [spt[0].strip(), spt[1].strip(), spt[2].strip()]
            self.img_data.append(line)

    def get_batch_data(self, index, input_height=28, input_width=28, resize_height=28,
                       resize_width=28, crop=True, grayscale=False):
        assert index*self.batch < self.img_data.__len__()
        lines = self.img_data[index*self.batch: min((index+1)*self.batch, len(self.img_data))]
        batch_labels = []
        batch_labels.clear()
        for line in lines:
            label_img = get_image(os.path.join(self.sampler_images, line[2]), input_height=input_height,
                                  input_width=input_width, resize_height=resize_height,
                                  resize_width=resize_width, crop=crop, grayscale=grayscale)
            batch_labels.append(label_img)
        return np.array(batch_labels).astype(np.float32)

    """
    def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)
    """

    def get_batch_data(self, count, sess):
        assert count*self.batch < self.img_data.__len__()
        lines = self.img_data[count*self.batch: min((count+1)*self.batch, len(self.img_data))]
        #batch_imgs = []
        batch_labels = []
        for line in lines:
            #sample_img = imread(os.path.join(self.sampler_images, line[0]), grayscale=False)
            #standard_img = imread(os.path.join(self.standard_pic_dir, line[1]), grayscale=True)
            label_img = imread(os.path.join(self.sampler_images, line[2]), grayscale=False)
            batch_labels.append(np.array(label_img) / 127.5 - 1.)
           # _tensor_img = sess.run(get_4d_pic(sample_img, standard_img))
           # tensor_img = np.array(_tensor_img) / 127.5 - 1.
           # batch_imgs.append(tensor_img)
        # return batch_imgs, np.array(batch_labels).astype(np.float32)
        return None, np.array(batch_labels).astype(np.float32)
    def get_len(self):
        return self.img_data.__len__()

    def shuffle(self):
        random.shuffle(self.img_data)

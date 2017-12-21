from glob import glob
import os
import sys
sys.path.append("../common")
from utils import *
import numpy as np
import random
import shutil
import math


DEBUG = False


class Data:
    def __init__(self,
                 sampler_images="../../../../data/m_dcgan/sample/format_255",
                 standard_pic_dir="../../../../data/m_dcgan/standard/format_255",
                 input_fname_pattern="*.jpg",
                 config_file="config.txt",
                 train_config_file="./runconfig/format_255_train_config.txt",
                 create_train_config=True,
                 batch=64,
                 test_rate=0.02
                 ):
        self.sampler_images = sampler_images
        self.standard_pic_dir = standard_pic_dir
        self.input_fname_pattern = input_fname_pattern
        self.config_file = config_file
        self.train_config_file = train_config_file
        self.data = glob(os.path.join(self.sampler_images, self.input_fname_pattern))
        self.data.sort()
        self.batch = batch
        self.test_rate = test_rate


        self.style_dic = {}
        self.get_style_dictionary()
        self.img_data = []
        self.train_data = []
        self.test_data = []

        if create_train_config:
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
        self.shuffle()
        self.test_data = self.img_data[0: int(self.get_len() * self.test_rate)]
        self.train_data = self.img_data[int(self.get_len() * self.test_rate):-1]

    def get_batch_data(self, index, train=True):
        if train:
            assert self.get_train_batch_idxs() > index
            lines = self.train_data[index * self.batch: (index + 1) * self.batch]
        else:
            assert self.get_test_batch_idxs() > index
            lines = self.test_data[index * self.batch: (index + 1) * self.batch]
        batch_samples = []
        batch_standards = []
        batch_labels = []
        for line in lines:
            sample_img = np.array(imread(os.path.join(self.sampler_images, line[0]), grayscale=False)) / 127.5 - 1.
            standard_img = np.array(imread(os.path.join(self.standard_pic_dir, line[1]), grayscale=True)) / 127.5 - 1.
            label_img = np.array(imread(os.path.join(self.sampler_images, line[2]), grayscale=False)) / 127.5 - 1.
            batch_samples.append(sample_img)
            batch_standards.append(standard_img)
            batch_labels.append(label_img)
        return np.array(batch_samples).astype(np.float32), np.array(batch_standards).astype(np.float32), np.array(
            batch_labels).astype(np.float32)

    def get_train_len(self):
        return self.train_data.__len__()

    def get_test_len(self):
        return self.test_data.__len__()

    def get_train_batch_idxs(self):
        return int(math.ceil(self.get_train_len() / self.batch)) - 1

    def get_test_batch_idxs(self):
        return int(math.ceil(self.get_test_len() / self.batch)) - 1

    def get_len(self):
        return self.img_data.__len__()

    def shuffle(self):
        random.shuffle(self.img_data)

    def train_shuffle(self):
        random.shuffle(self.train_data)

    def get_test_data(self, index):
        return self.get_batch_data(index=index, train=False)

    def get_train_data(self, index):
        return self.get_batch_data(index=index, train=True)


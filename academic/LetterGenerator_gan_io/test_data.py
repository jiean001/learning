from glob import glob
import os
from utils import *
import numpy as np
import random

DEBUG = False
class TestData:
    def __init__(self,
                 test_images="../../../../data/m_dcgan/test/format_255",
                 standard_pic_dir="../../../../data/m_dcgan/standard/format_255",
                 input_fname_pattern="*.jpg",
                 test_config_file="./runconfig/format_255_test_config.txt",
                 create_frain_config=True,
                 batch = 64
                 ):

        self.test_images = test_images
        self.standard_pic_dir = standard_pic_dir
        self.input_fname_pattern = input_fname_pattern
        self.test_config_file = test_config_file
        self.batch = batch
        self.test_data = glob(os.path.join(self.test_images, self.input_fname_pattern))
        self.standard_data = glob(os.path.join(self.standard_pic_dir, self.input_fname_pattern))


        self.input_data = []

        if create_frain_config:
            self.fp = open(self.test_config_file, "w+")
            self.save_test_config()
            self.fp.close()
        self.init_input_data()

    def save_test_config(self):
        for test_img in self.test_data:
            i = 0
            self.standard_data.sort()
            for standard_img in self.standard_data:
                i += 1
                assert not i > self.batch
                self.fp.write("%s : %s\n" %(test_img, standard_img))
            random.shuffle(self.standard_data)
            while(self.batch > i):
                i += 1
                self.fp.write("%s : %s\n" % (test_img, self.standard_data[i % 62]))

    def init_input_data(self):
        bf = open(self.test_config_file).read().encode("utf-8").splitlines()
        for idx in bf:
            idx = idx.decode("utf-8")
            spt = idx.split(":")
            assert spt.__len__() == 2
            line = [spt[0].strip(), spt[1].strip()]
            self.input_data.append(line)

    def get_batch_data(self, count):
        assert count < self.input_data.__len__()
        lines = self.input_data[count*self.batch: min((count+1)*self.batch, len(self.input_data))]
        batch_samples = []
        batch_standards = []
        for line in lines:
            sample_img = np.array(imread(os.path.join(line[0]), grayscale=False)) / 127.5 - 1.
            standard_img = np.array(imread(os.path.join(line[1]), grayscale=True)) / 127.5 - 1.
            batch_samples.append(sample_img)
            batch_standards.append(standard_img)
        return np.array(batch_samples).astype(np.float32), np.array(batch_standards).astype(np.float32)

    def get_len(self):
        return self.test_data.__len__()

    def shuffle(self):
        random.shuffle(self.test_data)

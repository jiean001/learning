import tensorflow as tf
import numpy as np
import scipy.misc
from glob import glob
import os
import cv2

DEBUG = True
FORMAT_255_CONFIG = "config.txt"
TRAIN_CONFIG_B = "train_config_b.txt"
TRAIN_CONFIG_W = "train_config_w.txt"

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)


def imread(path, grayscale=False):
    if(grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def transform(image):
    cropped_image = scipy.misc.imresize(image, [96, 96])
    return np.array(cropped_image)/127.5 - 1.


def save_standard_img(inputfile, output_height=96, output_weight=96, output_path="../../../../data/m_dcgan/standard"):
    img_basename = os.path.basename(inputfile)
    in_image = cv2.imread(inputfile, cv2.IMREAD_GRAYSCALE).copy()
    format_255 = cv2.resize(in_image, (output_height, output_weight), interpolation=cv2.INTER_NEAREST)
    format_0 = 255 - format_255
    cv2.imwrite(output_path + "/b_0/" + img_basename, format_0)
    cv2.imwrite(output_path + "/b_255/" + img_basename, format_255)


def get_4d_pic(rgb_pic, gray_pic):
    assert rgb_pic.shape[0] == gray_pic.shape[0] and rgb_pic.shape[1] == gray_pic.shape[1]
    four_d_pic = concat([rgb_pic, gray_pic.reshape(gray_pic.shape[0], gray_pic.shape[1], 1)], axis=2)
    return four_d_pic


class Data:
    def __init__(self, sess, sampler_images="../../../../data/m_dcgan/sample/format_255", standard_pic_dir="../../../../data/m_dcgan/standard/",
                 input_fname_pattern="*.jpg", train_config_path="../../../../data/m_dcgan/train_config", create_frain_config=False,
                 batch=100):
        self.sess = sess
        self.sampler_images = sampler_images
        self.standard_pic_dir = standard_pic_dir
        self.input_fname_pattern = input_fname_pattern
        self.data = glob(os.path.join(self.sampler_images, self.input_fname_pattern))
        self.data.sort()
        self.batch = batch

        self.style_dic = {}
        self.get_style_dictionary()
        self.img_data = []

        self.train_config_b = train_config_path + TRAIN_CONFIG_B
        self.train_config_w = train_config_path + TRAIN_CONFIG_W

        if create_frain_config:
            self.tcb_fp = open(self.train_config_b, "w+")
            self.tcw_fp = open(self.train_config_w, "w+")
            self.save_train_config()
            self.tcb_fp.close()
            self.tcw_fp.close()
        else:
            self.init_img_data()
            # self.get_batch_data(26)

    def init_img_data(self):
        bf = open(self.train_config_w).read().encode("utf-8").splitlines()
        for idx in bf:
            idx = idx.decode("utf-8")
            spt = idx.split(":")
            assert spt.__len__() == 3
            line = [spt[0].strip(), spt[1].strip(), spt[2].strip()]
            self.img_data.append(line)

    def save_train_config_b(self, sample_img, standard_img, label_img):
        self.tcb_fp.write("%s : %s : %s\n" %(sample_img, standard_img, label_img))

    def save_train_config_w(self, sample_img, standard_img, label_img):
        self.tcw_fp.write("%s : %s : %s\n" % (sample_img, standard_img, label_img))

    # inital the dictionary
    def get_style_dictionary(self):
        bf = open(os.path.join(self.sampler_images, FORMAT_255_CONFIG)).read().encode("utf-8").splitlines()
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

    def get_standard_img_name(self, label_img):
        spt = label_img.split("_")
        assert spt.__len__() == 3
        return spt[1] + ".jpg"

    def save_train_config(self):
        for key, value in self.style_dic.items():
            assert type(value) == list and value.__len__() > 1
            value_len = len(value)
            for i in range(value_len):
                for j in range(i+1, value_len):
                    sample_img = value[i]
                    label_img = value[j]
                    self.save_train_config_w(sample_img=sample_img, standard_img=self.get_standard_img_name(label_img),
                                             label_img=label_img)
                    self.save_train_config_w(sample_img=label_img, standard_img=self.get_standard_img_name(sample_img),
                                             label_img=sample_img)

    def get_batch_data(self, count, sess):
        assert count*self.batch < self.img_data.__len__()
        lines = self.img_data[count*self.batch: min((count+1)*self.batch, len(self.img_data))]
        batch_imgs = []
        batch_labels = []
        batch_imgs.clear()
        batch_labels.clear()
        for line in lines:
            sample_img = imread(os.path.join(self.sampler_images, line[0]), grayscale=False)
            standard_img = imread(os.path.join(self.standard_pic_dir, "b_255", line[1]), grayscale=True)
            label_img = imread(os.path.join(self.sampler_images, line[2]), grayscale=False)
            batch_labels.append(label_img)
            a = sess.run(get_4d_pic(sample_img, standard_img))
            batch_imgs.append(a)
        if DEBUG:
            print("get_batch_data begin")
            print(batch_imgs[0])
            print("get_batch_data end")
            
            #print(batch_labels)
        return batch_imgs, batch_labels

'''
if __name__ == '__main__':
    Data()
    # get the standard image


    standard_imgs = glob(os.path.join("../data/m_dcgan/standard/o_b_255", "*.jpg"))
    standard_imgs.sort()
    for img in standard_imgs:
        save_standard_img(img)
    '''

from glob import glob
import os
from utils import *


class Standard:
    def __init__(self, standard_img_path="../../../../data/m_dcgan/standard", standard_img_origin_path="origin",
                 output_black_path="format_0", output_white_path="format_255", output_height=28, output_width=28,
                 input_fname_pattern="*.jpg"):
        """
        :param standard_img_path:
        :param standard_img_origin_path:
        :param output_black_path:
        :param output_white_path:
        :param output_height:
        :param output_weight:
        :param input_fname_pattern:
        """
        self.standard_img_path = standard_img_path
        self.standard_img_origin_path = standard_img_origin_path
        self.output_black_path = output_black_path
        self.output_white_path = output_white_path
        self.output_height = output_height
        self.output_width = output_width
        self.input_fname_pattern = input_fname_pattern

        origin_imgs = glob(os.path.join(self.standard_img_path, self.standard_img_origin_path, self.input_fname_pattern))
        origin_imgs.sort()
        for img in origin_imgs:
            save_standard_img(inputfile=img, output_height=self.output_height, output_width=self.output_width,
                              output_black_path=os.path.join(self.standard_img_path, self.output_black_path),
                              output_white_path=os.path.join(self.standard_img_path, self.output_white_path))
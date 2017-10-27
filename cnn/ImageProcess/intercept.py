import os
import path
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageProcess:
    def __init__(self, gt_text_dir: str, result_dir: str, image_dir: str, background_color: int, format_result_dir: str) -> object:
        # ground truth directory
        self.gt_text_dir = gt_text_dir

        # where to save the images
        self.result_dir = result_dir
        self.format_result_dir = format_result_dir

        # original images directory
        self.image_dir = image_dir
        self.imgDirs = []
        self.imgLists = glob.glob(self.image_dir)

        # the format of output image
        self.format_img_weigth = 96
        self.format_img_height = 96

        # the color of output image
        self.background_color = background_color

        # global parameters
        self.letter_list = []
        self.current_img = ""
        self.gt_img_pix = []
        self.spt = []
        self.dir_num = 0

        # the format line is:
        #  0   1   2   3   4   5   6   7   8   9
        # 108 225 132 168 147 158 131 187 172 "F"
        # temp parameter
        self.top_left = np.zeros((1, 2))
        self.bottom_right = np.zeros((1, 2))

        # what is the meaning of index
        self.gt_img_pix_index = [0, 1, 2]
        self.top_left_index = [5, 6]
        self.bottom_right_index = [7, 8]
        self.letter_index = 9

        self.init_imgDirs()
        self.fp = open(result_dir + "./log2.txt", "w+")
        self.process()
        self.fp.close()

    def init_imgDirs(self):
        for item in self.imgLists:
            self.imgDirs.append(item)

    # 画边框
    def draw_rectangle(self):
        print(self.top_left)
        print(self.bottom_right)
        x1 = []
        y1 = []
        x1.append(self.top_left[0])
        x1.append(self.bottom_right[0])
        x1.append(self.bottom_right[0])
        x1.append(self.top_left[0])
        x1.append(self.top_left[0])

        y1.append(self.top_left[1])
        y1.append(self.top_left[1])
        y1.append(self.bottom_right[1])
        y1.append(self.bottom_right[1])
        y1.append(self.top_left[1])
        # plt.plot(x1, y1, 'r', label='border')

    def delete_img_background_on_onechannel(self, img_onechannel, gt_img_onechannel, channel):
        signal_pix = self.background_color
        out_height = img_onechannel.shape[0]
        out_weight = img_onechannel.shape[1]
        out_img = img_onechannel[0: out_height, 0: out_weight].copy()
        # np.set_printoptions(threshold=1e6)

        for i in range(out_height):
            for j in range(out_weight):
                if gt_img_onechannel[i][j] != self.gt_img_pix[channel]:
                    out_img[i][j] = signal_pix
        return out_img

    def get_pic_name(self, letter):
        try:
            letter = eval(letter.decode())
        except SyntaxError:
            msg = self.result_dir + self.current_img + "_" + str(letter) + ".jpg"
            self.fp.write("error:" + msg + "\n")
        self.letter_list.append(letter)
        letter_num = int(self.letter_list.count(letter))
        return self.result_dir + self.current_img + "_" + str(letter) + "_" + str(letter_num) + ".jpg"

    def get_format_pic_name(self, letter):
        return self.get_pic_name(letter).replace(self.result_dir, self.format_result_dir)

    def get_letter(self, img, gt_img):
        letter_height = self.bottom_right[1] - self.top_left[1]
        letter_weight = self.bottom_right[0] - self.top_left[0]

        img_b = img[self.top_left[1]: self.top_left[1] + letter_height,
                self.top_left[0]: self.top_left[0] + letter_weight, 0]
        img_g = img[self.top_left[1]: self.top_left[1] + letter_height,
                self.top_left[0]: self.top_left[0] + letter_weight, 1]
        img_r = img[self.top_left[1]: self.top_left[1] + letter_height,
                self.top_left[0]: self.top_left[0] + letter_weight, 2]

        gt_img_b = gt_img[self.top_left[1]: self.top_left[1] + letter_height,
                   self.top_left[0]: self.top_left[0] + letter_weight, 0]
        gt_img_g = gt_img[self.top_left[1]: self.top_left[1] + letter_height,
                   self.top_left[0]: self.top_left[0] + letter_weight, 1]
        gt_img_r = gt_img[self.top_left[1]: self.top_left[1] + letter_height,
                   self.top_left[0]: self.top_left[0] + letter_weight, 2]

        letter_b = self.delete_img_background_on_onechannel(img_b, gt_img_b, 0)
        letter_g = self.delete_img_background_on_onechannel(img_g, gt_img_g, 1)
        letter_r = self.delete_img_background_on_onechannel(img_r, gt_img_r, 2)

        letter_matrix = np.dstack([letter_b, letter_g, letter_r])
        plt.imshow(letter_matrix)
        name = self.get_pic_name(self.spt[self.letter_index])
        format_letter = cv2.resize(letter_matrix, (self.format_img_weigth, self.format_img_height))
        # print(name, letter_matrix.shape)
        # plt.figure(figsize=(48, 48))
        try:
            plt.savefig(name)
            save_filename = name.replace(self.result_dir, self.format_result_dir)
            # save_filename = self.get_format_pic_name(self.spt[self.letter_index])
            cv2.imwrite(save_filename, format_letter)
            # cv2.imshow('img', format_letter)
        except FileNotFoundError:
            self.fp.write("error: " + name + "\n")

    def process(self):
        for img_dir in self.imgDirs:
            self.fp.write("\n" + str(self.dir_num) + img_dir + ":\n")
            self.dir_num += 1
            # if (self.dir_num < 2) and (self.background_color == 255):
            if self.dir_num > 1:
                exit()
            img_basename = os.path.basename(img_dir)
            (img_name, tmp) = os.path.splitext(img_basename)
            img_gt_text_name = img_name + "_GT.txt"
            img = plt.imread(img_dir)
            gt_img = plt.imread(self.gt_text_dir + "/" + img_gt_text_name.replace("txt", "bmp"))
            # gt_img = plt.imread(img_dir.replace(".jpg", "_GT.bmp"))
            self.current_img = img_name
            self.letter_list = []
            bf = open(os.path.join(self.gt_text_dir, img_gt_text_name)).read().encode("utf-8").splitlines()
            print(self.current_img)
            # plt.imshow(img1)
            lines = []
            for idx in bf:
                if lines.__contains__(idx):
                    continue  # delete the repetition
                lines.append(idx)
                self.spt = idx.split()
                if self.spt.__len__() == 10:
                    self.top_left = []
                    self.bottom_right = []
                    self.gt_img_pix = [int(self.spt[self.gt_img_pix_index[0]]), int(self.spt[self.gt_img_pix_index[1]]),
                                       int(self.spt[self.gt_img_pix_index[2]])]
                    self.top_left.append(int(self.spt[self.top_left_index[0]]))
                    self.top_left.append(int(self.spt[self.top_left_index[1]]))
                    self.bottom_right.append(int(self.spt[self.bottom_right_index[0]]))
                    self.bottom_right.append(int(self.spt[self.bottom_right_index[1]]))
                    # draw_rectangle(top_left, bottom_right)
                    self.get_letter(img, gt_img)
                    # plt.show()


if __name__ == '__main__':
    gt_text_dir = "../../data/Challenge2_Training_Task2_GT"
    result_dir = "../../result/Challenge2_Training_Task12_Images/bkgrd_0/"
    image_dir = "../../data/Challenge2_Training_Task12_Images/*.jpg"
    format_result_dir = "../../result/Challenge2_Training_Task12_Images/format_bkgrd_0/"
    background_color = 0

    print("background is 0")
    ImageProcess(gt_text_dir = gt_text_dir, result_dir = result_dir, format_result_dir = format_result_dir,
               image_dir = image_dir, background_color = background_color)

    result_dir = "../../result/Challenge2_Training_Task12_Images/bkgrd_255/"
    background_color = 255
    format_result_dir = "../../result/Challenge2_Training_Task12_Images/format_bkgrd_255/"
    print("background is 255")
    ImageProcess(gt_text_dir=gt_text_dir, result_dir=result_dir, format_result_dir=format_result_dir,
                 image_dir=image_dir, background_color=background_color)

    '''
    gt_text_dir = "../../images"
    result_dir = "../../tmp_result/"
    format_result_dir = "../../tmp_result_format/"
    image_dir = "../../images/*.jpg"
    background_clor = 255
    ImageProcess(gt_text_dir = gt_text_dir, result_dir = result_dir, format_result_dir = format_result_dir, image_dir = image_dir, background_clor = background_clor)
    '''


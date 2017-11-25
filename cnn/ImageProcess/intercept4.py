import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import time

GT_R = 0
GT_G = 1
GT_B = 2
CV_R = 2
CV_G = 1
CV_B = 0
PLT_R = 0
PLT_G = 1
PLT_B = 2
COLOR_WRITE = 255
COLOR_BLACK = 0
MIN_DELTA_COLOR = 10
MIN_RAW = 0
MIN_COLUMN = 0


class ImageProcess:
    # gt_did: ground truth image's directory       eg: "../../data/Challenge2_Training_Task2_GT"
    # s_dir : the source image's directory         eg: "../../data/Challenge2_Training_Task12_Images/*.jpg"
    # output_dir: the output images's directory    eg: "../../result/Challenge2_Training_Task12_Images/tmp_0/"
    # format_size: the format size of output image eg: [96, 96]
    def __init__(self, gt_dir, s_dir, output_dir, format_size=[96, 96]):
        # initial parameter
        self.gt_dir = gt_dir
        self.s_dir = s_dir
        self.output_dir = output_dir
        self.format_size = format_size
        self.log_file = self.output_dir + "log" + str(time.time()) + ".txt"

        # original images directory
        self.imgDirs = []
        self.imgLists = glob.glob(self.s_dir)

        # the output message
        msg = ("gt_dir: %s\ns_dir: %s\nformat_size: %s\nlog_file: %s\nfinal_result_dir: " %(self.gt_dir, self.s_dir, str(self.format_size), self.log_file))

        # global parameters
        self.style_count = 1
        self.letter_list = []
        self.current_img = ""
        self.gt_img_pix = []
        self.spt = []
        self.dir_num = 0
        self.style_num = 0
        self.line = ""
        self.letter =""

        # the format line is:
        #  0R  1G  2B   3   4   5   6   7   8   9
        # 108 225 132 168 147 158 131 187 172 "F"
        # temp parameter
        self.top_left = np.zeros((1, 2))
        self.bottom_right = np.zeros((1, 2))

        # what is the meaning of index
        self.gt_img_pix_index = [0, 1, 2]
        self.top_left_index = [5, 6]
        self.bottom_right_index = [7, 8]
        self.letter_index = 9

        # open the log file
        self.fp = open(self.log_file, "w+")

        # the background color
        self.bkgrd_color = [COLOR_BLACK, COLOR_WRITE]
        self.final_result_dir = []
        for i in self.bkgrd_color:
            tmp_dir = ("%sorigin_%d/" %(self.output_dir, i))
            self.final_result_dir.append(tmp_dir)
            msg = msg + tmp_dir + "\n\t"
            self.make_dir(tmp_dir)
            tmp_dir = ("%sformat_%d/" % (self.output_dir, i))
            self.final_result_dir.append(tmp_dir)
            msg = msg + tmp_dir + "\n\t"
            self.make_dir(tmp_dir)
        print(msg)

        self.log(msg)
        self.init_imgDirs()

        self.process()
        # close the log file
        self.fp.close()

    def init_imgDirs(self):
        for item in self.imgLists:
            self.imgDirs.append(item)
        self.log("init image directory is success")

    def log(self, record):
        self.fp.write(record + "\n")

    def make_dir(self, pt):
        pt = pt.strip()
        pt = pt.rstrip("\\")
        is_exists = os.path.exists(pt)

        if not is_exists:
            os.makedirs(pt)
            self.log("create directory: " + pt + " is success")
            return True
        else:
            self.log("the directory " + pt + " is exists")
            print()
        return False

    def add_style(self, is_new_file=False):
        if is_new_file:
            self.style_count = 1
        elif len(self.letter_list) > 1:
            self.add_style_log()
            self.style_count += 1
        self.letter_list = []

    def add_style_log(self):
        letter_num = len(self.letter_list)
        self.log("style%d have %d letters" % (self.style_count, letter_num))
        if letter_num < 2:
            self.log("error! not enough letters")
            print("--------------error! not enough letters-----------------")

    def get_pic_name(self, current_dir, isFirst):
        try:
            letter = self.letter  # = eval(letter.decode())
        except SyntaxError:
            msg = current_dir + self.current_img + "_" + str(letter) + ".jpg"
            self.log("error:" + self.line)
            self.log("error:" + msg)
        if isFirst:
            self.letter_list.append(letter)
        letter_num = int(self.letter_list.count(letter))
        return current_dir + self.current_img + "_style_" + str(self.style_count) + "_" + str(letter) + "_" + str(letter_num) + ".jpg"

    def get_filter_0(self, gt_image):
        gt_letter_height = self.bottom_right[1] - self.top_left[1]
        gt_letter_width = self.bottom_right[0] - self.top_left[0]
        gt_letter = gt_image[self.top_left[1]: self.top_left[1] + gt_letter_height, self.top_left[0]: self.top_left[0] + gt_letter_width, :]
        filter_0 = np.zeros((gt_letter_height, gt_letter_width, 3))
        for i in range(gt_letter_height):
            for j in range(gt_letter_width):
                if ((gt_letter[i][j][PLT_B] == self.gt_img_pix[CV_B]) and
                        (gt_letter[i][j][PLT_G] == self.gt_img_pix[CV_G]) and (gt_letter[i][j][PLT_R] == self.gt_img_pix[CV_R])):
                    filter_0[i][j][:] = 1
        return filter_0

    def get_origin_letter_0(self, filter, image):
        img_letter_height = self.bottom_right[1] - self.top_left[1]
        img_letter_width = self.bottom_right[0] - self.top_left[0]
        img_letter = image[self.top_left[1]: self.top_left[1] + img_letter_height, self.top_left[0]: self.top_left[0] + img_letter_width, :]
        letter_0 = img_letter * filter
        return letter_0

    def get_origin_letter_255(self, filter_0, letter_0):
        filter_255 = (filter_0 + COLOR_WRITE) % (COLOR_WRITE + 1)
        letter_255 = letter_0 + filter_255
        return letter_255

    def get_mean_color(self, letter_0, filter_0):
        return sum(sum(sum(letter_0))) / sum(sum(sum(filter_0)))

    def is_save_size(self, filter_0):
        if (filter_0.shape[0] < MIN_RAW) and (filter_0.shape[1] < MIN_COLUMN):
            self.log("NO SAVE: %s, the shape(%d, %d) is too small." %(self.line, filter_0.shape[0], filter_0.shape[1]))
            return False
        return True

    def is_save_color(self, letter_0, filter_0):
        mean_color = self.get_mean_color(letter_0, filter_0)
        save_0 = True
        save_255 = True
        if abs(COLOR_BLACK - mean_color) < MIN_DELTA_COLOR:
            self.log(
                "NO SAVE: %s, The color(%f, %f) of the gap is too small." % (self.line, mean_color, COLOR_BLACK))
            save_0 = False
        if abs(COLOR_WRITE - mean_color) < MIN_DELTA_COLOR:
            self.log(
                "NO SAVE: %s, The color(%f, %f) of the gap is too small." % (self.line, mean_color, COLOR_WRITE))
            save_255 = False
        return save_0, save_255

    def get_letter(self, img, gt_img):
        filter_0 = self.get_filter_0(gt_img)
        if not self.is_save_size(filter_0):
            return

        letter_0 = self.get_origin_letter_0(filter_0, img)
        letter_255 = self.get_origin_letter_255(filter_0, letter_0)
        save_0, save_255 = self.is_save_color(letter_0, filter_0)

        if save_0:
            origin_name_0 = self.get_pic_name(self.final_result_dir[0], True)
            try:
                cv2.imwrite(origin_name_0, letter_0)
                self.log("the shape of %s is (%d, %d) " %(origin_name_0, letter_0.shape[0], letter_0.shape[1]))
            except FileNotFoundError:
                self.log("error:" + self.line)
                self.log("error: " + origin_name_0)
            format_name_0 = origin_name_0.replace(self.final_result_dir[0], self.final_result_dir[1])
            format_letter_0 = cv2.resize(letter_0, (self.format_size[0], self.format_size[1]), interpolation=cv2.INTER_NEAREST)
            try:
                cv2.imwrite(format_name_0, format_letter_0)
            except FileNotFoundError:
                self.log("error:" + self.line)
                self.log("error: " + format_name_0)

        if save_255:
            origin_name_255 = self.get_pic_name(self.final_result_dir[2], False)
            try:
                cv2.imwrite(origin_name_255, letter_255)
            except FileNotFoundError:
                self.log("error:" + self.line)
                self.log("error: " + origin_name_255)
            format_name_255 = origin_name_255.replace(self.final_result_dir[2], self.final_result_dir[3])
            format_letter_255 = cv2.resize(letter_255, (self.format_size[0], self.format_size[1]), interpolation=cv2.INTER_NEAREST)
            try:
                cv2.imwrite(format_name_255, format_letter_255)
            except FileNotFoundError:
                self.log("error:" + self.line)
                self.log("error: " + format_name_255)

    def process(self):
        start_time = time.time()
        for img_dir in self.imgDirs:
            self.log("\n" + str(self.dir_num) + img_dir + ":\n")
            self.dir_num += 1
            # if self.dir_num > 1:
                # exit()
            img_basename = os.path.basename(img_dir)
            (img_name, tmp) = os.path.splitext(img_basename)
            img_gt_text_name = img_name + "_GT.txt"
            img = plt.imread(img_dir)  # , 0)
            gt_img = plt.imread(self.gt_dir + "/" + img_gt_text_name.replace("txt", "bmp"))  # , 0)
            self.current_img = img_name
            self.letter_list = []
            self.add_style(is_new_file=True)
            bf = open(os.path.join(self.gt_dir, img_gt_text_name)).read().encode("utf-8").splitlines()
            print("%d %s time: %4.4f" %(229-self.dir_num, self.current_img, time.time() - start_time))
            lines = []
            img_r, img_g, img_b = cv2.split(img)
            gt_img_r, gt_img_g, gt_img_b = cv2.split(gt_img)
            img_cv = cv2.merge([img_b, img_g, img_r])
            gt_img_cv = cv2.merge([gt_img_b, gt_img_g, gt_img_r])
            current_style_num = 0
            last_letter = Last_letter()
            last_line = ""
            for idx in bf:
                self.line = idx
                if not len(idx):
                    self.add_style()
                    if current_style_num == 1:
                        self.log(str(last_line))
                        self.log("error -- there has not enough letter, so we not save this letter")
                        print(str(last_line))
                        print("there has not enough letter, so we not save this letter")
                    current_style_num = 0
                    continue
                self.spt = idx.split()
                if self.spt.__len__() == 10:
                    if lines.__contains__(idx):
                        self.log("error repeat")
                        self.log(str(idx))
                        print(str(idx))
                        print("-------error repeat-------")
                        continue  # delete the repetition
                    lines.append(idx)
                    self.top_left = []
                    self.bottom_right = []
                    self.gt_img_pix = [int(self.spt[self.gt_img_pix_index[0]]), int(self.spt[self.gt_img_pix_index[1]]),
                                       int(self.spt[self.gt_img_pix_index[2]])]
                    self.top_left.append(int(self.spt[self.top_left_index[0]]))
                    self.top_left.append(int(self.spt[self.top_left_index[1]]))
                    self.bottom_right.append(int(self.spt[self.bottom_right_index[0]]))
                    self.bottom_right.append(int(self.spt[self.bottom_right_index[1]]))
                    try:
                        self.letter = self.spt[self.letter_index]
                        self.letter = eval(self.letter.decode())
                        if not self.letter.isalnum():
                            continue
                        '''
                        if letter != 'a':
                            continue
                        '''
                    except SyntaxError:
                        self.log("error SyntaxError--------")
                        print("SyntaxError--------")
                        continue
                    current_style_num += 1
                    if current_style_num == 1:
                        last_letter = Last_letter(top_left=self.top_left, bottom_right=self.bottom_right,
                                    gt_img_pix=self.gt_img_pix, letter=self.letter)
                        last_line = idx
                        continue
                    self.get_letter(img_cv, gt_img_cv)
                    if current_style_num == 2:
                        self.top_left = last_letter.top_left
                        self.bottom_right = last_letter.bottom_right
                        self.gt_img_pix = last_letter.gt_img_pix
                        self.letter = last_letter.letter
                        self.get_letter(img_cv, gt_img_cv)

            self.add_style()


class Last_letter:
    def __init__(self, top_left=[], bottom_right=[], gt_img_pix=[], letter=""):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.gt_img_pix = gt_img_pix
        self.letter = letter


if __name__ == '__main__':
    gt_dir = "../../data/Challenge2_Training_Task2_GT"
    output_dir = "../../result/Challenge2_Training_Task12_Images/"
    s_dir = "../../data/Challenge2_Training_Task12_Images/*.jpg"
    ImageProcess(gt_dir=gt_dir, s_dir=s_dir, output_dir=output_dir)

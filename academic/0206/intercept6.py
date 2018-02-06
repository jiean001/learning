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
MIN_DELTA_COLOR_WRITE = 20
MIN_DELTA_COLOR_BLACK = 20
MIN_RAW = 0
MIN_COLUMN = 0


class ImageProcess:
    # gt_did: ground truth image's directory       eg: "../../data/Challenge2_Training_Task2_GT"
    # s_dir : the source image's directory         eg: "../../data/Challenge2_Training_Task12_Images/*.jpg"
    # output_dir: the output images's directory    eg: "../../result/Challenge2_Training_Task12_Images/tmp_0/"
    # format_size: the format size of output image eg: [96, 96]
    def __init__(self, gt_dir, s_dir, output_dir, format_size=[32, 32]):
        # initial parameter
        self.gt_dir = gt_dir
        self.s_dir = s_dir
        self.output_dir = output_dir
        self.format_size = format_size
        self.log_file = self.output_dir + "log" + str(time.time()) + ".txt"
        self.w_file = self.output_dir + "_white_config.txt"
        self.b_file = self.output_dir + "_black_config.txt"
        self.pytorch_w_config = self.output_dir + "_pytorch_white_config.txt"
        self.pytorch_b_config = self.output_dir + "_pytorch_black_config.txt"

        self.cache_white_letter = []
        self.cache_black_letter = []

        # original images directory
        self.imgDirs = []
        self.imgLists = glob.glob(self.s_dir)

        # the output message
        msg = ("gt_dir: %s\ns_dir: %s\nformat_size: %s\nlog_file: %s\nfinal_result_dir: " %(self.gt_dir, self.s_dir, str(self.format_size), self.log_file))

        # global parameters
        self.style_count_write = 1
        self.style_count_black = 1
        self.letter_list_write = []
        self.letter_list_black = []
        self.current_img = ""
        self.gt_img_pix = []
        self.spt = []
        self.dir_num = 0
        self.line = ""
        self.letter = ""
        self.last_letter_b = Last_letter()
        self.last_letter_w = Last_letter()

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
        self.b_fp = open(self.b_file, "w+")
        self.w_fp = open(self.w_file, "w+")
        self.pytorch_w_config_fp = open(self.pytorch_w_config, "w+")
        self.pytorch_b_config_fp = open(self.pytorch_b_config, "w+")

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
        self.b_fp.close()
        self.w_fp.close()
        self.pytorch_w_config_fp.close()
        self.pytorch_b_config_fp.close()

    def w_2_pytorch_config(self):
        if self.cache_white_letter.__len__() > 1:
            for w_letter in self.cache_white_letter:
                self.pytorch_w_config_fp.write(w_letter + "\t")
            self.pytorch_w_config_fp.write("\n")
        if self.cache_black_letter.__len__() > 1:
            for b_letter in self.cache_black_letter:
                self.pytorch_b_config_fp.write(b_letter + "\t")
            self.pytorch_b_config_fp.write("\n")
        self.cache_white_letter = []
        self.cache_black_letter = []

    def init_imgDirs(self):
        for item in self.imgLists:
            self.imgDirs.append(item)
        self.log("init image directory is success")

    def log(self, record):
        self.fp.write(record + "\n")

    def w_2_w(self, record):
        self.w_fp.write(record + "\n")

    def w_2_b(self, record):
        self.b_fp.write(record + "\n")

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
            self.style_count_write = 1
            self.style_count_black = 1
        else:
            self.w_2_pytorch_config()
            self.add_style_log()
            if len(self.letter_list_black) > 1:
                self.style_count_black += 1
            if len(self.letter_list_write) > 1:
                self.style_count_write += 1
        self.letter_list_black = []
        self.letter_list_write = []

    def add_style_log(self):
        letter_num_write = len(self.letter_list_write)
        letter_num_black = len(self.letter_list_black)
        if letter_num_black > 0:
            self.log("black style%d have %d letters" % (self.style_count_black, letter_num_black))
        if letter_num_write > 0:
            self.log("write style%d have %d letters" % (self.style_count_write, letter_num_write))
        if letter_num_black > 1:
            self.w_2_b("key : %sstyle%d" %(self.current_img, self.style_count_black))
        if letter_num_write > 1:
            self.w_2_w("key : %sstyle%d" %(self.current_img, self.style_count_write))

    def get_pic_name(self, current_dir_b, current_dir_w, is_save_b=True, is_save_w=True):
        letter_name_b = ""
        letter_name_w = ""
        if is_save_b:
            self.letter_list_black.append(self.letter)
            letter_num_b = int(self.letter_list_black.count(self.letter))
            letter_name_b = (
            "%s%sstyle%d_%c_%d.jpg" % (current_dir_b, self.current_img, self.style_count_black, self.letter, letter_num_b))
        if is_save_w:
            self.letter_list_write.append(self.letter)
            letter_num_w = int(self.letter_list_write.count(self.letter))
            letter_name_w = ("%s%sstyle%d_%c_%d.jpg" % (current_dir_w, self.current_img, self.style_count_write, self.letter, letter_num_w))
        return letter_name_b, letter_name_w

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

    def save_letter_b(self, origin_name_0, letter_0):
        try:
            cv2.imwrite(origin_name_0, letter_0)
            self.log("the shape of %s is (%d, %d) " % (origin_name_0, letter_0.shape[0], letter_0.shape[1]))
            letter_basename = os.path.basename(origin_name_0)
            self.w_2_b("value : %s" %(letter_basename))
            self.cache_black_letter.append(letter_basename)
        except FileNotFoundError:
            self.log("error:" + self.line)
            self.log("error: " + origin_name_0)
        format_name_0 = origin_name_0.replace(self.final_result_dir[0], self.final_result_dir[1])
        format_letter_0 = cv2.resize(letter_0, (self.format_size[0], self.format_size[1]),
                                     interpolation=cv2.INTER_NEAREST)
        try:
            cv2.imwrite(format_name_0, format_letter_0)
        except FileNotFoundError:
            self.log("error:" + self.line)
            self.log("error: " + format_name_0)

    def save_letter_w(self, origin_name_255, letter_255):
        try:
            cv2.imwrite(origin_name_255, letter_255)
            letter_basename = os.path.basename(origin_name_255)
            self.w_2_w("value : %s" %(letter_basename))
            self.cache_white_letter.append(letter_basename)
        except FileNotFoundError:
            self.log("error:" + self.line)
            self.log("error: " + origin_name_255)
        format_name_255 = origin_name_255.replace(self.final_result_dir[2], self.final_result_dir[3])
        format_letter_255 = cv2.resize(letter_255, (self.format_size[0], self.format_size[1]),
                                       interpolation=cv2.INTER_NEAREST)
        try:
            cv2.imwrite(format_name_255, format_letter_255)
        except FileNotFoundError:
            self.log("error:" + self.line)
            self.log("error: " + format_name_255)

    def is_save_color(self, letter_0, filter_0):
        mean_color = self.get_mean_color(letter_0, filter_0)
        save_0 = True
        save_255 = True
        if abs(COLOR_BLACK - mean_color) < MIN_DELTA_COLOR_BLACK:
            self.log(
                "NO SAVE: %s, The color(%f, %f) of the gap is too small." % (self.line, mean_color, COLOR_BLACK))
            save_0 = False
        if abs(COLOR_WRITE - mean_color) < MIN_DELTA_COLOR_WRITE:
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

        origin_name_0, origin_name_255 = self.get_pic_name(current_dir_b=self.final_result_dir[0],
                                                           current_dir_w=self.final_result_dir[2],
                                                           is_save_b=save_0, is_save_w=save_255)
        if origin_name_0 != "":
            if len(self.letter_list_black) == 1:
                self.last_letter_b = Last_letter(origin_name=origin_name_0, letter=letter_0)
                cv2.imwrite(origin_name_0, letter_0)
            elif len(self.letter_list_black) == 2:
                self.save_letter_b(origin_name_0=self.last_letter_b.origin_name, letter_0=self.last_letter_b.letter)
                self.save_letter_b(origin_name_0=origin_name_0, letter_0=letter_0)
            else:
                self.save_letter_b(origin_name_0=origin_name_0, letter_0=letter_0)
        if origin_name_255 != "":
            if len(self.letter_list_write) == 1:
                self.last_letter_w = Last_letter(origin_name=origin_name_255, letter=letter_255)
            elif len(self.letter_list_write) == 2:
                self.save_letter_w(origin_name_255=self.last_letter_w.origin_name, letter_255=self.last_letter_w.letter)
                self.save_letter_w(origin_name_255=origin_name_255, letter_255=letter_255)
            else:
                self.save_letter_w(origin_name_255=origin_name_255, letter_255=letter_255)


    def process(self):
        start_time = time.time()
        self.imgDirs.sort()
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
            self.letter_list_black = []
            self.letter_list_write = []
            self.add_style(is_new_file=True)
            bf = open(os.path.join(self.gt_dir, img_gt_text_name)).read().encode("utf-8").splitlines()
            print("%d %s time: %4.4f" %(229-self.dir_num, self.current_img, time.time() - start_time))
            lines = []
            img_r, img_g, img_b = cv2.split(img)
            gt_img_r, gt_img_g, gt_img_b = cv2.split(gt_img)
            img_cv = cv2.merge([img_b, img_g, img_r])
            gt_img_cv = cv2.merge([gt_img_b, gt_img_g, gt_img_r])
            for idx in bf:
                self.line = idx
                if not len(idx):
                    if len(self.letter_list_write) == 1:
                        self.log(str(idx))
                        self.log("error! not enough write letters, so we do not save this letter")
                        print("error! not enough write letters, so we do not save this letter")
                    if len(self.letter_list_black) == 1:
                        self.log(str(idx))
                        self.log("error! not enough black letters, so we do not save this letter")
                        print("error! not enough black letters, so we do not save this letter")
                    self.add_style()
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
                    self.get_letter(img_cv, gt_img_cv)
            self.add_style()

class Last_letter:
    def __init__(self, origin_name="", letter=[]):
        self.origin_name = origin_name
        self.letter = letter

if __name__ == '__main__':
    gt_dir = "../../data/Challenge2_Training_Task2_GT"
    output_dir = "../../result/Challenge2_Training_Task12_Images/"
    s_dir = "../../data/Challenge2_Training_Task12_Images/*.jpg"
    ImageProcess(gt_dir=gt_dir, s_dir=s_dir, output_dir=output_dir)

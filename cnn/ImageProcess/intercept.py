import cv2
import numpy as np
import os
import time
import glob
import matplotlib.pyplot as plt
import time


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
        self.letter_list = []
        self.current_img = ""
        self.gt_img_pix = []
        self.spt = []
        self.dir_num = 0
        self.line = ""

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

        # open the log file
        self.fp = open(self.log_file, "w+")

        # the background color
        self.bkgrd_color = [0, 255]
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

    def get_pic_name(self, current_dir, letter, isFirst):
        try:
            letter = eval(letter.decode())
        except SyntaxError:
            msg = current_dir + self.current_img + "_" + str(letter) + ".jpg"
            self.log("error:" + self.line)
            self.log("error:" + msg)
        if isFirst:
            self.letter_list.append(letter)
        letter_num = int(self.letter_list.count(letter))
        return current_dir + self.current_img + "_" + str(letter) + "_" + str(letter_num) + ".jpg"

    def set_img_background_on_onechannel(self, img_onechannel, gt_img_onechannel, channel, target_bkgrd_color):
        signal_pix = target_bkgrd_color
        out_height = img_onechannel.shape[0]
        out_weight = img_onechannel.shape[1]
        out_img = img_onechannel[0: out_height, 0: out_weight].copy()
        np.set_printoptions(threshold=1e6)

        for i in range(out_height):
            for j in range(out_weight):
                if gt_img_onechannel[i][j] != self.gt_img_pix[channel]:
                    out_img[i][j] = signal_pix
        return out_img

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

        i = -1
        for bkgrd_color in self.bkgrd_color:
            i += 1
            letter_b = self.set_img_background_on_onechannel(img_b, gt_img_b, 0, bkgrd_color)
            letter_g = self.set_img_background_on_onechannel(img_g, gt_img_g, 1, bkgrd_color)
            letter_r = self.set_img_background_on_onechannel(img_r, gt_img_r, 2, bkgrd_color)
            letter_matrix = np.dstack([letter_b, letter_g, letter_r])
            # origin image
            origin_name = self.get_pic_name(self.final_result_dir[i*2], self.spt[self.letter_index], i==0)
            origin_letter = cv2.resize(letter_matrix, (letter_weight, letter_height), interpolation=cv2.INTER_NEAREST)
            try:
                cv2.imwrite(origin_name, origin_letter)
            except FileNotFoundError:
                self.log("error:" + self.line)
                self.log("error: " + origin_name)
            # format image
            format_name = origin_name.replace(self.final_result_dir[i*2], self.final_result_dir[i*2 + 1])
            format_letter = cv2.resize(letter_matrix, (self.format_size[0], self.format_size[1]),
                                       interpolation=cv2.INTER_NEAREST)
            try:
                cv2.imwrite(format_name, format_letter)
            except FileNotFoundError:
                self.log("error:" + self.line)
                self.log("error: " + format_name)

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
            bf = open(os.path.join(self.gt_dir, img_gt_text_name)).read().encode("utf-8").splitlines()
            print("%d %s time: %4.4f" %(328-self.dir_num, self.current_img, time.time() - start_time))
            lines = []
            for idx in bf:
                self.line = idx
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
                    try:
                        letter = self.spt[self.letter_index]
                        letter = eval(letter.decode())
                        if not letter.isalnum():
                            continue
                    except SyntaxError:
                        continue
                    self.get_letter(img, gt_img)





if __name__ == '__main__':
    gt_dir = "../../data/Challenge2_Training_Task2_GT"
    output_dir = "../../result/Challenge2_Training_Task12_Images/"
    s_dir = "../../data/Challenge2_Training_Task12_Images/*.jpg"
    ImageProcess(gt_dir=gt_dir, s_dir=s_dir, output_dir=output_dir)
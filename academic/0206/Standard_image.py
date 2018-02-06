import cv2
import glob
import os

old_dir = "../../DCGAN-tensorflow/data/m_dcgan/standard/new_format_255/*.jpg"
new_dir = "./data/standard/"

def resize_img(height=32, width=32):
    imgLists = glob.glob(old_dir)
    for item in imgLists:
        letter = os.path.basename(item)
        old_img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(old_img, (height, width), interpolation=cv2.INTER_CUBIC)
        new_img_path = os.path.join(new_dir, letter)
        cv2.imwrite(new_img_path, new_img)
        print(new_img_path)

resize_img()

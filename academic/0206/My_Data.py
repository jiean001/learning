from torch.utils.data import Dataset
from PIL import Image
import os

def default_loder(img):
    return Image.open(img)

MAX_BATCH = 64


import torchvision.transforms as transforms
import torch

# transforms.ToTensor()
transform1 = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
)

# [-1.0 , 1.0]
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)


def get_image(path, channel=3):
    if channel == 1:
        return Image.open(path).convert('L')
    if channel == 3:
        return Image.open(path).convert('RGB')


def get_image_tensor(path, channel=3):
    im = get_image(path, channel)
    im2 = transform1(im)
    return im2


def get_images_tensor(path_list, channel=3):
    is_first = True
    for path in path_list:
        im = get_image_tensor(path, channel)
        if is_first:
            is_first = False
            ims = im
        else:
            ims = torch.cat((ims, im))
    return ims.view(-1, 3, ims.size(-2), ims.size(-1))


class My_Data(Dataset):
    def __init__(self, config_file='../0126/pytorch_white_config.txt',
                 appearance_img_path='../../result/Challenge2_Training_Task12_Images/format_255',
                 core_img_path='./data/standard/',
                 img_transform=None, loader=default_loder):
        with open(config_file, "r") as f:
            lines = f.readlines()
            self.appearance_letters_list = []
            self.core_letter_list = []
            self.target_letter_list = []
            self.appearance_letters_count_same_map = {}
            self.max_appearcance_len = -1
            appearance_letter_count_same = 0
            for line in lines:
                word = line.split(":")
                appearance_letters = word[0].split(" ")
                appearance_letter_list = []
                if self.max_appearcance_len == -1:
                    self.max_appearcance_len = appearance_letters.__len__() - 1
                if self.max_appearcance_len < appearance_letters.__len__() - 1:
                    self.appearance_letters_count_same_map[self.max_appearcance_len] = appearance_letter_count_same
                    self.max_appearcance_len = appearance_letters.__len__() - 1
                    appearance_letter_count_same = 1
                else:
                    appearance_letter_count_same += 1
                for j in range(self.max_appearcance_len):
                    appearance_letter_list.append(os.path.join(appearance_img_path, appearance_letters[j]))
                self.core_letter_list.append(os.path.join(core_img_path, word[1].strip()))
                self.target_letter_list.append(os.path.join(appearance_img_path, word[2].strip()))
                self.appearance_letters_list.append(appearance_letter_list)
            self.appearance_letters_count_same_map[self.max_appearcance_len] = appearance_letter_count_same
        self.image_transform = img_transform
        self.loader = loader
        self.reset_index()
        # self.print_map()
        # aa = self.get_batch_data()[0]
        # while(aa):
            # aa = self.get_batch_data()[0]

    def __getitem__(self, index):
        appearance_letter_list = self.appearance_letters_list[index]
        core_letter = self.core_letter_list[index]
        target_letter = self.target_letter_list[index]
        return get_images_tensor(appearance_letter_list), get_image_tensor(core_letter, channel=1), get_image_tensor(target_letter)

    def __len__(self):
        return len(self.target_letter_list)

    def print_map(self):
        for key, values in self.appearance_letters_count_same_map.items():
            print(str(key) + ":" + str(values))

    def reset_index(self):
        self.current_index = 0
        self.current_group_value = 1
        self.current_group_index = 0

    def get_letters_tensor(self, count=MAX_BATCH):
        appearance_letters, core_letter, target_letter = self.__getitem__(self.current_index)
        self.current_index += 1
        for i in range(count-1):
            _appearance_letters, _core_letter, _target_letter = self.__getitem__(self.current_index)
            appearance_letters = torch.cat((appearance_letters, _appearance_letters))
            core_letter = torch.cat((core_letter, _core_letter))
            target_letter = torch.cat((target_letter, _target_letter))
            self.current_index += 1
        appearance_letters = appearance_letters.view(count, -1, 3,  appearance_letters.size(-2), appearance_letters.size(-1))
        core_letter = core_letter.view(count, -1, core_letter.size(-2), core_letter.size(-1))
        target_letter = target_letter.view(count, -1, target_letter.size(-2), target_letter.size(-1))
        # print(appearance_letters.size())
        # print(core_letter.size())
        # print(target_letter.size())
        return appearance_letters.transpose(0, 1), core_letter, target_letter

    def get_batch_data(self):
        if self.appearance_letters_count_same_map[self.current_group_value] - self.current_group_index > MAX_BATCH:
            out = self.get_letters_tensor()
            self.current_group_index += MAX_BATCH
            return True, out[0], out[1], out[2]
        else:
            count = self.appearance_letters_count_same_map[self.current_group_value] - self.current_group_index
            out = self.get_letters_tensor(count=count)
            self.current_group_index += count
            if self.current_group_value == self.max_appearcance_len and self.current_group_index == \
                    self.appearance_letters_count_same_map[self.current_group_value]:
                return False, out[0], out[1], out[2]
            else:
                self.current_group_value += 1
                self.current_group_index = 0
                return True, out[0], out[1], out[2]
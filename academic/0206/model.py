import torch
from torch import nn
from torch.autograd import Variable
import torchvision.utils as vutils
from My_Data import My_Data

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
INPUT_SIZE = 1          # rnn input size / image width
LR = 0.02               # learning rate
EPOCH = 100


class Dymanic_Generate_Letter(nn.Module):
    def __init__(self, nc_appearance=3, nef=16, nc_kernel=1, nkf=16, nc_generate=3, ngf=16, f_size=1024):
        super(Dymanic_Generate_Letter, self).__init__()
        self.extract_appearance_feature(nc_appearance, nef)
        self.extract_kernel_feature(nc_kernel=nc_kernel, nkf=nkf)
        self.generate_letter(nc_generate=nc_generate, ngf=ngf, f_size=f_size)

    def extract_appearance_feature(self, nc_appearance=3, nef=16):
        self.extract_appearance_layer1 = nn.Sequential(
            nn.Conv2d(nc_appearance, nef, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.extract_appearance_layer2 = nn.Sequential(
            nn.Conv2d(nef, nef * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.extract_appearance_layer3 = nn.Sequential(
            nn.Conv2d(nef * 2, nef * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.extract_appearance_layer4 = nn.Sequential(
            nn.Conv2d(nef * 4, nef * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def extract_kernel_feature(self, nc_kernel=1, nkf=16):
        self.extract_kernel_layer1 = nn.Sequential(
            nn.Conv2d(nc_kernel, nkf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nkf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.extract_kernel_layer2 = nn.Sequential(
            nn.Conv2d(nkf, nkf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nkf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.extract_kernel_layer3 = nn.Sequential(
            nn.Conv2d(nkf * 2, nkf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nkf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.extract_kernel_layer4 = nn.Sequential(
            nn.Conv2d(nkf * 4, nkf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nkf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def generate_letter(self, nc_generate=3, ngf=16, f_size=256):
        self.generate_layer1 = nn.Sequential(nn.ConvTranspose2d(f_size, ngf * 4, kernel_size=4),
                                    nn.BatchNorm2d(ngf * 4),
                                    nn.ReLU())
        # 4 x 4
        self.generate_layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 2),
                                    nn.ReLU())
        # 8 x 8
        self.generate_layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf),
                                    nn.ReLU())
        # 16 x 16
        self.generate_layer4 = nn.Sequential(nn.ConvTranspose2d(ngf, nc_generate, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh())

    def extract_appearance_forward(self, appearance_letter_list):
        out = self.extract_appearance_layer1(appearance_letter_list)
        out = self.extract_appearance_layer2(out)
        out = self.extract_appearance_layer3(out)
        out = self.extract_appearance_layer4(out)
        return out.view(appearance_letter_list.size(0), -1)

    def extract_kernel_forward(self, kernel_letter_list):
        out = self.extract_kernel_layer1(kernel_letter_list)
        out = self.extract_kernel_layer2(out)
        out = self.extract_kernel_layer3(out)
        out = self.extract_kernel_layer4(out)
        return out.view(kernel_letter_list.size(0), -1)

    def generate_letter_forward(self, feature_map_list):
        out = self.generate_layer1(feature_map_list)
        out = self.generate_layer2(out)
        out = self.generate_layer3(out)
        out = self.generate_layer4(out)
        return out

    '''def forward(self, x):
        # x (batch, appearance_letter_count, appearance_letter_list, kernel_letter_list)
        appearance_feature_map = self.extract_appearance_forward(x.size(2)[0])
        for i in range(1, x.size(1)):
            appearance_feature_map += self.extract_appearance_forward(x.size(2)[i])
        appearance_feature_map = (appearance_feature_map * 255.0 / x.size(1)).ToTensor()
        kernel_feature_map = self.extract_kernel_forward(x.size(3))
        generate_feature_map = ((appearance_feature_map + kernel_feature_map) * 255.0 / 2.0).ToTensor()
        return self.generate_letter_forward(generate_feature_map)
    '''

    def forward(self, appearance_letters, kernel_letters):
        count = appearance_letters.size(0)
        # print(appearance_letters[0][0][0][0])
        appearance_feature_map = self.extract_appearance_forward(appearance_letters[0])
        # for i in range(1, count):
            # appearance_feature_map += self.extract_appearance_forward((appearance_letters[i]))
        appearance_feature_map = appearance_feature_map  # / count
        kernel_feature_map = self.extract_kernel_forward(kernel_letters)
        generate_feature_map = torch.cat((appearance_feature_map, kernel_feature_map), dim=1)
        return self.generate_letter_forward(generate_feature_map.view(generate_feature_map.size(0), generate_feature_map.size(1), 1, 1))



dymanic_generate_letter = Dymanic_Generate_Letter()
print(dymanic_generate_letter)
data = My_Data()



optimizer = torch.optim.Adam(dymanic_generate_letter.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

step = 0
for epoch in range(EPOCH):
    data.reset_index()
    is_end, appearance_letters, kernel_letters, target_letters = data.get_batch_data()
    i = 0
    print("--------%d--------------" %(epoch))
    while(is_end):
        i += 1
        appearance_letters = Variable(appearance_letters)
        kernel_letters = Variable(kernel_letters)
        y = Variable(target_letters)
        generation_letters = dymanic_generate_letter(appearance_letters, kernel_letters)
        loss = loss_func(generation_letters, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        is_end, appearance_letters, kernel_letters, target_letters = data.get_batch_data()
        if i % 20 == 0:
            print("%d loss is %f" %(i, loss.data[0]))
            vutils.save_image(generation_letters.data,
                              'result//generation_letters%03d_%03d.jpg' % (epoch, i),
                              normalize=True)
            vutils.save_image(y.data,
                              'result//target_letter%03d_%03d.jpg' % (epoch, i),
                              normalize=True)

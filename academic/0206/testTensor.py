import torch
import numpy as np

kernel = [[1.0, 2.0, 3.0, 4.0],
          [2.0, 3.0, 4.0, 5.0],
          [3.0, 4.0, 5.0, 6.0],
          [4.0, 5.0, 6.0, 7.0]]
kernel = np.array(kernel)

im1 = [kernel*0.3, kernel*0.4, kernel*3.2]
im2 = [kernel*0.6, kernel*2, kernel*0.3]
im3 = [kernel*0.8, kernel*4, kernel*0.1]
im4 = [kernel*0.46, kernel*22, kernel*0.2]
im5 = [kernel*0.76, kernel*52, kernel*0.544]

im1 = torch.FloatTensor(im1)
im2 = torch.FloatTensor(im2)
im3 = torch.FloatTensor(im3)
im4 = torch.FloatTensor(im4)
im5 = torch.FloatTensor(im5)


im = torch.cat((im1, im2))
im = torch.cat((im, im3))
im = torch.cat((im, im4))
im = torch.cat((im, im5))

im = im.view(-1, 3, 4, 4)
im = im.transpose(0, 1)
print(im1[0])
print(im2[0])
print(im3[0])
print(im4[0])
print(im5[0])

print(im[0])
print(im[0].size())
import torch
import torch.nn as nn
from PIL import Image
from math import sqrt
from numpy import clip
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from modules.deform_conv import DeformConvPack, DeformConv
import functools
import torch.nn.functional as F
import numpy as np

img_num = 7
kernel_size = 1
stride = 1
pad = 0
img = Image.open('im1.png')
img = np.array(img, dtype=np.float32) / 255.0 # (64, 112, 3)
img = img.transpose(2, 0, 1) # (64, 112, 3)
sequence = img[np.newaxis,:,:,:]
sequence = torch.from_numpy(np.ascontiguousarray(sequence)).cuda()
dcn = DeformConv(3, 3, kernel_size=kernel_size, stride=stride, padding=pad).cuda()
conv = nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=pad).cuda()
# dcn = DeformConvPack(3, 3, kernel_size=kernel_size, stride=stride, padding=pad).cuda()
b,c,w,h = sequence.shape
offset = torch.zeros(b, kernel_size * kernel_size * 2, w, h).cuda()
# offset[:,2,:,:,:] = 10
y = dcn(sequence, offset)
y_conv = conv(sequence)

# y = dcn(sequence)
1
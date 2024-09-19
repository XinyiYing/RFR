from math import sqrt
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *
from model.RFR_framework import RFR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        self.model_name = model_name
        if 'RFR' in self.model_name:
            head_name = model_name[:-4]
        self.model = RFR(head_name=head_name)
        self.cal_loss = SoftIoULoss()

    def forward_train(self, img, gt_mask):
        pred = self.model.forward_train(img)
        loss = self.cal_loss(pred, gt_mask)
        return loss
        
    def forward_test(self, img, feat_prop):
        pred, feat_prop = self.model.forward_test(img, feat_prop)
        return pred, feat_prop

    
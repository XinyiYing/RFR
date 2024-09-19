import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
import torch.nn as nn
from skimage import measure
import torch.nn.functional as F
import os
from torch.nn import init
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('SplAtConv2d') == -1:
        init.xavier_normal(m.weight.data)
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0
                
def random_crop(img, mask, patch_size): 
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        h, w = img.shape
    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size


    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch

def random_crop_seq(img_seq, mask_seq, patch_size, pos_prob=False): 
    _, h, w = img_seq.shape
    if min(h, w) < patch_size:
        for i in range(len(img_seq)):
            img_seq[i,:,:] = np.pad(img_seq[i,:,:], ((0, 0),(0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
            mask_seq[i,:,:] = np.pad(mask_seq[i,:,:], ((0, 0),(0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
            _, h, w = img_seq.shape
    
    cur_prob = random.random()
    
    if pos_prob == None or cur_prob > pos_prob or mask_seq.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask_seq > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0])-1)
        h_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], h-patch_size))
        w_start = random.randint(max(0, loc[2][idx] - patch_size), min(loc[2][idx], w-patch_size))
        
    h_end = h_start + patch_size
    w_end = w_start + patch_size
    img_patch_seq = img_seq[:, h_start:h_end, w_start:w_end]
    mask_patch_seq = mask_seq[:, h_start:h_end, w_start:w_end]

    return img_patch_seq, mask_patch_seq

def Normalized(img, img_norm_cfg):
    return (img-img_norm_cfg['mean'])/img_norm_cfg['std']
    
def Denormalization(img, img_norm_cfg):
    return img*img_norm_cfg['std']+img_norm_cfg['mean']

def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'IRSatVideo-LEO':   
        img_norm_cfg = {'mean': 72.1040267944336, 'std': 12.302865028381348}
    else:
        with open(dataset_dir+'/'+dataset_name+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir+'/'+dataset_name+'/img_idx/test_' + dataset_name + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//','/')+'.jpg').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//','/')+'.png').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//','/')+'.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
        print(dataset_name)
        print(img_norm_cfg)
    return img_norm_cfg

def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'Adagrad':
        optimizer  = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer  = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'])
    
    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['step'], gamma=scheduler_settings['gamma'])
    elif scheduler_name   == 'CosineAnnealingLR':
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'], eta_min=scheduler_settings['min_lr'])
    
    return optimizer, scheduler
        
def extend_img_list(img_list, seq_len, mode='replicate'):
    out_img_list = img_list
    
    if mode=='replicate':
        for i in range(len(img_list)):
            if img_list[i] == 'Not same sequence':
                out_img_list[i] = img_list[i-1]
        
    if mode=='extend':    
        if len(img_list)>1:
            extend_times = (seq_len - len(img_list))//(len(img_list)-1) + 1
            for _ in range(extend_times):
                out_img_list = out_img_list + img_list[::-1][1:]
                img_list = img_list[::-1]
        else:
            for _ in range(seq_len-1):   
                out_img_list = out_img_list + img_list
    
    return out_img_list[:seq_len]

def PadImg(img, times):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img

def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

if __name__ == '__main__':
    # img_list = ['1', '2', '3']
    # seq_len = 10
    # out_img_list = extend_img_list(img_list, seq_len, 'extend')
    # print(out_img_list)
    
    # img_list = ['Not same sequence', 'Not same sequence', 'Not same sequence', '1', '2', '3', 'Not same sequence']
    # seq_len = 10
    # out_img_list = extend_img_list(img_list, seq_len, 'replicate')
    # print(out_img_list)
    
    print(get_img_norm_cfg(dataset_name='SIRST3', dataset_dir='/home/y/yxy/DNAnet/dataset'))
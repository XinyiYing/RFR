import argparse
from net import Net
import os
import time
from thop import profile
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD Parameter and FLOPs")
parser.add_argument("--model_names", default=['ACM_RFR', 'ALCNet_RFR', 'DNANet_RFR', 'ISTUDNet_RFR', 'ResUNet_RFR'], type=list)

global opt
opt = parser.parse_args()

if __name__ == '__main__':
    opt.f = open('./params_' + (time.ctime()).replace(' ', '_') + '.txt', 'w')
    for model_name in opt.model_names:
        n, c, h, w = 1, 1, 256, 256
        img = torch.rand(n, c, h, w).cuda()
        feat_prop = torch.rand(n, c, h, w).cuda()
        net = Net(model_name).cuda()    
        flops, params = profile(net, inputs=(img, feat_prop, ))
        print(model_name)
        print('Params: %2fM' % (params/1e6))
        print('FLOPs: %2fGFLOPs' % (flops/1e9))
        opt.f.write(model_name + '\n')
        opt.f.write('Params: %2fM\n' % (params/1e6))
        opt.f.write('FLOPs: %2fGFLOPs\n' % (flops/1e9))
        opt.f.write('\n')
    opt.f.close()
        
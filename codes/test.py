import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['ISTUDNet_RFR', 'ResUNet_RFR'], type=list, 
                    help="model_name: 'ACM_RFR', 'ALCNet_RFR', 'DNANet_RFR', 'ISTUDNet_RFR', 'ResUNet_RFR'")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--pth_dirs", default=None, type=list, help="checkpoint dir, default=None")#
parser.add_argument("--dataset_dir", default='/home/y/yxy/evaluation_new/data', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['IRSatVideo-LEO'], type=list,
                    help="dataset_name: 'IRSatVideo-LEO'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=False, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()

def test(): 
    with open(opt.dataset_dir + '/' + dataset_name +'/video_idx/test_' + opt.test_dataset_name + '.txt', 'r') as f:
        test_list = f.read().splitlines()
    
    net = Net(model_name=opt.model_name).cuda()
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    net.eval()
    
    eval_mIoU_all = mIoU() 
    eval_PD_FA_all = PD_FA()
    
    for video_dir in  test_list:  
        test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, video_dir, opt.img_norm_cfg)
        test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
        
        eval_mIoU = mIoU() 
        eval_PD_FA = PD_FA()
        
        with torch.no_grad():
            for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
                img = Variable(img).cuda()
                if idx_iter == 0:
                    feat_prop = None
                pred, feat_prop  = net.forward_test(img, feat_prop)
                pred = pred[:,:,:size[0],:size[1]]
                gt_mask = gt_mask[:,:,:size[0],:size[1]]
                
                eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
                eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)  
                eval_mIoU_all.update((pred>opt.threshold).cpu(), gt_mask)
                eval_PD_FA_all.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)
            
                ### save img
                if opt.save_img == True:
                    img_save = transforms.ToPILImage()((pred[0,0,:,:]).cpu())
                    save_pth = opt.save_img_dir + opt.dataset_name + '/' + opt.model_name + '/' + img_dir[i][0]
                    if not os.path.exists(os.path.dirname(save_pth)):
                        os.makedirs(os.path.dirname(save_pth))
                    img_save.save(save_pth)  

            results1 = eval_mIoU.get()
            results2 = eval_PD_FA.get()
            print(video_dir) 
            print("pixAcc, mIoU:\t" + str(results1))
            print("PD, FA:\t" + str(results2))
            opt.f.write(video_dir + '\n')
            opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
            opt.f.write("PD, FA:\t" + str(results2) + '\n')
            
    results1 = eval_mIoU_all.get()
    results2 = eval_PD_FA_all.get()
    print('Total')
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write('Total \n')
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(opt.model_name)
                opt.f.write(opt.model_name + '\n')
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for dataset_name in opt.dataset_names:
            opt.dataset_name = dataset_name
            opt.test_dataset_name = dataset_name
            for pth_dir in opt.pth_dirs:
                for model_name in opt.model_names:
                    if model_name in pth_dir:
                        opt.model_name = model_name
                opt.train_dataset_name = pth_dir.split('/')[0]
                print(opt.model_name)
                opt.f.write(pth_dir + '\n')
                print(opt.test_dataset_name)
                opt.f.write(opt.test_dataset_name + '\n')
                opt.pth_dir = opt.save_log + pth_dir
                test()
                print('\n')
                opt.f.write('\n')
        opt.f.close()
        

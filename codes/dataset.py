from utils import *
import matplotlib.pyplot as plt
from SIFT_module import *
import scipy.io as scio
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF', '.mat')

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, seq_len, patch_size, sample_rate=1, img_norm_cfg=None, img_register=None, pos_prob=0.5):
        super(TrainSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.seq_len = seq_len
        self.img_register = img_register
        self.sample_rate = sample_rate
        self.dataset_name = dataset_name
        with open(self.dataset_dir+'/video_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        ## calculate the number of total frame
        ## get video dir by ratio
        seq_list = []
        for seq_dir in self.train_list:
            with open(dataset_dir+'/'+dataset_name+'/img_idx/' + seq_dir + '.txt', 'r') as f:
                img_list = f.read().splitlines()
            seq_list = seq_list + [seq_dir for _ in range(len(img_list))]
        self.total_len = len(seq_list)
        self.seq_list = seq_list
        
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
        
    def __getitem__(self, idx):
        seq_name = random.sample(self.seq_list, 1)[0]
        with open(self.dataset_dir+'/img_idx/' + seq_name + '.txt', 'r') as f:
            img_list = f.read().splitlines()
        img_ext = os.path.splitext(os.listdir(self.dataset_dir + '/images/' + seq_name)[0])[-1]
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        
        img_seq = []
        mask_seq = []
        idx = random.randint(0, len(img_list)-1)
        for i in range(0, self.seq_len):
            cur_idx = idx + i
            if cur_idx > len(img_list) - 1:
                cur_idx = len(img_list) - 1
            img = Image.open(self.dataset_dir + '/images/' + seq_name + '/' + img_list[cur_idx] + img_ext).convert('I')
            img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
            mask = Image.open(self.dataset_dir + '/masks/' + seq_name + '/' + img_list[cur_idx] + img_ext)    
            mask = np.array(mask, dtype=np.float32)  / 255.0
            mask = np.array(mask > 0, dtype=np.float32)
            
            if len(mask.shape)>2:
                mask = mask[:,:,0]
            
            img_seq.append(img)
            mask_seq.append(mask)
       
        img_seq = np.stack(img_seq, 0)
        mask_seq = np.stack(mask_seq, 0)
            
        img_patch, mask_patch = random_crop_seq(img_seq, mask_seq, self.patch_size, pos_prob=self.pos_prob)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[:, np.newaxis,: , :], mask_patch[:, np.newaxis,: , :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch
    def __len__(self):
        return self.total_len // self.sample_rate

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, video_dir, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        self.video_dir = video_dir
        with open(self.dataset_dir+'/img_idx/' + video_dir + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()  
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        
        img_list = os.listdir(self.dataset_dir + '/images/' + self.video_dir)
        img_ext = os.path.splitext(img_list[0])[-1]
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        
        img_dir = self.test_list[idx]
        
        img = Image.open(self.dataset_dir + '/images/'  + self.video_dir + '/' + img_dir + img_ext).convert('I')
        mask = Image.open(self.dataset_dir + '/masks/'  + self.video_dir + '/' + img_dir + img_ext)
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
            
        if len(mask.shape)>2:
            mask = mask[:,:,0]

        h, w = img.shape 
        
        img = PadImg(img, 32)
        mask = PadImg(mask, 32)

        img, mask = img[np.newaxis, : , :], mask[np.newaxis,: , :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], img_dir
    def __len__(self):
        return len(self.test_list)
  
class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random()<0.5:
            input = input[:, :, ::-1]
            target = target[:, :, ::-1]
        if random.random()<0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random()<0.5:
            input = input.transpose(0, 2, 1)
            target = target.transpose(0, 2, 1)
        return input, target


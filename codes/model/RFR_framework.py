import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.dcn.modules.deform_conv import DeformConv
import os
from model import *
from model.utils import *

def model_lib(model_chose):
    model_factory = {
                     'ACM': ACM,
                     'ALCNet': ALCNet,
                     'ISTUDNet': ISTUDNet,
                     'ResUNet': ResUNet,
                     'DNANet': DNANet,
                     }
    return model_factory[model_chose]

class RFR(nn.Module):
    def __init__(self,
                 mid_channels=16,
                 head_name='ResUnet'):

        super(RFR, self).__init__()
        self.mid_channels = mid_channels

        # feature extraction module
        self.feat_extract = nn.Conv2d(1, mid_channels, 3, 1, 1)

        # propagation branches
        self.deform_align = PDA(num_feat=mid_channels, deformable_groups=4)
        self.spatio_temporal_fusion = TSFM(num_feat=mid_channels)
        self.fusion_align = nn.Conv2d(
        2 * mid_channels, mid_channels, 3, 1, 1)

        self.fusion = nn.Conv2d(
            2 * mid_channels, mid_channels, 3, 1, 1)
        
        ### detection_head
        net = model_lib(head_name)
        self.detection_head = net(input_channels=mid_channels)

    def forward_train(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output sequence with shape (n, t, 1, h, w).
        """

        n, t, c, h, w = lqs.size()

        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)
        feats = [feats_[:, i, :, :, :] for i in range(0, t)]
        
        # feature propgation
        t = len(feats)
        frame_idx = range(0, t )
        feat_prop = feats[0]
        feat_props = []
        for i, idx in enumerate(frame_idx):
            feat_current = feats[idx]
            if i > 0:
                feat_prop = self.deform_align(feat_prop, feat_current)
            feat_prop = self.spatio_temporal_fusion(feat_prop, feat_current)
            feat_props.append(feat_prop) 
        
        # Detection Head    
        outputs = []
        for i in range(0, t):
            fea = torch.cat([feats[i], feat_props[i]], dim=1)
            fea = self.fusion(fea)
            out = self.detection_head(fea).sigmoid()
            outputs.append(out) 
        return torch.stack(outputs, dim=1)

    def forward_test(self, lq, feat_prop):
        feat_current = self.feat_extract(lq)
        
        if feat_prop == None:
            feat_prop = feat_current
        else:
            feat_prop = self.deform_align(feat_prop, feat_current)
        feat_prop = self.spatio_temporal_fusion(feat_prop, feat_current)
        
        fea = torch.cat([feat_current, feat_prop], dim=1)
        fea = self.fusion(fea)
        out = self.detection_head(fea).sigmoid()
        
        return out, feat_prop
    
class TSFM(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64):
        super(TSFM, self).__init__()
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(2 * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.sa = SpatialAttention()
        self.fa = FreqAttention(num_feat)
        
        self.lrelu = nn.ReLU(inplace=True)

    def forward(self, aligned_feat, curr_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, c, h, w).
            curr_feat (Tensor): Current features with shape (b, c, h, w).
        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        # temporal attention
        embedding_ref = self.temporal_attn1(curr_feat.clone())
        emb_neighbor = self.temporal_attn2(aligned_feat.clone())

        corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
        corr_prob = torch.sigmoid(corr.unsqueeze(1))  # (b, 1, h, w)
        
        # fusion
        feat_t = self.feat_fusion(torch.cat((aligned_feat*corr_prob, curr_feat), dim=1))
        
        # spatial attention
        s_att = self.sa(feat_t)
        feat_s = s_att * feat_t

        # Freq attention
        feat_f = self.fa(feat_s)
        feat = self.lrelu(feat_f)
        
        return feat

class PDA(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=4):
        super(PDA, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = \
            DeformConv(num_feat,
                num_feat,
                3,
                padding=1,
                stride=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.ReLU(inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l, video_dir=None):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        nbr_feat_l1 = nbr_feat_l
        nbr_feat_l2 = self.lrelu(self.conv_l2_1(nbr_feat_l1))
        nbr_feat_l2 = self.lrelu(self.conv_l2_2(nbr_feat_l2))
        # L3
        nbr_feat_l3 = self.lrelu(self.conv_l3_1(nbr_feat_l2))
        nbr_feat_l3 = self.lrelu(self.conv_l3_2(nbr_feat_l3))
        
        nbr_feat_l = [nbr_feat_l1, nbr_feat_l2, nbr_feat_l3]
        
        ref_feat_l1 = ref_feat_l
        ref_feat_l2 = self.lrelu(self.conv_l2_1(ref_feat_l1))
        ref_feat_l2 = self.lrelu(self.conv_l2_2(ref_feat_l2))
        # L3
        ref_feat_l3 = self.lrelu(self.conv_l3_1(ref_feat_l2))
        ref_feat_l3 = self.lrelu(self.conv_l3_2(ref_feat_l3))
        
        ref_feat_l = [ref_feat_l1, ref_feat_l2, ref_feat_l3]
        
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)
                
            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat) 
        return feat


if __name__ == '__main__':
    from thop import profile
    n, t, c, h, w = 1, 20, 1, 256, 256
    in_fea = torch.rand(n, t, c, h, w).cuda()
    net = RFR().cuda()
    flops, params = profile(net, inputs=(in_fea, ))
    print('Params: %2fM' % (params/1e6))
    print('FLOPs: %2fGFLOPs' % (flops/1e9))
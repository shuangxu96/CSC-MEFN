# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:55:07 2020

@author: Shuang Xu
"""
import torch
import torch.nn as nn
import blocks as B
from torch.nn.init import kaiming_normal_, constant_

EPS = 1e-8     
def rgb2ycbcr(img):
    return [0. / 256. + img[:, 0:1, :, :] * 0.299000 + img[:, 1:2, :, :] * 0.587000 + img[:, 2:3, :, :] * 0.114000,
       128. / 256. - img[:, 0:1, :, :] * 0.168736 - img[:, 1:2, :, :] * 0.331264 + img[:, 2:3, :, :] * 0.500000,
       128. / 256. + img[:, 0:1, :, :] * 0.500000 - img[:, 1:2, :, :] * 0.418688 - img[:, 2:3, :, :] * 0.081312]

def ycbcr2rgb(img):
    return [img[:, 0:1, :, :] + (img[:, 2:3, :, :] - 128 / 256.) * 1.402,
        img[:, 0:1, :, :] - (img[:, 1:2, :, :] - 128 / 256.) * 0.344136 - (img[:, 2:3, :, :] - 128 / 256.) * 0.714136,
        img[:, 0:1, :, :] + (img[:, 1:2, :, :] - 128 / 256.) * 1.772]

class CSC_Fusion_MEF(nn.Module):
    '''The proposed network'''
    def __init__(self, num_blocks=4,
                 img_channels=1, 
                 num_feat=24,
                 kernel_size=3,
                 num_convs=1,
                 act='sst',
                 act_init=None,
                 norm=True):
        super(CSC_Fusion_MEF, self).__init__()
        
        main = [B.Conv2d(img_channels, num_feat, 3),
                nn.BatchNorm2d(num_feat),
                B.get_activation(act,None,act_init)]
        for i in range(num_blocks-3):
            main+= [B.DictConv2dBlock(
                img_channels, num_feat, num_feat, kernel_size, dilation=1,
                num_convs=num_convs, act=act, norm=norm)]
        main+= [B.DictConv2dBlock(
                img_channels, num_feat, num_feat, kernel_size, dilation=1,
                num_convs=num_convs, act=act, norm=norm),
                B.Conv2d(num_feat, 1, 1)]
        self.main = nn.Sequential(*main)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
      
    def get_code(self, data):
        code = data
        for _, layer in enumerate(self.main):
            if type(layer).__name__.startswith('DictConv2d'):
                code = layer(data, code)
            else:
                code = layer(code)
        return code
        
    def train_forward(self, img):
        # img: torch.Tensor [N,C,K,H,W]
        N,C,K,H,W = img.shape
        Wf = [] #[N,K,H,W]
        Ys = [] #[N,K,H,W]
        Cbf = [] #[N,K,H,W]
        Crf = [] #[N,K,H,W]
        for k in range(K):
            Y, Cb, Cr = rgb2ycbcr(img[:,:,k,:,:]) #[N,1,H,W]
            W = self.get_code(Y) #[N,1,H,W]
            Wf.append(W)
            Ys.append(Y)
            Cbf.append(Cb)
            Crf.append(Cr)
        
        Ys = torch.cat(Ys,  dim=1)
        Wf = torch.cat(Wf,  dim=1) #[N,K,H,W]
        Wf = nn.functional.softmax(Wf, dim=1)
        Yf = torch.sum(Wf * Ys, dim=1, keepdim=True) #[N,1,H,W]
        
        Cbf= torch.cat(Cbf, dim=1)
        Crf= torch.cat(Crf, dim=1)
        Wb = (torch.abs(Cbf - 0.5) + EPS) / torch.sum(torch.abs(Cbf - 0.5) + EPS, dim=1, keepdim=True)
        Wr = (torch.abs(Crf - 0.5) + EPS) / torch.sum(torch.abs(Crf - 0.5) + EPS, dim=1, keepdim=True)
        Cbf = torch.sum(Wb * Cbf, dim=1, keepdim=True)
        Crf = torch.sum(Wr * Crf, dim=1, keepdim=True)
        
        fusion = torch.cat((Yf,Cbf,Crf), dim=1)
        fusion = torch.cat(ycbcr2rgb(fusion), dim=1)
        return fusion
    
    def forward(self, img, phase='train'):
        if phase=='train':
            fusion = self.train_forward(img)
            return fusion
        if phase=='test':
            fusion = self.train_forward(img)
            fusion = torch.clamp(fusion, min=0., max=1.)
            return fusion
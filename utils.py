# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:22:36 2020

@author: BSawa
"""

import torch.utils.data as Data
import torchvision.transforms as transforms

from glob import glob
from PIL import Image

import torch
import kornia
import numpy as np
import os
from mefssim import MEF_MSSSIM
from skimage.io import imsave

class FolderDataset(Data.Dataset):
    # Load images in folders. The images are organized as follows:
    # root
    #  - folder1
    #    -- img1.jpg
    #    -- img2.jpg
    #    -- img3.jpg
    #  - folder2
    #    -- img1.jpg
    #    -- img2.jpg
    #  ........
    def __init__(self, root, k=None):
        # root: The path of folders
        # k: the maximum number of images in a batch. k=None by default. In 
        #    this case, all the images in a folder will be loaded. If set k to
        #    a integer, e.g. 4, it will randomly load 4 images in each folder.
        #    Generally, if the numbers of images in different folders are not 
        #    equal, you are suggested to set k to a integer to make sure load 
        #    a batch of images with shape [N,C,K,H,W]
        self.root = root
        self.folders = glob(root+'/*')
        self._tensor = transforms.ToTensor()
        self.k = k

    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, index):
        files = glob(os.path.join(self.folders[index],'*'))
        if self.k!=None:
            files_index = torch.randperm(len(files))[:self.k]
            imgs = []
            for i in files_index:
                I = Image.open(files[i])
                imgs.append(self._tensor(I))
        else:
            imgs = []
            for single_file in files:
                I = Image.open(os.path.join(single_file))
                imgs.append(self._tensor(I))

        return torch.stack(imgs, dim=1) #[C,K,H,W]
    
def loss_fn(imgf, img):
    # compute the MEF-SSIM value between imgf and img
    # imgf - A batch of fused images with shape [N,C,H,W]
    # img - Corresponding source images with shape [N,C,K,H,W]
    mefssim_fn = MEF_MSSSIM(is_lum=True)
    loss = 0.
    for n in range(img.shape[0]):
        loss = loss+mefssim_fn(imgf[n:n+1,:,:,:], img[n,:,:,:,:].permute(1,0,2,3))
    return -loss/img.shape[0]  

def halo_fn(img):
    # compute the halo loss (i.e. the L1-norm of the gradients of the Y channel)
    # img - A batch of images with shape [N,C,H,W]
    weight = torch.Tensor([0.299,0.587,0.114]).to(img.dtype).to(img.device).reshape([1,3,1,1])
    Y = torch.sum(weight*img, dim=1, keepdim=True)
    return kornia.filters.SpatialGradient()(Y).abs().mean()
  
def post_process(img,a=0.005,b=0.995):
    # post-process function. The values between 100a% and 100b% intensity level
    # are rescaled to [0,1], and values out of this range is clipped. 
    img_post = img.clone()
    img_sorted=torch.sort(img_post.reshape(img_post.numel()))[0]
    low  = img_sorted[int(img_sorted.numel()*a)]
    high = img_sorted[int(img_sorted.numel()*b)]
    img_post[img_post<low]=low
    img_post[img_post>high]=high
    img_post = 255.*(img_post-img_post.min())/(img_post.max()-img_post.min())
    return img_post.cpu().numpy().astype(np.uint8)

def mkdir(path):
    # create a path if this path is not exist
    if os.path.exists(path) is False:
        os.makedirs(path)

def test(net, loader, save_path, identifier='HDRPS'):
    # test phase.
    # net - pretrained CSC-MEFN
    # loader - the data loader of the test dataset
    # save_path - save path
    # identifier - the name of the test dataset
    
    # 1. create the output path
    mkdir(os.path.join(save_path,'test_%s%d')%(identifier,995))
    #mkdir(os.path.join(save_path,'test_%s')%(identifier))
    # 2. output test results
    with torch.no_grad():
        net.eval()
        for i, img in enumerate(loader):
            img = img.cuda()
            imgf = net(img, 'test')
            imgf = 255.*imgf.squeeze(0).permute(1,2,0).detach()
            imgf995 = post_process(imgf, a=0.005, b=0.995)
            #imgf = imgf.byte().cpu().numpy()
            
            file_name = loader.dataset.folders[i].split('\\')[-1]+'.png'
            imsave(os.path.join(save_path,'test_%s%d'%(identifier,995),file_name), imgf995)
            #imsave(os.path.join(save_path,'test_%s'%(identifier),file_name),       imgf)
            print('%s: %dth image saved...'%(identifier,i+1))
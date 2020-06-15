# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:08:52 2020

@author: BSawa
"""

import torch
import datetime
import os
from glob import glob
import numpy as np
from skimage.io import imsave, imread
from networks import CSC_Fusion_MEF
from utils import FolderDataset, mkdir, post_process, loss_fn, halo_fn
from torch.utils.data import DataLoader 

if __name__ == "__main__":
    act = 'sst'
    num_blocks = 4

    # testset loaders
    testset4          = FolderDataset('MEF_data/test',None)
    testloader4       = DataLoader(testset4,       batch_size=1)
    
    loader = {'HDRPS': testloader4}
    
    save_path = r'05-03-03-17_bs8_epoch50_lr0.000500_lw0.000000'
    net = CSC_Fusion_MEF(act=act,num_blocks=num_blocks).cuda()
    net.load_state_dict(torch.load(os.path.join(save_path,'best_net.pth')))
    test(net, loader['HDRPS'],  save_path, 'HDRPS')


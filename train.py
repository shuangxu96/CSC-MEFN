# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:18:05 2020

@author: Anonymous
"""
import torch
import datetime
import os

from networks import CSC_Fusion_MEF
from utils import FolderDataset, loss_fn, halo_fn, test
from torch.utils.data import DataLoader 
from tensorboardX import SummaryWriter

# Hyper-parameters
batch_size = 8
lr = 5e-4
num_epoch = 50
num_blocks = 8
loss_weight = 0. # the initial loss weight
loss_weight_max = 10.

# Network
net = CSC_Fusion_MEF(act='sst',num_blocks=num_blocks).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Loaders
trainset         = FolderDataset(r'MEF_data\train', 4)
trainloader      = DataLoader(trainset,      batch_size=batch_size, shuffle=True) # load a batch with shape [N,C,K,H,W]
validationset    = FolderDataset(r'MEF_data\validation',None)
validationloader = DataLoader(validationset, batch_size=1)
loader = {'train': trainloader,
        'validation': validationloader}

# Loggers
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join('MFF_logs',timestamp+'_bs%d_epoch%d_lr%f_lw%f'%(batch_size,num_epoch,lr,loss_weight_max))
writer = SummaryWriter(save_path)

# Iterations
step = 0
best_mefssim_val = 0.
torch.backends.cudnn.benchmark = True
for epoch in range(num_epoch):
    ''' train '''
    for i, img in enumerate(loader['train']):
        img = img.cuda()
        img = torch.rot90(img, int(torch.randint(4,[1])), [-1,-2])
        
        #1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        imgf = net(img)
        _ssim = loss_fn(imgf, img) 
        _l1penalty = halo_fn(imgf)
        loss = _ssim+loss_weight*_l1penalty
        loss.backward()
        optimizer.step()
        
        loss_weight = min(loss_weight+0.25,10) # update loss weight
        
        #2.  print information
        print("[%d,%d] MEFSSIM: %.4f, L1: %.4f, Loss: %.4f" %
                (epoch+1, i+1, _ssim.item(), _l1penalty.item(), loss.item()))

        #3. log the scalar values
        writer.add_scalar('loss', loss.item(), step)
        step+=1
    
    ''' validation ''' 
    mefssim_val = 0.
    with torch.no_grad():
        net.eval()
        for i, img in enumerate(loader['validation']):
            img = img.cuda()
            imgf = net(img, 'test')
            mefssim_val -= loss_fn(imgf, img)*img.shape[0]
        mefssim_val = float(mefssim_val/validationset.__len__())
    writer.add_scalar('MEFSSIM on validation data', mefssim_val, epoch)

    ''' save model ''' 
    # save best model
    if best_mefssim_val<mefssim_val:
        best_mefssim_val = mefssim_val
        torch.save(net.state_dict(), os.path.join(save_path, 'best_net.pth'))
        print('Best MEFSSIM value is updated. Weight is saved at %s.'%(save_path))
    # save current model
    torch.save({'net':net.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch},
                os.path.join(save_path, 'last_net.pth'))

# test
best_net = CSC_Fusion_MEF(act='sst',num_blocks=num_blocks).cuda()
best_net.load_state_dict(torch.load(os.path.join(save_path,'best_net.pth')))

testset1          = FolderDataset(r'MEF_data\test',None)
testloader1       = DataLoader(testset1,       batch_size=1)
loader = {'HDRPS': testloader1}

test(net, loader['HDRPS'], save_path, 'HDRPS')
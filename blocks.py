# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:20:30 2020

@author: Shuang Xu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SST(nn.Module):
    def __init__(self, num_parameters=1, 
                 init=1e-3):
        super(SST,self).__init__()
        self.theta = nn.Parameter(torch.full(size=(1, num_parameters, 1, 1), 
                                             fill_value=init)) 
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        return x.sign()*self.relu(x.abs()-self.theta)

def get_activation(act='relu', 
                   num_parameters=None, 
                   init=None):
    if act.lower() not in ['relu','prelu','sst','leakyrelu']:
        raise Exception('Only support "relu","prelu" or "sst". But get "%s."'%(act))
    act = act.lower()
    if act=='relu':
        return nn.ReLU(True)
    if act=='leakyrelu':
        init=0.2 if init==None else init
        return nn.LeakyReLU(init, True)
    if act=='prelu':
        num_parameters=1 if num_parameters==None else num_parameters
        init=0.25 if init==None else init
        return nn.PReLU(num_parameters, init)
    if act=='sst':
        num_parameters=1 if num_parameters==None else num_parameters
        init=1e-3 if init==None else init
        return SST(num_parameters, init)

def get_padder(padding_mode='reflection',
               padding=1,
               value=None):
    if padding_mode.lower() not in ['reflection','replication','zero','zeros','constant']:
        raise Exception('Only support "reflection","replication","zero" or "constant". But get "%s."'%(padding_mode))
    padding_mode = padding_mode.lower()
    if padding_mode=='reflection':
        return nn.ReflectionPad2d(padding)
    if padding_mode=='replication':
        return nn.ReplicationPad2d(padding)
    if padding_mode in ['zero','zeros']:
        return nn.ZeroPad2d(padding)
    if padding_mode in 'constant':
        value=0 if value==None else value
        return nn.ConstantPad2d(padding,value)

class Conv2d(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1,
                 dilation=1, 
                 groups=1, 
                 bias=False,
                 padding_mode='reflection',
                 padding='same',
                 value=None):
        super(Conv2d, self).__init__()
        padding = int(int(1+dilation*(kernel_size-1))//2) if padding=='same' else 0
        self.pad = get_padder(padding_mode, padding, value)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                0, dilation, groups, bias)
    def forward(self, code):
        return self.conv2d(self.pad(code))

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 act='relu', # activation function (None will disable it)
                 act_value=None, # the initialization of SST or PReLU (Ignore it if act is "relu")
                 norm=True, # batch_normalization (None will disable it)
                 padding_mode='reflection',
                 padding='same',
                 padding_value=None):
        super(Conv2dBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, 
                           dilation, groups, bias, padding_mode, padding, 
                           padding_value)
        self.norm = nn.BatchNorm2d(out_channels) if norm!=None else None
        num_parameters = out_channels if act!='relu' else None
        self.activation = get_activation(act, num_parameters, act_value) if act!=None else None

    def _forward(self, data):
        code = self.conv(data)
        if self.norm!=None:
            code = self.norm(code)
        if self.activation!=None:
            code = self.activation(code)
        return code
    
    def forward(self, data):
        return self._forward(data)
    
class DictConv2d(nn.Module):
    def __init__(self, img_channels, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, # only support stride=1
                 dilation=1, 
                 groups=1, 
                 bias=False, 
                 num_convs=1,
                 padding_mode='reflection',
                 padding='same',
                 value=None):
        super(DictConv2d, self).__init__()
        self.in_channels = in_channels
        
        # decoder
        self.conv_decoder = nn.Sequential()
        self.conv_decoder.add_module('de_conv0', Conv2d(in_channels, img_channels, kernel_size, 1, dilation, groups, bias, padding_mode, padding, value) )
        for i in range(1,num_convs):
            self.conv_decoder.add_module('de_conv'+str(i), Conv2d(img_channels, img_channels, kernel_size, 1, dilation, groups, bias, padding_mode, padding, value))
        
        # encoder
        self.conv_encoder = nn.Sequential()
        for i in range(num_convs-1):
            self.conv_encoder.add_module('en_conv'+str(i), Conv2d(img_channels, img_channels, kernel_size, 1, dilation, groups, bias, padding_mode, padding, value))
        self.conv_encoder.add_module('en_conv'+str(num_convs-1), Conv2d(img_channels, in_channels, kernel_size, 1, dilation, groups, bias, padding_mode, padding, value) )
        
        # 如果输入输出通道数不一致，使用1x1卷积改变conv_encoder的通道数
        self.shift_flag = out_channels != in_channels
        self.conv_channel_shift = nn.Conv2d(in_channels, out_channels, 1) if self.shift_flag else None
            
    def _forward(self, data, code):
        B,_,H,W = data.shape
        code = torch.zeros(B,self.in_channels,H,W).to(data.device).to(data.dtype) if code is None else code 
        dcode = self.conv_decoder(code)
        
        if data.shape[2]!=dcode.shape[2] or data.shape[3]!=dcode.shape[3]:
            data = F.interpolate(data, size=dcode.shape[2:])
        res = data - dcode
        dres = self.conv_encoder(res)
        
        if code.shape[2]!=dres.shape[2] or code.shape[3]!=dres.shape[3]:
            code = F.interpolate(code, size=dres.shape[2:])
        code = code+dres
        
        if self.shift_flag:
            code = self.conv_channel_shift(code)
        return code
    
    def forward(self, data, code):
        return self._forward(data, code)

class DictConv2dBlock(nn.Module):
    def __init__(self, img_channels, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 dilation=1, 
                 groups=1, 
                 bias=False, 
                 num_convs=1, # the number of conv units in decoder and encoder
                 act='relu', # activation function (None will disable it)
                 act_value=None, # the initialization of SST or PReLU (Ignore it if act is "relu")
                 norm=True, # batch_normalization (None will disable it)
                 padding_mode='reflection',
                 padding_value=None, # padding constant (Ignore it if padding_mode is not "constant")
                 ):
        super(DictConv2dBlock, self).__init__()
        self.conv = DictConv2d(img_channels,in_channels,out_channels, kernel_size, stride, 
                   dilation, groups, bias, num_convs)
        self.norm = nn.BatchNorm2d(out_channels) if norm!=None else None
        num_parameters = out_channels if act!='relu' else None
        self.activation = get_activation(act, num_parameters, act_value) if act!=None else None
    
    def _forward(self, data, code):
        code = self.conv(data,code)
        if self.norm!=None:
            code = self.norm(code)
        if self.activation!=None:
            code = self.activation(code)
        return code
    
    def forward(self, data, code):
        return self._forward(data, code)
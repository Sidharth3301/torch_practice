


import torch

import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,dim1,dim2,kernel_size=3,stride=1,pad=0):
        super(ConvBlock, self).__init__()

        if pad >=0:
            self.block = nn.Sequential(
            nn.Conv2d(dim1,dim2,kernel_size,stride,padding=pad), 
            nn.ReLU(), 
            nn.BatchNorm2d(dim2)
        )  
        else:
            self.block = nn.Sequential(
            nn.Conv2d(dim1,dim2,kernel_size,stride,padding='same'), 
            nn.ReLU(), 
            nn.BatchNorm2d(dim2)
        )  
    
       
    def forward(self, x):
        return self.block(x)


class Resblock(nn.Module):
    def __init__(self,dim1,dim2,kernel_size=3,stride=1,is_proj=False):
        super(Resblock, self).__init__()
        
        self.is_proj=is_proj
        if is_proj:
            
            self.block = nn.Sequential(ConvBlock(dim1,dim2,kernel_size,stride=2,pad=1),ConvBlock(dim2,dim2,kernel_size,stride,pad=-1))
            self.proj=nn.Sequential(ConvBlock(dim1,dim2,kernel_size=1, stride=2))
        
        else :
            
            self.block = nn.Sequential(ConvBlock(dim1,dim2,kernel_size,stride,pad=-1),ConvBlock(dim2,dim2,kernel_size,stride,pad=-1))
        
    
    def forward(self, x):
        if self.is_proj:
            print(f'Shape at proj : {self.block(x).shape}')
            out=self.block(x)+self.proj(x)
            
        else:
            print(f'Shape at not proj : {self.block(x).shape}')
            out=self.block(x)+x
            
        return F.relu(out)

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel,self).__init__()
        self.is_proj=False
        self.conv1_block =nn.Sequential(ConvBlock(3,64,7,2,pad=3),nn.MaxPool2d(3,stride=2,padding=1))
        self.layer_list=[3,4,6,3]

    def forward(self,x):
        x=self.conv1_block(x)
        dim=64
        planes=[64,128,256,512]
        for i,j in enumerate(self.layer_list):
            if(dim!=planes[i]):
                self.is_proj=True  
            else:
                self.is_proj=False

            for _ in range(j):
                if dim!=planes[i]:
                    #dim= planes[i-1]  if i!=0 else 64
                    x=Resblock(dim1=dim,dim2=planes[i],is_proj=self.is_proj)(x)
                    dim=planes[i]
                    self.is_proj=False
                else:
                    x=Resblock(dim1=planes[i],dim2=planes[i],is_proj=self.is_proj)(x)
            
        return x
        
input=torch.rand(2,3,224,224)
net=FinalModel()


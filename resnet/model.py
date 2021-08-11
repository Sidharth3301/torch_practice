import torch.nn as nn
from torch.utils.hooks import RemovableHandle

class ConvBlock(nn.Module):
    def __init__(self,dim1,dim2,kernel_size=3,stride=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim1,dim2,kernel_size,stride=stride), 
            nn.ReLU(), 
            nn.BatchNorm2d(dim2)
        )  
    
    def forward(self, x):
        return self.block(x)

class Conv1DBlock(nn.Module):
    def __init__(self,dim1,dim2,kernel_size=3,stride=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim1,dim2,kernel_size,stride), 
            nn.ReLU(), 
            
        )  
    
    def forward(self, x):
        return self.block(x)

class SimpleNetwork(nn.Module):
    def __init__(self, num_resnet_list=[2,2,2,2]):
        super(SimpleNetwork, self).__init__()
        # here we add the individual layers
        k=64
        layers = [ConvBlock(3,64,7,2),ConvBlock(64,64,3,2)]
        for i,j in enumerate(num_resnet_list):
            diff_weight=True
            
            for _ in range(j):
                if diff_weight:
                    layers += [ResnetBlock(k,2*k,diff_weight)]
                    k*=2
                    diff_weight=False
                else:
                    layers+=[ResnetBlock(k,k,diff_weight)]
                    
                

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim1,dim2, use_dropout,diff_weight):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim1,dim2, use_dropout)
        self.diff_weight=diff_weight
        if diff_weight:
            self.projection= self.build_proj(dim1,dim2)

    def build_conv_block(self,dim1,dim2, use_dropout):
        conv_block = []

        if dim1==dim2:
            dim=dim1
            conv_block += [ConvBlock(dim,dim,3,1)]
            if use_dropout:
                conv_block += [nn.Dropout(0.5)]
                
            conv_block += [ConvBlock(dim,dim,3,1)]
        else:
            conv_block += [ConvBlock(dim1,dim2,3,2)]
            if use_dropout:
                conv_block += [nn.Dropout(0.5)]
                
            conv_block += [ConvBlock(dim2,dim2)]

        
        return nn.Sequential(*conv_block)

    def build_proj(self,dim1,dim2):
        return nn.Sequential([Conv1DBlock(dim1,dim2)])

    def forward(self, x):
        if self.diff_weight==False:
            out = x + self.conv_block(x)
        else :
            out= self.projection(x)+self.conv_block(x)
        return out

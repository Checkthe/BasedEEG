import torch
import torch.nn as nn
from modeling.SparableLinearBlock import PerChannelLinear
from baseline.MSCNN import MSCNN

class DiffFeatureExtraction(nn.Module):
    def __init__(self,in_chans,n_points,d_model=128):
        super(DiffFeatureExtraction,self).__init__()
        self.AvgActive = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 2), padding=(0, 0), ceil_mode=True)
        self.PCLArrow = PerChannelLinear(in_features=n_points,out_features=n_points,C=in_chans)
        self.PCLColumn = PerChannelLinear(in_features=5, out_features=5, C=in_chans)
        self.DownSample = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=in_chans,out_channels=in_chans)
        self.Relu = nn.ReLU()
    def forward(self,x):
        #===========差分特征===========
        # x = self.AvgActive(x)
        # x = torch.diff(x)
        #===========特征歧视===========
        dx = self.PCLArrow(x)
        dx = dx.transpose(-2, -1)
        dx = self.PCLColumn(dx)
        dx = dx.transpose(-2, -1)
        # dx = self.DownSample(dx)
        dx = dx + x
        dx = self.Relu(dx)
        return dx #(B,C,10,d_model)

class ERNCNN(nn.Module):
    def __init__(self,in_chans,n_points,n_classes):
        super(ERNCNN,self).__init__()
        self.DiffFeatureExtraction = nn.Sequential(*[DiffFeatureExtraction(in_chans=in_chans,n_points=n_points) for _ in range(1)])
        self.MSCNN = MSCNN(n_classes=n_classes,n_channels=in_chans)
        self.head = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,n_classes)
        )
    def forward(self,x):
        x = self.DiffFeatureExtraction(x)
        x = self.MSCNN(x)
        return x
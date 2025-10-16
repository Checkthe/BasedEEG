import torch
import torch.nn as nn

class ParameterOptimizer(nn.Module):
    def __init__(self,n_channels,n_points):
        super(ParameterOptimizer,self).__init__()
        self.linear_group = nn.Sequential(
            nn.Linear(n_points,128),
            nn.ReLU(128,128),
            nn.Linear(128,64),
            nn.ReLU(64,64),
            nn.Linear(64,1)
        )
        self.linear = nn.Sequential(
            nn.Linear(n_channels,128),
            nn.ReLU(128,64),
            nn.Linear(64,n_channels)
        )
    def forward(self,x,para):
        # x = self.linear_group(x).squeeze(-1)
        # para = x*para
        para = self.linear(para)
        return para

class FilterLayer(nn.Module):
    def __init__(self,n_channels,n_points):
        super(FilterLayer,self).__init__()
        # 初始化滤波参数
        self.time_points = torch.arange(0, n_points, requires_grad=False).repeat(n_channels, 1).unsqueeze(0)
        self.coefficient = torch.full((1, n_channels), 1, dtype=torch.float32, requires_grad=True)
        self.position = torch.full((1, n_channels), 1, dtype=torch.float32, requires_grad=True)
        self.amplitude = torch.full((1, n_channels), 1, dtype=torch.float32, requires_grad=True)
        self.intercept = torch.full((1, n_channels), 0, dtype=torch.float32, requires_grad=True)
        # 参数优化器
        self.L_coe = ParameterOptimizer(n_channels=n_channels, n_points=n_points)
        self.L_pos = ParameterOptimizer(n_channels=n_channels, n_points=n_points)
        self.L_amp = ParameterOptimizer(n_channels=n_channels, n_points=n_points)
        self.L_int = ParameterOptimizer(n_channels=n_channels, n_points=n_points)
        #线性滤波层
        self.LinearFilter = nn.Sequential(
            nn.Linear(n_points,n_points),
            nn.ReLU(n_points,n_points),
            nn.Linear(n_points,n_points)
        )
    def forward(self,x):
        #优化参数
        coefficient = self.L_coe(x,self.coefficient).unsqueeze(-1)
        position = self.L_pos(x,self.position).unsqueeze(-1)
        amplitude = self.L_amp(x,self.amplitude).unsqueeze(-1)
        intercept = self.L_int(x,self.intercept).unsqueeze(-1)
        #构建滤波矩阵
        filter_metrix = amplitude * torch.cos(coefficient * self.time_points + position) + intercept
        x = x - filter_metrix
        x = self.LinearFilter(x)
        return x

class Deepfilter(nn.Module):
    def __init__(self,n_channels,n_points,n_layres=4):
        super(Deepfilter,self).__init__()
        #构建滤波器层
        self.filter = nn.Sequential(
            *[FilterLayer(n_channels=n_channels,n_points=n_points) for _ in range(n_layres)]
        )
    def forward(self,x):
        #构建滤波矩阵
        x = self.filter(x)
        return x
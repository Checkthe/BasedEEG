"""
schirrmeister_nets.py

PyTorch implementations (DeepConvNet, ShallowConvNet, HybridConvNet) consistent with
Schirrmeister et al., 2017.

兼容性修复：
- 若输入为 (N, n_channels, n_times)（3D），forward 自动转为 (N, 1, n_channels, n_times)（4D）。
- 输出为 log-probabilities（使用 F.log_softmax），适配 NLLLoss。

示例：
    python schirrmeister_nets.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_output_length(L_in, kernel_size, stride=1, padding=0, dilation=1):
    """1D 卷积输出长度计算（下取整），用于预计算 classifier 的时域宽度"""
    return math.floor((L_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)


class SafeLog(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class Square(nn.Module):
    def forward(self, x):
        return x ** 2


class DeepConvNet(nn.Module):
    """
    Deep ConvNet (Deep4Net) - Schirrmeister et al., 2017 (PyTorch)
    Input expected: (batch, 1, n_channels, n_times) OR (batch, n_channels, n_times)
    Output: log-probabilities (batch, n_classes)
    """
    def __init__(self, n_channels, n_times, n_classes,
                 first_filters=25, filters_factor=2,
                 temporal_kernel_size=10, pool_kernel_time=3, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.dropout_p = dropout

        # Block 1: temporal conv then spatial conv
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_filters,
            kernel_size=(1, temporal_kernel_size),
            stride=(1, 1),
            bias=False
        )
        self.spatial_conv = nn.Conv2d(
            in_channels=first_filters,
            out_channels=first_filters,
            kernel_size=(n_channels, 1),
            stride=(1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(first_filters, momentum=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, pool_kernel_time), stride=(1, pool_kernel_time))
        self.drop1 = nn.Dropout(p=self.dropout_p)
        self.elu = nn.ELU()

        # Blocks 2..4
        f2 = first_filters * filters_factor
        f3 = f2 * filters_factor
        f4 = f3 * filters_factor

        self.conv2 = nn.Conv2d(first_filters, f2, kernel_size=(1, temporal_kernel_size), bias=False)
        self.bn2 = nn.BatchNorm2d(f2, momentum=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, pool_kernel_time), stride=(1, pool_kernel_time))
        self.drop2 = nn.Dropout(p=self.dropout_p)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=(1, temporal_kernel_size), bias=False)
        self.bn3 = nn.BatchNorm2d(f3, momentum=0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, pool_kernel_time), stride=(1, pool_kernel_time))
        self.drop3 = nn.Dropout(p=self.dropout_p)

        self.conv4 = nn.Conv2d(f3, f4, kernel_size=(1, temporal_kernel_size), bias=False)
        self.bn4 = nn.BatchNorm2d(f4, momentum=0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, pool_kernel_time), stride=(1, pool_kernel_time))
        self.drop4 = nn.Dropout(p=self.dropout_p)

        # 计算经所有卷积与池化后的最终时间长度，用于 classifier kernel width
        t = n_times
        t = _conv_output_length(t, temporal_kernel_size, stride=1, padding=0)  # conv1
        t = _conv_output_length(t, pool_kernel_time, stride=pool_kernel_time)   # pool1
        t = _conv_output_length(t, temporal_kernel_size)                        # conv2
        t = _conv_output_length(t, pool_kernel_time, stride=pool_kernel_time)   # pool2
        t = _conv_output_length(t, temporal_kernel_size)                        # conv3
        t = _conv_output_length(t, pool_kernel_time, stride=pool_kernel_time)   # pool3
        t = _conv_output_length(t, temporal_kernel_size)                        # conv4
        t = _conv_output_length(t, pool_kernel_time, stride=pool_kernel_time)   # pool4
        final_time = t
        if final_time < 1:
            raise ValueError(f"Final temporal size < 1 (got {final_time}). "
                             "请增大 n_times 或减小 kernel/pool 大小。")

        # classifier convolution covers the remaining time axis
        self.classifier = nn.Conv2d(f4, n_classes, kernel_size=(1, final_time), bias=True)

    def forward(self, x, return_logits=False):
        # 兼容 3D 或 4D 输入
        # 如果输入来自 DataLoader: (N, n_channels, n_times) -> 转为 (N, 1, n_channels, n_times)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Unexpected input dim: {x.dim()}. Expect 3D (N,C,T) or 4D (N,1,C,T), got shape {x.shape}")

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x); x = self.bn2(x); x = self.elu(x); x = self.pool2(x); x = self.drop2(x)
        x = self.conv3(x); x = self.bn3(x); x = self.elu(x); x = self.pool3(x); x = self.drop3(x)
        x = self.conv4(x); x = self.bn4(x); x = self.elu(x); x = self.pool4(x); x = self.drop4(x)

        x = self.classifier(x)   # shape: (batch, n_classes, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # -> (batch, n_classes)
        if return_logits:
            return x
        return F.log_softmax(x, dim=1)


class ShallowConvNet(nn.Module):
    """
    Shallow ConvNet (FBCSP-like) - Schirrmeister et al., 2017
    Input: (batch, 1, n_channels, n_times) OR (batch, n_channels, n_times)
    Output: log-probabilities (batch, n_classes)
    """
    def __init__(self, n_channels, n_times, n_classes,
                 n_filters_time=40, temporal_kernel_size=25,
                 pool_kernel_time=75, pool_stride_time=15, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.dropout_p = dropout

        self.conv_time = nn.Conv2d(1, n_filters_time, kernel_size=(1, temporal_kernel_size), bias=False)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_time, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_time, momentum=0.1)

        self.square = Square()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel_time), stride=(1, pool_stride_time))
        self.log = SafeLog(eps=1e-6)
        self.drop = nn.Dropout(p=self.dropout_p)

        t = n_times
        t = _conv_output_length(t, temporal_kernel_size)  # after conv_time
        t = _conv_output_length(t, pool_kernel_time, stride=pool_stride_time)
        if t < 1:
            raise ValueError(f"Final temporal size < 1 (got {t}). adjust kernel/stride or n_times.")
        self.classifier = nn.Conv2d(n_filters_time, n_classes, kernel_size=(1, t), bias=True)

    def forward(self, x, return_logits=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Unexpected input dim: {x.dim()}. Expect 3D (N,C,T) or 4D (N,1,C,T). Got shape {x.shape}")

        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.log(x)
        x = self.drop(x)

        x = self.classifier(x)
        x = x.squeeze(-1).squeeze(-1)
        if return_logits:
            return x
        return F.log_softmax(x, dim=1)


class HybridConvNet(nn.Module):
    """
    Hybrid: concatenate Deep 和 Shallow 两个分支的特征做最终分类
    输入同上。为简单起见，两个分支从头并行运行，然后各自通过 1x1 投影，
    使用自适应池将时域降为 1，再通道拼接并分类。
    """
    def __init__(self, deep_cfg: dict, shallow_cfg: dict):
        assert deep_cfg['n_channels'] == shallow_cfg['n_channels']
        assert deep_cfg['n_times'] == shallow_cfg['n_times']
        assert deep_cfg['n_classes'] == shallow_cfg['n_classes']
        n_classes = deep_cfg['n_classes']
        super().__init__()
        self.deep = DeepConvNet(**deep_cfg)
        self.shallow = ShallowConvNet(**shallow_cfg)

        deep_out_c = self.deep.conv4.out_channels
        shallow_out_c = self.shallow.conv_spat.out_channels

        self.project_deep = nn.Conv2d(deep_out_c, 60, kernel_size=(1, 1), bias=True)
        self.project_shallow = nn.Conv2d(shallow_out_c, 40, kernel_size=(1, 1), bias=True)

        self.classifier = nn.Conv2d(60 + 40, n_classes, kernel_size=(1, 1), bias=True)

    def forward(self, x, return_logits=False):
        # 允许 3D 或 4D 输入
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Unexpected input dim: {x.dim()}. Expect 3D (N,C,T) or 4D (N,1,C,T). Got shape {x.shape}")

        # Deep branch (重复 DeepConvNet 的前向直到 conv4/pool4)
        d = self.deep.temporal_conv(x)
        d = self.deep.spatial_conv(d)
        d = self.deep.bn1(d); d = self.deep.elu(d); d = self.deep.pool1(d); d = self.deep.drop1(d)
        d = self.deep.conv2(d); d = self.deep.bn2(d); d = self.deep.elu(d); d = self.deep.pool2(d); d = self.deep.drop2(d)
        d = self.deep.conv3(d); d = self.deep.bn3(d); d = self.deep.elu(d); d = self.deep.pool3(d); d = self.deep.drop3(d)
        d = self.deep.conv4(d); d = self.deep.bn4(d); d = self.deep.elu(d); d = self.deep.pool4(d); d = self.deep.drop4(d)
        # d: (batch, deep_out_c, 1, t_d)

        # Shallow branch (到 pool)
        s = self.shallow.conv_time(x); s = self.shallow.conv_spat(s); s = self.shallow.bn(s)
        s = self.shallow.square(s); s = self.shallow.pool(s); s = self.shallow.log(s); s = self.shallow.drop(s)
        # s: (batch, shallow_out_c, 1, t_s)

        pd = self.project_deep(d)     # -> (batch, 60, 1, t_d)
        ps = self.project_shallow(s)  # -> (batch, 40, 1, t_s)

        pd = F.adaptive_avg_pool2d(pd, output_size=(1, 1))
        ps = F.adaptive_avg_pool2d(ps, output_size=(1, 1))

        cat = torch.cat([pd, ps], dim=1)  # -> (batch, 100, 1, 1)
        out = self.classifier(cat)        # -> (batch, n_classes, 1, 1)
        out = out.squeeze(-1).squeeze(-1)
        if return_logits:
            return out
        return F.log_softmax(out, dim=1)



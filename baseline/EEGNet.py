import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels, time_points, n_classes,
                 F1=4, D=2, dropout=0.5):
        super(EEGNet, self).__init__()
        F2 = F1 * D
        kernel_time = 64
        separable_kernel_time = 16

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_time),
                               bias=False, padding=(0, kernel_time//2))
        self.bn1 = nn.BatchNorm2d(F1)
        self.depth_conv = nn.Conv2d(F1, F1*D, (n_channels, 1),
                                    groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2
        self.separable_conv = nn.Conv2d(F1*D, F1*D, (1, separable_kernel_time),
                                        groups=F1*D, bias=False,
                                        padding=(0, separable_kernel_time//2))
        self.pointwise_conv = nn.Conv2d(F1*D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # 使用 dummy input 自动算 fc 输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, time_points).unsqueeze(1)
            feat = self._forward_features(dummy)
            fc_in_dim = feat.view(1, -1).size(1)
        self.fc = nn.Linear(fc_in_dim, n_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depth_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, n_channels, time_points)
        x = self._forward_features(x)
        x = torch.flatten(x, 1)  # (batch, feat_dim)
        x = self.fc(x)
        return x

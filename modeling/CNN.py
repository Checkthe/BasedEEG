import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DBlock(nn.Module):
    """Conv -> BN -> ReLU -> (optional) Pool"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, pool_kernel=2):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_kernel) if pool_kernel is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class Conv1DClassifier(nn.Module):
    """
    A compact, robust 1D Conv model.
    Input shape (batch, in_channels, seq_len)
    Returns logits (batch, num_classes)
    """
    def __init__(self, in_channels, num_classes, channels=[32,64,128], kernel_sizes=[7,5,3],
                 pool_kernel=2, dropout=0.4, use_global_pool=True):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        layers = []
        ch_in = in_channels
        for ch_out, k in zip(channels, kernel_sizes):
            layers.append(Conv1DBlock(ch_in, ch_out, kernel_size=k, pool_kernel=pool_kernel))
            ch_in = ch_out
        self.features = nn.Sequential(*layers)
        self.use_global_pool = use_global_pool
        if use_global_pool:
            self.global_pool = nn.AdaptiveAvgPool1d(1)  # output shape (B, C, 1)
            fc_in = channels[-1]
        else:
            # if not using global pool, user must compute flattened length or pass example input to infer
            fc_in = channels[-1] *  (None)  # placeholder - prefer use_global_pool=True
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, C, L)
        if len(x.shape)==2:
            x = x.unsqueeze(1)
        x = self.features(x)
        if self.use_global_pool:
            x = self.global_pool(x)      # (B, C, 1)
            x = x.view(x.size(0), -1)    # (B, C)
        else:
            x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

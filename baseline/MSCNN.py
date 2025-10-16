import torch
import torch.nn as nn
from typing import List, Optional, Callable, Union


ActivationSpec = Union[nn.Module, Callable[[], nn.Module], type]

def _make_activation(activation: Optional[ActivationSpec]) -> nn.Module:
    if activation is None:
        return nn.ELU()
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()
    if callable(activation):
        try:
            return activation()
        except Exception:
            # called but failed — fall back to ELU
            return nn.ELU()
    if isinstance(activation, nn.Module):
        # try to create a fresh instance of same class (best-effort),
        # otherwise reuse (safe for stateless activations)
        try:
            return activation.__class__()
        except Exception:
            return activation
    raise TypeError("Unsupported activation spec")


class MSCNNBranch(nn.Module):
    """
    Single multi-scale branch.
    Input: (B, 1, n_channels, n_time)
    Output: (B, temporal_filters, 1, T_out)
    """
    def __init__(
        self,
        n_channels: int,
        temporal_kernel_size: int,
        temporal_filters: int = 8,
        pool_kernel_time: int = 4,
        pool_stride_time: int = 4,
        dropout: float = 0.5,
        activation: Optional[ActivationSpec] = None
    ):
        super().__init__()
        pad = temporal_kernel_size // 2
        self.temporal_conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=temporal_filters,
            kernel_size=(1, temporal_kernel_size),
            padding=(0, pad),
            bias=False
        )
        # spatial conv to aggregate across channels -> output height becomes 1
        self.spatial_conv = nn.Conv2d(
            in_channels=temporal_filters,
            out_channels=temporal_filters,
            kernel_size=(5, 1),
            bias=False
        )
        self.bn = nn.BatchNorm2d(temporal_filters)
        self.activation = _make_activation(activation)
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_kernel_time),
            stride=(1, pool_stride_time)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.temporal_conv(x)   # -> (B, F_t, C, T)
        x = self.spatial_conv(x)    # -> (B, F_t, 1, T)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)            # -> (B, F_t, 1, T_out)
        x = self.dropout(x)
        return x


class MSCNN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        temporal_kernels: Optional[List[int]] = None,
        temporal_filters: int = 16,
        pool_kernel_time: int = 8,
        pool_stride_time: int = 8,
        fc_units: int = 128,
        dropout: float = 0.5,
        activation: Optional[ActivationSpec] = None
    ):
        super().__init__()
        if temporal_kernels is None:
            temporal_kernels = [3, 7, 15]

        self.n_branches = len(temporal_kernels)
        # create branches (each branch will create its own activation instance)
        self.branches = nn.ModuleList([
            MSCNNBranch(
                n_channels=n_channels,
                temporal_kernel_size=k,
                temporal_filters=temporal_filters,
                pool_kernel_time=pool_kernel_time,
                pool_stride_time=pool_stride_time,
                dropout=dropout,
                activation=activation
            ) for k in temporal_kernels
        ])

        self.concat_channels = temporal_filters * self.n_branches
        self.fuse_conv = nn.Conv2d(
            in_channels=self.concat_channels,
            out_channels=self.concat_channels,
            kernel_size=(1, 1),
            bias=False
        )
        self.fuse_bn = nn.BatchNorm2d(self.concat_channels)
        self.fuse_activation = _make_activation(activation)

        # global pooling to (1,1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        if fc_units > 0:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.concat_channels, fc_units, bias=True),
                nn.BatchNorm1d(fc_units),
                _make_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(fc_units, n_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.concat_channels, n_classes)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if getattr(m, 'weight', None) is not None:
                    nn.init.constant_(m.weight, 1.0)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        x: (B, C, T) or (B, 1, C, T)
        return_features: if True return (logits, features) where features are the
                         flattened pooled features before classifier.
        """
        if x.dim() == 3:  # (B, C, T)
            x = x.unsqueeze(1)  # -> (B, 1, C, T)

        # basic validation (time dimension should be >= 1 after pooling)
        if x.size(2) <= 0:
            raise ValueError("n_channels (height) must be > 0")
        if x.size(3) <= 0:
            raise ValueError("time dimension must be > 0")

        branch_outs = [b(x) for b in self.branches]  # each -> (B, F, 1, T_out)
        x_cat = torch.cat(branch_outs, dim=1)        # (B, concat_channels, 1, T_out)

        x = self.fuse_conv(x_cat)
        x = self.fuse_bn(x)
        x = self.fuse_activation(x)

        x = self.global_pool(x)                      # (B, concat_channels, 1, 1)
        features = torch.flatten(x, 1)               # (B, concat_channels)
        logits = self.classifier(x)                  # (B, n_classes)

        if return_features:
            return logits, features
        return logits

from typing import Optional, List
import math
import torch
from torch import nn
import torch.nn.functional as F

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm applied over channels for (N, C, H, W) by converting to channels-last.
    Matches official implementation behavior (eps=1e-6).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)

class StochasticDepth(nn.Module):
    """
    Stochastic Depth per sample (a.k.a. DropPath). Modeled after torchvision/timm style.
    drop_prob: probability to drop the path (0.0 - 1.0).
    """
    def __init__(self, drop_prob: float = 0.0, mode: str = "row"):
        super().__init__()
        self.drop_prob = drop_prob
        self.mode = mode

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # For "row" mode, shape = (batch, 1, 1, 1) to broadcast across channels & spatial dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ---------------------------
# ConvNeXt Block (CNBlock)
# ---------------------------
class CNBlock(nn.Module):
    """
    ConvNeXt block:
      - depthwise conv 7x7 (groups=dim)
      - LayerNorm (channels-last) + Linear(in=dim, out=4*dim) + GELU + Linear(4*dim, dim)
      - layer_scale (gamma) and stochastic depth
    """

    def __init__(self, dim: int, layer_scale: float = 1e-6, stochastic_depth_prob: float = 0.0,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = lambda normalized_shape: nn.LayerNorm(normalized_shape, eps=1e-6)

        # depthwise conv
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),                 # to (N, H, W, C)
            norm_layer(dim),                       # LayerNorm over channels (channels-last)
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),                 # back to (N, C, H, W)
        )

        # layer scale (shape (C,1,1) to broadcast across spatial dims)
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale, requires_grad=True)

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x)
        res = res * self.layer_scale
        res = self.stochastic_depth(res)
        return x + res

class ConvNeXt(nn.Module):
    def __init__(self,
                 in_chans: int = 1,
                 num_classes: int = 1000,
                 depths: List[int] = [3, 3, 9, 3],
                 dims: List[int] = [96, 192, 384, 768],
                 stochastic_depth_prob: float = 0.0,
                 layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.0):
        """
        depths: number of CNBlocks per stage
        dims: channel dims for stages
        stochastic_depth_prob: max drop probability (linearly assigned across blocks)
        """
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4, "depths and dims must be lists of length 4"

        # stem / patchify (conv with stride=4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=(1,4), stride=(1,4))
        )

        # build stages
        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        # linearly spaced drop probs from 0 to stochastic_depth_prob across total_blocks
        if total_blocks > 1:
            dpr = [x.item() for x in torch.linspace(0.0, stochastic_depth_prob, total_blocks)]
        else:
            dpr = [0.0] * total_blocks

        cur = 0
        for stage_idx in range(4):
            blocks = []
            for i in range(depths[stage_idx]):
                blocks.append(
                    CNBlock(dim=dims[stage_idx],
                            layer_scale=layer_scale_init_value,
                            stochastic_depth_prob=dpr[cur + i],
                            norm_layer=lambda normalized_shape: nn.LayerNorm(normalized_shape, eps=1e-6))
                )
            cur += depths[stage_idx]
            self.stages.append(nn.Sequential(*blocks))
            # add downsample between stages (except after last stage)
            if stage_idx < 3:
                # Official uses a LayerNorm followed by a Conv2d(kernel=2, stride=2) as downsample.
                # Applying LayerNorm in channels-last requires a small wrapper; but for simplicity and
                # equivalence we apply LayerNorm2d (channels-last) BEFORE conv if desired.
                self.stages.append(
                    nn.Sequential(
                        LayerNorm2d(dims[stage_idx]),
                        nn.Conv2d(dims[stage_idx], dims[stage_idx + 1], kernel_size=2, stride=2)
                    )
                )

        # final norm (channels-last)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # initialization to match official heuristics
        self._init_weights(head_init_scale)

    def _init_weights(self, head_init_scale: float = 1.0):
        # follow official init style: truncated normal for conv/linear weights and zeros for biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # use normal truncation-like initialization (approximate)
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

        # head special init scaling as done in many official implementations
        if head_init_scale != 1.0:
            nn.init.normal_(self.head.weight, std=0.01 * head_init_scale)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # -> (N, C, H/4, W/4)
        # run through interleaved stages (blocks and downsample layers)
        for layer in self.stages:
            x = layer(x)
        # global pool
        x = x.mean([-2, -1])  # (N, C)
        x = self.norm(x)      # LayerNorm over channel dim (applied as 1D)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        # x = x.repeat(1,3,1,1)
        x = self.forward_features(x)
        x = self.head(x)
        return x

# ---------------------------
# Factory helpers (same variants as official)
# ---------------------------
def _convnext(depths: List[int], dims: List[int], num_classes: int = 1000, stochastic_depth_prob: float = 0.0):
    return ConvNeXt(depths=depths, dims=dims, num_classes=num_classes, stochastic_depth_prob=stochastic_depth_prob)

def convnext_tiny(num_classes: int = 1000, stochastic_depth_prob: float = 0.1):
    return _convnext(depths=[3,3,9,3], dims=[96,192,384,768], num_classes=num_classes,
                     stochastic_depth_prob=stochastic_depth_prob)

def convnext_small(num_classes: int = 1000, stochastic_depth_prob: float = 0.4):
    return _convnext(depths=[3,3,27,3], dims=[96,192,384,768], num_classes=num_classes,
                     stochastic_depth_prob=stochastic_depth_prob)

def convnext_base(num_classes: int = 1000, stochastic_depth_prob: float = 0.5):
    return _convnext(depths=[3,3,27,3], dims=[128,256,512,1024], num_classes=num_classes,
                     stochastic_depth_prob=stochastic_depth_prob)

def convnext_large(num_classes: int = 1000, stochastic_depth_prob: float = 0.5):
    return _convnext(depths=[3,3,27,3], dims=[192,384,768,1536], num_classes=num_classes,
                     stochastic_depth_prob=stochastic_depth_prob)

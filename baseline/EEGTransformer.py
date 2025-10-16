import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialChannelAttention(nn.Module):
    """
    Applies self-attention across EEG channels.
    Input:  (batch, n_channels, time_points)
    Output: (batch, n_channels, time_points)
    """
    def __init__(self, n_channels: int, time_points: int, dropout: float = 0.3, num_heads: int = 1):
        super().__init__()
        # Each channel is a token; embedding = raw time series of length 'time_points'.
        self.attn = nn.MultiheadAttention(embed_dim=time_points, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(time_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (C, B, L) for MHA (batch_first=False)
        x_perm = x.permute(1, 0, 2)  # (C, B, L)
        # Pre-norm
        x_norm = self.norm(x_perm)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # (C, B, L)
        out = x_perm + self.dropout(attn_out)
        return out.permute(1, 0, 2)  # (B, C, L)


class TemporalTransformerBlock(nn.Module):
    """
    Single pre-norm Transformer block for temporal slices.
    Input: (batch, seq_len, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, expansion_factor: int = 4, dropout: float = 0.5):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) -> (S, B, D)
        x_perm = x.permute(1, 0, 2)

        # MHA with pre-norm
        x1 = self.norm1(x_perm)
        attn_out, _ = self.attn(x1, x1, x1)
        x_perm = x_perm + self.dropout1(attn_out)

        # FFN with pre-norm
        x2 = self.norm2(x_perm)
        ff_out = self.ff(x2)
        x_perm = x_perm + self.dropout2(ff_out)

        return x_perm.permute(1, 0, 2)  # (B, S, D)


class S3T(nn.Module):
    """
    S3T: Spatial-Temporal Tiny Transformer for EEG decoding.
      - Channel-wise attention (SpatialChannelAttention).
      - Temporal conv for positional/contextual mixing.
      - Compress channels -> slice -> Transformer layers.
      - Global average pooling -> classifier.
    """
    def __init__(
        self,
        n_channels: int,
        time_points: int,
        n_classes: int,
        slice_size: int = 10,
        conv_kernel: int = 10,
        conv_stride: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        expansion_factor: int = 4,
        spatial_heads: int = 1,
        conv_padding: int = 0,   # allow user control; 0 matches your original
        dropout_attn: float = 0.3,
        dropout_block: float = 0.5,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."

        # 1) Spatial/channel attention
        self.spatial_attn = SpatialChannelAttention(n_channels, time_points, dropout=dropout_attn, num_heads=spatial_heads)

        # 2) Temporal conv (learned positional/contextual mixing along time)
        self.pos_conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=conv_kernel,
            stride=conv_stride,
            padding=conv_padding,
        )

        # 3) Channel compression
        self.compress = nn.Conv1d(in_channels=n_channels, out_channels=1, kernel_size=1)

        # 4) Slicing + embedding
        self.slice_size = slice_size
        self.slice_embed = nn.Linear(slice_size, d_model)

        # 5) Stacked temporal Transformer blocks
        self.temporal_layers = nn.ModuleList(
            [TemporalTransformerBlock(d_model, n_heads, expansion_factor, dropout=dropout_block) for _ in range(n_layers)]
        )

        # 6) Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # 1. Spatial channel attention
        x = self.spatial_attn(x)  # (B, C, T)

        # 2. Temporal conv
        x = self.pos_conv(x)      # (B, C, T2)
        # 3. Compress channels
        x = self.compress(x)      # (B, 1, T2)

        B, _, T2 = x.shape

        # --- Make T2 a multiple of slice_size by right-padding zeros (safe, deterministic) ---
        remainder = T2 % self.slice_size
        if remainder != 0:
            pad_len = self.slice_size - remainder
            # pad format for 1D: (pad_left, pad_right) on last dim
            x = F.pad(x, (0, pad_len), mode="constant", value=0.0)
            T2 = T2 + pad_len

        # 4. Slice into segments of length slice_size
        n_slices = T2 // self.slice_size
        x = x.view(B, 1, n_slices, self.slice_size).squeeze(1)  # (B, n_slices, slice_size)

        # 5. Embed slices + Transformer
        x = self.slice_embed(x)  # (B, n_slices, D)
        for layer in self.temporal_layers:
            x = layer(x)         # (B, n_slices, D)

        # 6. Pool and classify
        x = x.mean(dim=1)        # (B, D)
        logits = self.classifier(x)  # (B, n_classes)
        return logits

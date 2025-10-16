import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EEGDenoisingAutoencoder(nn.Module):
    """
    EEG 降噪自编码器（输入: (B, channels, length)）。
    流程:
      conv_downsample -> (B,1,1000) -> squeeze -> (B,1000)
      -> add_noise (训练时) -> encoder_mlp -> latent (B, z_dim)
      -> decoder_mlp -> recon (B,1000)
    forward 返回: latent (默认) 或 (latent, recon) 当 return_recon=True 时。
    """
    def __init__(
        self,
        in_channels: int,
        conv_channels: int = 64,
        conv_blocks: int = 1,
        latent_dim: int = 128,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.noise_std = noise_std
        # 卷积特征提取 / 降采样模块（1D）
        convs = []
        cur_channels = in_channels
        for i in range(conv_blocks):
            out_ch = conv_channels if i == 0 else conv_channels
            # 使用 stride=2 的 conv 来做一些时间域上的下采样 + 提取特征
            convs.append(nn.Conv1d(cur_channels, out_ch, kernel_size=7, stride=2, padding=3, bias=False))
            convs.append(nn.BatchNorm1d(out_ch))
            convs.append(nn.ReLU(inplace=True))
            cur_channels = out_ch
        # 最后用 1x1 conv 聚合通道 -> 1 通道
        convs.append(nn.Conv1d(cur_channels, 1, kernel_size=1, stride=1, padding=0, bias=False))
        convs.append(nn.ReLU(inplace=True))
        self.feature_extractor = nn.Sequential(*convs)

        # 将时间轴固定到长度 1000
        self.pool_to_1000 = nn.AdaptiveAvgPool1d(output_size=1000)

        # Denoising autoencoder (MLP encoder/decoder)
        self.encoder = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1000),
        )

        # 可选的归一化/激活（根据需要）
        self.latent_norm = nn.LayerNorm(latent_dim)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """训练时在向量上加入高斯噪声（可调 std）"""
        if not self.training or self.noise_std <= 0.0:
            return x
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def forward(self, x: torch.Tensor, return_recon: bool = False):
        """
        x: (B, C, L)
        return:
          latent (B, latent_dim)  or  (latent, recon) if return_recon=True
        recon is shape (B, 1000) representing reconstruction of the pooled vector.
        """
        # Conv 特征提取 -> (B, 1, T_var)
        feats = self.feature_extractor(x)  # shape (B,1,T')
        # Pool/resize 时间轴到固定 1000 -> (B,1,1000)
        pooled = self.pool_to_1000(feats)  # (B,1,1000)
        vec1000 = pooled.squeeze(1)        # (B,1000)

        # 加噪（训练时）
        noisy = self.add_noise(vec1000)

        # Encoder -> latent
        latent = self.encoder(noisy)      # (B, latent_dim)
        latent = self.latent_norm(latent)

        if return_recon:
            recon = self.decoder(latent)  # (B,1000)
            return latent, recon
        return latent


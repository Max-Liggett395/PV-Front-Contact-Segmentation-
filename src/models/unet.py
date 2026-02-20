"""Custom U-Net architecture matching the original training notebook."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """U-Net with 5 encoder blocks, 4 decoder blocks, skip connections, and dropout.

    Architecture:
        Encoder: 64 → 128 → 256 → 512 → 1024 (bottleneck)
        Decoder: 512 → 256 → 128 → 64 → num_classes
        Dropout: 0.3 after each pool and before each decoder conv block
        Activation: ReLU (no BatchNorm)
        Skip connections with padding for size mismatch
    """

    def __init__(self, in_channels=1, num_classes=6, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # Encoder
        self.enc1a = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.enc1b = nn.Conv2d(64, 64, 3, 1, 1)
        self.enc2a = nn.Conv2d(64, 128, 3, 1, 1)
        self.enc2b = nn.Conv2d(128, 128, 3, 1, 1)
        self.enc3a = nn.Conv2d(128, 256, 3, 1, 1)
        self.enc3b = nn.Conv2d(256, 256, 3, 1, 1)
        self.enc4a = nn.Conv2d(256, 512, 3, 1, 1)
        self.enc4b = nn.Conv2d(512, 512, 3, 1, 1)

        # Bottleneck
        self.bottleneck_a = nn.Conv2d(512, 1024, 3, 1, 1)
        self.bottleneck_b = nn.Conv2d(1024, 1024, 3, 1, 1)

        # Decoder (transposed convolutions for upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        self.dec1a = nn.Conv2d(1024, 512, 3, 1, 1)
        self.dec1b = nn.Conv2d(512, 512, 3, 1, 1)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2, 0)
        self.dec2a = nn.Conv2d(512, 256, 3, 1, 1)
        self.dec2b = nn.Conv2d(256, 256, 3, 1, 1)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.dec3a = nn.Conv2d(256, 128, 3, 1, 1)
        self.dec3b = nn.Conv2d(128, 128, 3, 1, 1)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.dec4a = nn.Conv2d(128, 64, 3, 1, 1)
        self.dec4b = nn.Conv2d(64, 64, 3, 1, 1)

        # Output
        self.out = nn.Conv2d(64, num_classes, 1, 1, 0)

    def _downsample(self, x, conv_a, conv_b):
        f = F.relu(conv_a(x))
        f = F.relu(conv_b(f))
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p

    def _upsample(self, x, skip, up_conv, conv_a, conv_b):
        x = up_conv(x)
        # Pad if sizes don't match
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        x = F.relu(conv_a(x))
        x = F.relu(conv_b(x))
        return x

    def forward(self, x):
        # Encoder
        f1, p1 = self._downsample(x, self.enc1a, self.enc1b)
        f2, p2 = self._downsample(p1, self.enc2a, self.enc2b)
        f3, p3 = self._downsample(p2, self.enc3a, self.enc3b)
        f4, p4 = self._downsample(p3, self.enc4a, self.enc4b)

        # Bottleneck
        bn = F.relu(self.bottleneck_a(p4))
        bn = F.relu(self.bottleneck_b(bn))

        # Decoder
        u1 = self._upsample(bn, f4, self.up1, self.dec1a, self.dec1b)
        u2 = self._upsample(u1, f3, self.up2, self.dec2a, self.dec2b)
        u3 = self._upsample(u2, f2, self.up3, self.dec3a, self.dec3b)
        u4 = self._upsample(u3, f1, self.up4, self.dec4a, self.dec4b)

        return self.out(u4)

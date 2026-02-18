"""
U-Net Architecture for SEM Image Segmentation

This module implements the U-Net architecture used in the paper:
"Semantic Segmentation for Cross-Sectional Scanning Electron Microscopy Images
of Photovoltaic Cell Metallization: A Deep Learning Approach" (2025)

Architecture details:
- Encoder-decoder structure with skip connections
- 5 encoder blocks with max pooling and dropout (p=0.3)
- BatchNorm after every convolution, before ReLU
- Bottleneck with double convolution
- 4 decoder blocks with transposed convolution and skip connections
- Output: 6 classes (Background, Ag, Glass, Si, Void, Interfacial Void)
- Optional FiLM (Feature-wise Linear Modulation) conditioning on magnification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Applies an affine transformation (scale and shift) to feature maps,
    conditioned on an external input (e.g., magnification embedding).

    Args:
        condition_dim (int): Dimension of the conditioning vector
        num_features (int): Number of feature map channels to modulate
    """

    def __init__(self, condition_dim, num_features):
        super().__init__()
        self.gamma_fc = nn.Linear(condition_dim, num_features)
        self.beta_fc = nn.Linear(condition_dim, num_features)

        # Initialize gamma near 1.0 (identity scale) and beta near 0.0 (no shift)
        nn.init.ones_(self.gamma_fc.weight.data[:, 0] if condition_dim > 0 else self.gamma_fc.weight.data)
        nn.init.zeros_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.weight.data)
        nn.init.zeros_(self.beta_fc.bias.data)

    def forward(self, x, condition):
        """
        Args:
            x: Feature maps [B, C, H, W]
            condition: Conditioning vector [B, condition_dim]

        Returns:
            Modulated feature maps [B, C, H, W]
        """
        gamma = self.gamma_fc(condition).unsqueeze(-1).unsqueeze(-1) + 1.0  # Center around 1
        beta = self.beta_fc(condition).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class UNet(nn.Module):
    """
    U-Net model for semantic segmentation of SEM images.

    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        num_classes (int): Number of output classes (default: 6)
        dropout (float): Dropout probability (default: 0.3)
        num_magnifications (int or None): Number of magnification categories for FiLM.
            If None, FiLM conditioning is disabled (backward compatible).
        film_embedding_dim (int): Dimension of magnification embedding (default: 32)
    """

    def __init__(self, in_channels=1, num_classes=6, dropout=0.3,
                 num_magnifications=None, film_embedding_dim=32):
        super(UNet, self).__init__()

        self.film_enabled = num_magnifications is not None

        # FiLM conditioning
        if self.film_enabled:
            self.mag_embedding = nn.Embedding(num_magnifications, film_embedding_dim)
            # FiLM layers after each encoder block (4) + bottleneck (1) = 5
            self.film1 = FiLMLayer(film_embedding_dim, 64)
            self.film2 = FiLMLayer(film_embedding_dim, 128)
            self.film3 = FiLMLayer(film_embedding_dim, 256)
            self.film4 = FiLMLayer(film_embedding_dim, 512)
            self.film5 = FiLMLayer(film_embedding_dim, 1024)  # Bottleneck

        # Encoder (downsampling path)
        self.enc1 = nn.LazyConv2d(64, 3, 1, 1)
        self.bn_enc1 = nn.BatchNorm2d(64)
        self.enc1b = nn.LazyConv2d(64, 3, 1, 1)
        self.bn_enc1b = nn.BatchNorm2d(64)

        self.enc2 = nn.LazyConv2d(128, 3, 1, 1)
        self.bn_enc2 = nn.BatchNorm2d(128)
        self.enc2b = nn.LazyConv2d(128, 3, 1, 1)
        self.bn_enc2b = nn.BatchNorm2d(128)

        self.enc3 = nn.LazyConv2d(256, 3, 1, 1)
        self.bn_enc3 = nn.BatchNorm2d(256)
        self.enc3b = nn.LazyConv2d(256, 3, 1, 1)
        self.bn_enc3b = nn.BatchNorm2d(256)

        self.enc4 = nn.LazyConv2d(512, 3, 1, 1)
        self.bn_enc4 = nn.BatchNorm2d(512)
        self.enc4b = nn.LazyConv2d(512, 3, 1, 1)
        self.bn_enc4b = nn.BatchNorm2d(512)

        self.enc5 = nn.LazyConv2d(1024, 3, 1, 1)
        self.bn_enc5 = nn.BatchNorm2d(1024)
        self.enc5b = nn.LazyConv2d(1024, 3, 1, 1)
        self.bn_enc5b = nn.BatchNorm2d(1024)

        # Decoder (upsampling path)
        self.dec1 = nn.LazyConvTranspose2d(512, 2, 2, 0)
        self.dec2 = nn.LazyConvTranspose2d(256, 2, 2, 0)
        self.dec3 = nn.LazyConvTranspose2d(128, 2, 2, 0)
        self.dec4 = nn.LazyConvTranspose2d(64, 2, 2, 0)

        # Decoder convolutions
        self.conv1a = nn.LazyConv2d(64, 3, 1, 1)
        self.conv1b = nn.LazyConv2d(64, 3, 1, 1)
        self.conv2a = nn.LazyConv2d(128, 3, 1, 1)
        self.conv2b = nn.LazyConv2d(128, 3, 1, 1)
        self.conv3a = nn.LazyConv2d(256, 3, 1, 1)
        self.conv3b = nn.LazyConv2d(256, 3, 1, 1)
        self.conv4a = nn.LazyConv2d(512, 3, 1, 1)
        self.conv4b = nn.LazyConv2d(512, 3, 1, 1)
        self.conv5a = nn.LazyConv2d(1024, 3, 1, 1)
        self.conv5b = nn.LazyConv2d(1024, 3, 1, 1)

        # Decoder BatchNorm layers
        self.bn_conv5a = nn.BatchNorm2d(1024)
        self.bn_conv5b = nn.BatchNorm2d(1024)
        self.bn_conv4a = nn.BatchNorm2d(512)
        self.bn_conv4b = nn.BatchNorm2d(512)
        self.bn_conv3a = nn.BatchNorm2d(256)
        self.bn_conv3b = nn.BatchNorm2d(256)
        self.bn_conv2a = nn.BatchNorm2d(128)
        self.bn_conv2b = nn.BatchNorm2d(128)

        # Regularization
        self.dropout = nn.Dropout(p=dropout)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # Output layer (1x1 convolution for classification)
        self.out = nn.LazyConv2d(num_classes, 1, 1, 0)

    def double_conv_block1(self, x):
        """Encoder block 1: Two 3x3 convolutions with BN + ReLU"""
        x = F.relu(self.bn_enc1(self.enc1(x)))
        x = F.relu(self.bn_enc1b(self.enc1b(x)))
        return x

    def double_conv_block2(self, x):
        """Encoder block 2: Two 3x3 convolutions with BN + ReLU"""
        x = F.relu(self.bn_enc2(self.enc2(x)))
        x = F.relu(self.bn_enc2b(self.enc2b(x)))
        return x

    def double_conv_block3(self, x):
        """Encoder block 3: Two 3x3 convolutions with BN + ReLU"""
        x = F.relu(self.bn_enc3(self.enc3(x)))
        x = F.relu(self.bn_enc3b(self.enc3b(x)))
        return x

    def double_conv_block4(self, x):
        """Encoder block 4: Two 3x3 convolutions with BN + ReLU"""
        x = F.relu(self.bn_enc4(self.enc4(x)))
        x = F.relu(self.bn_enc4b(self.enc4b(x)))
        return x

    def double_conv_block5(self, x):
        """Bottleneck: Two 3x3 convolutions with BN + ReLU"""
        x = F.relu(self.bn_enc5(self.enc5(x)))
        x = F.relu(self.bn_enc5b(self.enc5b(x)))
        return x

    def up_double_conv_block1(self, x):
        """Decoder block 1: Two 3x3 convolutions with BN + ReLU"""
        x = F.relu(self.bn_conv5a(self.conv5a(x)))
        x = F.relu(self.bn_conv5b(self.conv5b(x)))
        return x

    def up_double_conv_block2(self, x):
        """Decoder block 2: Two 3x3 convolutions with BN + ReLU"""
        x = F.relu(self.bn_conv4a(self.conv4a(x)))
        x = F.relu(self.bn_conv4b(self.conv4b(x)))
        return x

    def up_double_conv_block3(self, x):
        """Decoder block 3: Two 3x3 convolutions with BN + ReLU"""
        x = F.relu(self.bn_conv3a(self.conv3a(x)))
        x = F.relu(self.bn_conv3b(self.conv3b(x)))
        return x

    def up_double_conv_block4(self, x):
        """Decoder block 4: Two 3x3 convolutions with BN + ReLU"""
        x = F.relu(self.bn_conv2a(self.conv2a(x)))
        x = F.relu(self.bn_conv2b(self.conv2b(x)))
        return x

    def downsample_block1(self, x):
        """Downsample block 1: Conv -> MaxPool -> Dropout"""
        f = self.double_conv_block1(x)
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p

    def downsample_block2(self, x):
        """Downsample block 2: Conv -> MaxPool -> Dropout"""
        f = self.double_conv_block2(x)
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p

    def downsample_block3(self, x):
        """Downsample block 3: Conv -> MaxPool -> Dropout"""
        f = self.double_conv_block3(x)
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p

    def downsample_block4(self, x):
        """Downsample block 4: Conv -> MaxPool -> Dropout"""
        f = self.double_conv_block4(x)
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p

    def upsample_block1(self, x, conv_features):
        """Upsample block 1: Transpose conv -> Concatenate -> Dropout -> Conv"""
        # Upsample
        x = self.dec1(x)
        # Pad to match dimensions if needed
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        # Concatenate with skip connection
        x = torch.cat([x, conv_features], dim=1)
        # Dropout
        x = self.dropout(x)
        # Double convolution
        x = self.up_double_conv_block1(x)
        return x

    def upsample_block2(self, x, conv_features):
        """Upsample block 2: Transpose conv -> Concatenate -> Dropout -> Conv"""
        x = self.dec2(x)
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv_features], dim=1)
        x = self.dropout(x)
        x = self.up_double_conv_block2(x)
        return x

    def upsample_block3(self, x, conv_features):
        """Upsample block 3: Transpose conv -> Concatenate -> Dropout -> Conv"""
        x = self.dec3(x)
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv_features], dim=1)
        x = self.dropout(x)
        x = self.up_double_conv_block3(x)
        return x

    def upsample_block4(self, x, conv_features):
        """Upsample block 4: Transpose conv -> Concatenate -> Dropout -> Conv"""
        x = self.dec4(x)
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv_features], dim=1)
        x = self.dropout(x)
        x = self.up_double_conv_block4(x)
        return x

    def forward(self, x, mag_id=None):
        """
        Forward pass through U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            mag_id (torch.Tensor, optional): Magnification category IDs [B].
                Required when FiLM is enabled. Ignored when FiLM is disabled.

        Returns:
            torch.Tensor: Output tensor of shape [B, num_classes, H, W]
        """
        # Compute FiLM conditioning embedding
        cond = None
        if self.film_enabled and mag_id is not None:
            cond = self.mag_embedding(mag_id)  # [B, embedding_dim]

        # Encoder: contracting path
        # FiLM is applied after conv but before pool, so both the skip
        # connection (f) and pooled output (p) use modulated features.
        f1 = self.double_conv_block1(x)
        if cond is not None:
            f1 = self.film1(f1, cond)
        p1 = self.dropout(self.max_pool(f1))

        f2 = self.double_conv_block2(p1)
        if cond is not None:
            f2 = self.film2(f2, cond)
        p2 = self.dropout(self.max_pool(f2))

        f3 = self.double_conv_block3(p2)
        if cond is not None:
            f3 = self.film3(f3, cond)
        p3 = self.dropout(self.max_pool(f3))

        f4 = self.double_conv_block4(p3)
        if cond is not None:
            f4 = self.film4(f4, cond)
        p4 = self.dropout(self.max_pool(f4))

        # Bottleneck
        bottleneck = self.double_conv_block5(p4)
        if cond is not None:
            bottleneck = self.film5(bottleneck, cond)

        # Decoder: expanding path - upsample with skip connections
        u6 = self.upsample_block1(bottleneck, f4)
        u7 = self.upsample_block2(u6, f3)
        u8 = self.upsample_block3(u7, f2)
        u9 = self.upsample_block4(u8, f1)

        # Output layer
        outputs = self.out(u9)

        return outputs

    def count_parameters(self):
        """Count total trainable parameters (skips uninitialized LazyModule params)."""
        total = 0
        for p in self.parameters():
            if p.requires_grad and not isinstance(p, torch.nn.parameter.UninitializedParameter):
                total += p.numel()
        return total


def create_unet(in_channels=1, num_classes=6, dropout=0.3,
                num_magnifications=None, film_embedding_dim=32):
    """
    Factory function to create a U-Net model.

    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
        num_magnifications (int or None): Number of magnification categories for FiLM
        film_embedding_dim (int): Dimension of magnification embedding

    Returns:
        UNet: Initialized U-Net model
    """
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        num_magnifications=num_magnifications,
        film_embedding_dim=film_embedding_dim,
    )
    return model

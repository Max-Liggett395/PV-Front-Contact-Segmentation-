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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    U-Net model for semantic segmentation of SEM images.

    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        num_classes (int): Number of output classes (default: 6)
        dropout (float): Dropout probability (default: 0.3)
    """

    def __init__(self, in_channels=1, num_classes=6, dropout=0.3):
        super(UNet, self).__init__()

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

    def forward(self, x):
        """
        Forward pass through U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Output tensor of shape [B, num_classes, H, W]
        """
        # Encoder: contracting path - downsample
        f1, p1 = self.downsample_block1(x)
        f2, p2 = self.downsample_block2(p1)
        f3, p3 = self.downsample_block3(p2)
        f4, p4 = self.downsample_block4(p3)

        # Bottleneck
        bottleneck = self.double_conv_block5(p4)

        # Decoder: expanding path - upsample with skip connections
        u6 = self.upsample_block1(bottleneck, f4)
        u7 = self.upsample_block2(u6, f3)
        u8 = self.upsample_block3(u7, f2)
        u9 = self.upsample_block4(u8, f1)

        # Output layer
        outputs = self.out(u9)

        return outputs

    def count_parameters(self):
        """Count total trainable parameters"""
        total = 0
        for p in self.parameters():
            if p.requires_grad:
                # Check if parameter is initialized (for LazyModules)
                if hasattr(p, 'is_materialized'):
                    if p.is_materialized():
                        total += p.numel()
                else:
                    total += p.numel()
        return total


def create_unet(in_channels=1, num_classes=6, dropout=0.3):
    """
    Factory function to create a U-Net model.

    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        dropout (float): Dropout probability

    Returns:
        UNet: Initialized U-Net model
    """
    model = UNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    return model

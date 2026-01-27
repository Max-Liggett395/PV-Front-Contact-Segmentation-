"""
DeepLabV3 and DeepLabV3+ Model Implementations for SEM Segmentation

These models use torchvision's pre-trained DeepLabV3 architecture
adapted for grayscale SEM images and 6-class segmentation.
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
)


class DeepLabV3(nn.Module):
    """
    DeepLabV3 with ResNet backbone for SEM image segmentation.

    Adapts the pre-trained model for:
    - Grayscale input (1 channel) or RGB (3 channels)
    - Custom number of output classes
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize DeepLabV3 model.

        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of segmentation classes
            backbone: Backbone architecture ("resnet50" or "resnet101")
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate for the classifier
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load pre-trained DeepLabV3
        if backbone == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet50(weights=weights)
        elif backbone == "resnet101":
            weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet101(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'resnet50' or 'resnet101'")

        # Modify first conv layer if using grayscale input
        if in_channels != 3:
            old_conv = self.model.backbone.conv1
            self.model.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # Initialize new conv layer (average the pretrained weights across channels)
            if pretrained:
                with torch.no_grad():
                    self.model.backbone.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                    if old_conv.bias is not None:
                        self.model.backbone.conv1.bias.data = old_conv.bias.data

        # Modify classifier for custom number of classes
        # DeepLabV3 classifier structure: Conv2d -> BN -> ReLU -> Dropout -> Conv2d
        in_features = self.model.classifier[4].in_channels
        self.model.classifier[3] = nn.Dropout(dropout)
        self.model.classifier[4] = nn.Conv2d(in_features, num_classes, kernel_size=1)

        # Also modify auxiliary classifier if present
        if self.model.aux_classifier is not None:
            aux_in_features = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[3] = nn.Dropout(dropout)
            self.model.aux_classifier[4] = nn.Conv2d(aux_in_features, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        result = self.model(x)
        return result['out']


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with enhanced decoder for SEM image segmentation.

    DeepLabV3+ adds a simple yet effective decoder module to refine
    segmentation results, especially along object boundaries.

    This implementation adds a decoder that fuses low-level features
    with the ASPP output.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.1,
        low_level_channels: int = 48
    ):
        """
        Initialize DeepLabV3+ model.

        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of segmentation classes
            backbone: Backbone architecture ("resnet50" or "resnet101")
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate for the classifier
            low_level_channels: Number of channels for low-level features
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load pre-trained DeepLabV3
        if backbone == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet50(weights=weights)
            low_level_in_channels = 256  # From layer1 output
        elif backbone == "resnet101":
            weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet101(weights=weights)
            low_level_in_channels = 256  # From layer1 output
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Modify first conv layer if using grayscale input
        if in_channels != 3:
            old_conv = self.model.backbone.conv1
            self.model.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            if pretrained:
                with torch.no_grad():
                    self.model.backbone.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                    if old_conv.bias is not None:
                        self.model.backbone.conv1.bias.data = old_conv.bias.data

        # Low-level feature projection (from early backbone layer)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_in_channels, low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True)
        )

        # Get ASPP output channels (256 from standard DeepLabV3)
        aspp_out_channels = 256

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_out_channels + low_level_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        # Store backbone and classifier separately for forward pass
        self.backbone = self.model.backbone
        self.aspp = self.model.classifier[0]  # ASPP module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with decoder fusion.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        input_shape = x.shape[-2:]

        # Extract backbone features
        features = self.backbone(x)

        # Get low-level features (from layer1, stride 4)
        # We need to hook into intermediate features
        # For simplicity, re-run partial backbone
        x_low = self.backbone.conv1(x)
        x_low = self.backbone.bn1(x_low)
        x_low = self.backbone.relu(x_low)
        x_low = self.backbone.maxpool(x_low)
        low_level_features = self.backbone.layer1(x_low)  # stride 4

        # Get high-level features from ASPP
        high_level_features = features['out']
        aspp_output = self.aspp(high_level_features)

        # Upsample ASPP output to match low-level features
        aspp_output = nn.functional.interpolate(
            aspp_output,
            size=low_level_features.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # Project low-level features
        low_level_features = self.low_level_conv(low_level_features)

        # Concatenate and decode
        concat_features = torch.cat([aspp_output, low_level_features], dim=1)
        decoder_output = self.decoder(concat_features)

        # Upsample to input resolution
        output = nn.functional.interpolate(
            decoder_output,
            size=input_shape,
            mode='bilinear',
            align_corners=False
        )

        return output


def create_deeplabv3(
    in_channels: int = 1,
    num_classes: int = 6,
    backbone: str = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.1
) -> DeepLabV3:
    """Factory function to create DeepLabV3 model."""
    return DeepLabV3(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout
    )


def create_deeplabv3plus(
    in_channels: int = 1,
    num_classes: int = 6,
    backbone: str = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.1,
    low_level_channels: int = 48
) -> DeepLabV3Plus:
    """Factory function to create DeepLabV3+ model."""
    return DeepLabV3Plus(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        low_level_channels=low_level_channels
    )

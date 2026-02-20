"""Model factory for creating segmentation architectures."""

import torch.nn as nn
import torchvision.models.segmentation as segmentation

from .unet import UNet


def create_model(cfg):
    """Create a segmentation model from config.

    For 'unet': custom U-Net matching the original notebook.
    For 'deeplabv3': torchvision deeplabv3_resnet50 (no pretrained weights).
    For 'deeplabv3plus': torchvision deeplabv3_resnet101 (no pretrained weights).
    """
    arch = cfg["architecture"].lower()
    in_channels = cfg.get("in_channels", 1)
    num_classes = cfg.get("num_classes", 6)

    if arch == "unet":
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=cfg.get("dropout", 0.3),
        )
    elif arch == "deeplabv3":
        model = segmentation.deeplabv3_resnet50(weights=None)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier = None
        return model
    elif arch == "deeplabv3plus":
        model = segmentation.deeplabv3_resnet101(weights=None)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier = None
        return model
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from: unet, deeplabv3, deeplabv3plus")

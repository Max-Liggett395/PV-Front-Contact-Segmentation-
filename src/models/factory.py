"""Model factory for creating segmentation architectures."""

import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation

from .unet import UNet


def _adapt_deeplab_first_conv(model, in_channels):
    """Modify the backbone's first conv to accept in_channels != 3.

    When in_channels == 3 (grayscale stacked to 3-ch) no change is needed.
    When in_channels == 1, average the pretrained 3-channel weights into 1 channel.
    """
    if in_channels == 3:
        return

    first_conv = model.backbone.conv1
    old_weight = first_conv.weight.data  # (out_ch, 3, kH, kW)
    new_weight = old_weight.mean(dim=1, keepdim=True)  # (out_ch, 1, kH, kW)

    new_conv = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    new_conv.weight = nn.Parameter(new_weight)
    if first_conv.bias is not None:
        new_conv.bias = nn.Parameter(first_conv.bias.data.clone())

    model.backbone.conv1 = new_conv


def _create_deeplab(arch, cfg):
    """Create a torchvision DeepLab model, optionally with COCO pretrained weights."""
    in_channels = cfg.get("in_channels", 1)
    num_classes = cfg.get("num_classes", 6)
    pretrained = cfg.get("pretrained", False)

    if arch == "deeplabv3":
        if pretrained:
            weights = segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        else:
            weights = None
        model = segmentation.deeplabv3_resnet50(weights=weights)
    else:  # deeplabv3plus
        if pretrained:
            weights = segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        else:
            weights = None
        model = segmentation.deeplabv3_resnet101(weights=weights)

    # Replace the final classifier head to match num_classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier = None

    # Adapt first conv for non-3-channel input after loading pretrained weights
    if pretrained:
        _adapt_deeplab_first_conv(model, in_channels)

    return model


def _create_smp_model(arch, cfg):
    """Create a segmentation_models_pytorch model from config.

    Config fields used:
        encoder_name    (str)  e.g. "efficientnet-b4"
        encoder_weights (str | None)  e.g. "imagenet" or null
        in_channels     (int)  e.g. 3
        num_classes     (int)  e.g. 6
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required for smp_* architectures. "
            "Install it with: pip install segmentation-models-pytorch"
        ) from exc

    encoder_name = cfg.get("encoder_name", "resnet34")
    encoder_weights = cfg.get("encoder_weights", "imagenet") or None
    in_channels = cfg.get("in_channels", 3)
    num_classes = cfg.get("num_classes", 6)

    smp_kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )

    if arch == "smp_unetpp":
        return smp.UnetPlusPlus(**smp_kwargs)
    elif arch == "smp_unet":
        return smp.Unet(**smp_kwargs)
    elif arch == "smp_deeplabv3plus":
        return smp.DeepLabV3Plus(**smp_kwargs)
    else:
        raise ValueError(f"Unknown SMP architecture: '{arch}'")


def create_model(cfg):
    """Create a segmentation model from config.

    Supported architectures:
        unet             - custom U-Net matching the original notebook
        deeplabv3        - torchvision deeplabv3_resnet50; set pretrained=true for COCO weights
        deeplabv3plus    - torchvision deeplabv3_resnet101; set pretrained=true for COCO weights
        smp_unet         - smp.Unet with configurable encoder
        smp_unetpp       - smp.UnetPlusPlus with configurable encoder
        smp_deeplabv3plus - smp.DeepLabV3Plus with configurable encoder

    Config keys:
        architecture    (str, required)
        in_channels     (int, default 1)
        num_classes     (int, default 6)
        pretrained      (bool, default False)  -- torchvision DeepLab only
        dropout         (float, default 0.3)   -- UNet only
        encoder_name    (str)                  -- SMP models only
        encoder_weights (str | None)           -- SMP models only
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
    elif arch in ("deeplabv3", "deeplabv3plus"):
        return _create_deeplab(arch, cfg)
    elif arch in ("smp_unet", "smp_unetpp", "smp_deeplabv3plus"):
        return _create_smp_model(arch, cfg)
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            "Choose from: unet, deeplabv3, deeplabv3plus, "
            "smp_unet, smp_unetpp, smp_deeplabv3plus"
        )

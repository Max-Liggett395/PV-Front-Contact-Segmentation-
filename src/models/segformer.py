"""
SegFormer Model Implementation for SEM Segmentation

Hierarchical Vision Transformer with lightweight MLP decoder,
adapted for grayscale SEM images and 6-class segmentation.
Uses HuggingFace pretrained NVIDIA weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


VARIANT_MAP = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "b5": "nvidia/segformer-b5-finetuned-ade-512-512",
}


class SegFormer(nn.Module):
    """
    SegFormer with hierarchical Transformer encoder and MLP decoder
    for SEM image segmentation.

    Adapts the pre-trained model for:
    - Grayscale input (1 channel) or RGB (3 channels)
    - Custom number of output classes
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        variant: str = "b2",
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        if variant not in VARIANT_MAP:
            raise ValueError(f"Unknown variant: {variant}. Supported: {list(VARIANT_MAP.keys())}")

        model_id = VARIANT_MAP[variant]

        config = SegformerConfig.from_pretrained(
            model_id,
            num_labels=num_classes,
            classifier_dropout_prob=dropout,
        )

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id, config=config, ignore_mismatched_sizes=True,
            )
        else:
            self.model = SegformerForSemanticSegmentation(config)

        # Adapt first conv for grayscale input
        if in_channels != 3:
            old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                    if old_conv.bias is not None:
                        new_conv.bias.data = old_conv.bias.data
            self.model.segformer.encoder.patch_embeddings[0].proj = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

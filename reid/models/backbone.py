"""Backbone feature extractors for vehicle ReID.

The backbone returns a spatial feature map ``(N, C, H', W')``; global pooling is
the head's job (see :mod:`reid.models.head`). The torchvision classifier is
always truncated so ImageNet logits never leak into the embedding.

Supported (see configs/*.yaml):
  * ``convnext_tiny`` -> C = 768   (torchvision IMAGENET1K_V1)
  * ``resnet50``      -> C = 2048  (torchvision IMAGENET1K_V1)
  * ``dinov2_b``      -> C = 768   (timm ``vit_base_patch14_dinov2.lvd142m``)

DINOv2 is a self-supervised ViT; we reshape its patch tokens into a
``(N, C, H/14, W/14)`` map so the same GeM + BNNeck + ArcFace head is reused
unchanged. ``dynamic_img_size=True`` lets it run at any /14 resolution.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

_DINOV2 = "vit_base_patch14_dinov2.lvd142m"
_DINOV2_MODELS = {
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_l": "vit_large_patch14_reg4_dinov2.lvd142m",
}

_BACKBONES = {
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
}


class Backbone(nn.Module):
    """Truncated backbone returning a spatial feature map."""

    def __init__(self, name: str = "convnext_tiny", pretrained: bool = True,
                 image_size: int = 224):
        super().__init__()
        self.name = name
        self.image_size = int(image_size)

        if name in _DINOV2_MODELS:
            self.features = _DINOv2Features(model_name=_DINOV2_MODELS[name],
                                            pretrained=pretrained,
                                            image_size=self.image_size)
            self.out_dim = self.features.out_dim
            return

        if name not in _BACKBONES:
            raise ValueError(
                f"unknown backbone {name!r}; expected one of "
                f"{sorted(_BACKBONES) + sorted(_DINOV2_MODELS)}"
            )
        factory, weights, out_dim = _BACKBONES[name]
        self.out_dim = out_dim

        model = factory(weights=weights if pretrained else None)

        if name.startswith("convnext"):
            self.features = model.features
        elif name.startswith("resnet"):
            self.features = nn.Sequential(*list(model.children())[:-2])
        else:  # pragma: no cover - defensive
            raise ValueError(name)

    def forward(self, x):
        return self.features(x)


class _DINOv2Features(nn.Module):
    """timm DINOv2 ViT -> spatial feature map (patch tokens reshaped)."""

    def __init__(self, model_name: str = _DINOV2, pretrained: bool = True,
                 image_size: int = 224):
        super().__init__()
        import timm  # local import: convnext path must work without timm

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=int(image_size),
            dynamic_img_size=True,
        )
        self.out_dim = int(self.model.embed_dim)
        self.prefix_tokens = int(self.model.num_prefix_tokens)

    def forward(self, x):
        tokens = self.model.forward_features(x)      # (N, 1+L, C)
        tokens = tokens[:, self.prefix_tokens:, :]   # drop CLS (and registers)
        n, l, c = tokens.shape
        gh, gw = self.model.patch_embed.grid_size
        if gh * gw != l:  # pragma: no cover - dynamic grid fallback
            gh = gw = int(round(l ** 0.5))
        return tokens.transpose(1, 2).reshape(n, c, gh, gw).contiguous()


def build_backbone(name: str = "convnext_tiny", pretrained: bool = True,
                   image_size: int = 224) -> Backbone:
    return Backbone(name=name, pretrained=pretrained, image_size=image_size)

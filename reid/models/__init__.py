"""Vehicle ReID model: backbone -> GeM -> BNNeck -> ArcFace.

No ``camera_id`` ever enters the network (red line, see AGENTS.md): camera is
used only for the validation split / junk filtering.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from reid.models.backbone import Backbone, build_backbone
from reid.models.head import ArcFace, BNNeck, GeM

__all__ = [
    "Backbone",
    "build_backbone",
    "GeM",
    "BNNeck",
    "ArcFace",
    "ReIDModel",
    "build_model",
]


class ReIDModel(nn.Module):
    """End-to-end ReID network.

    ``emb_dim`` is the embedding dimension (BNNeck output). ``features`` (the
    BNNeck output) are L2-normalized for retrieval.
    """

    def __init__(self, backbone: str = "convnext_tiny", num_classes: int = 1,
                 emb_dim: int = 512, pretrained: bool = True,
                 margin: float = 0.3, scale: float = 30.0, gem_p: float = 3.0,
                 image_size: int = 224):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained=pretrained,
                                       image_size=image_size)
        self.gem = GeM(p=gem_p)
        self.neck = nn.Sequential(
            nn.Linear(self.backbone.out_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.PReLU(),
        )
        self.bnneck = BNNeck(emb_dim)
        self.arcface = ArcFace(emb_dim, num_classes, margin=margin, scale=scale)
        self.emb_dim = emb_dim

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        feat = self.gem(feat)
        feat = self.neck(feat)
        emb = self.bnneck(feat)
        if labels is not None:
            logits = self.arcface(emb, labels)
            return logits, emb
        return emb

    @property
    def num_classes(self):
        return self.arcface.num_classes

    def embed(self, x):
        """L2-normalized embedding for retrieval (call in ``eval()`` mode)."""
        return F.normalize(self.forward(x), dim=1)


def build_model(backbone: str = "convnext_tiny", num_classes: int = 1,
                emb_dim: int = 512, pretrained: bool = True,
                margin: float = 0.3, scale: float = 30.0,
                gem_p: float = 3.0, image_size: int = 224) -> ReIDModel:
    return ReIDModel(
        backbone=backbone,
        num_classes=num_classes,
        emb_dim=emb_dim,
        pretrained=pretrained,
        margin=margin,
        scale=scale,
        gem_p=gem_p,
        image_size=image_size,
    )

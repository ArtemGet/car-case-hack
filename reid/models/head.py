"""ReID head components: GeM pooling, BNNeck, ArcFace.

Pipeline: spatial feature map -> GeM -> (Linear+BN+PReLU) -> BNNeck -> ArcFace.
GeM and BNNeck are the discriminative-ReID standard (Luo et al., 2019/2020);
ArcFace supplies the angular-margin objective (Deng et al., 2019).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized-mean pooling over the spatial dims (learnable exponent)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        self.eps = eps
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
            self._learnable = True
        else:
            self.p = p
            self._learnable = False

    def forward(self, x):
        p = self.p if self._learnable else self.p
        # Pool in fp32: GeM's power/mean is precision-sensitive under bf16 AMP.
        x = x.float().clamp(min=self.eps).pow(p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1))).pow(1.0 / p)
        return x.flatten(1)


class BNNeck(nn.Module):
    """BatchNorm producing the metric embedding (features before it feed the loss)."""

    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        return self.bn(x)


class ArcFace(nn.Module):
    """Additive angular-margin softmax.

    Training (``labels`` given): returns margin-adjusted ``cosine * scale``.
    Inference (``labels=None``): returns plain ``cosine * scale``.
    """

    def __init__(self, in_features: int, num_classes: int, margin: float = 0.3,
                 scale: float = 30.0):
        super().__init__()
        if not 0.0 <= margin < math.pi:
            raise ValueError("margin must be in [0, pi)")
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.margin = float(margin)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, features, labels=None):
        # Cosine geometry is computed in fp32: bf16 underflows 1-cos^2 near 1
        # and would corrupt the angular margin.
        cosine = F.linear(F.normalize(features.float(), dim=1),
                          F.normalize(self.weight.float(), dim=1))
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        # Guard the monotonicity region for large margins.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale

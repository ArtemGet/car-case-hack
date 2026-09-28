"""fast-reid ResNet50-IBN (SBS, VeRi-776) reconstructor for EX-2.

The published checkpoint ``veri_sbs_R50-ibn.pth`` (ckevar/fastreid_models, public,
no auth) is a fast-reid "Strong Baseline" ResNet50-IBN with Non-local blocks,
trained on VeRi-776 (575 train IDs). We rebuild the EXACT architecture from the
state_dict (verified by a strict load) so it can be used as a frozen feature
extractor -- same val split / aspect-preserving crop as every other candidate.

Architecture (matches the checkpoint key names):
  stem conv1(7x7,s2)/bn1/relu/maxpool
  layer1..layer4 = ResNet50 bottlenecks, every bn1 is an IBN (IN||BN on halves)
  Non-local blocks: 2 after layer2 (512d), 3 after layer3 (1024d)
  head: GeM pool -> BatchNorm1d(2048) [BNNeck] -> 2048-d retrieval feature

Red lines: no camera_id input, no test data, per-query ranking.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IBN(nn.Module):
    """InstanceNorm || BatchNorm on the two channel halves (fast-reid style)."""

    def __init__(self, planes: int):
        super().__init__()
        half = planes // 2
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(planes - half)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        return torch.cat([self.IN(x1), self.BN(x2)], dim=1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, ibn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class NonLocal(nn.Module):
    """fast-reid Non-local block (g/theta/phi 1x1 convs -> scalar similarity)."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.g = nn.Conv2d(in_channels, 1, 1)
        self.theta = nn.Conv2d(in_channels, 1, 1)
        self.phi = nn.Conv2d(in_channels, 1, 1)
        self.W = nn.Sequential(
            nn.Conv2d(1, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        g_x = self.g(x).view(b, 1, n).permute(0, 2, 1)     # (b,N,1)
        theta_x = self.theta(x).view(b, 1, n).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, 1, n)                  # (b,1,N)
        f = torch.matmul(theta_x, phi_x)                   # (b,N,N)
        f_div_c = f / n                                    # fast-reid: f/N, not softmax
        y = torch.matmul(f_div_c, g_x)                     # (b,N,1)
        y = y.permute(0, 2, 1).contiguous().view(b, 1, h, w)
        y = self.W(y)
        return y + x


class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=1e-6).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class FastReIDResNet50IBN(nn.Module):
    def __init__(self, non_layers=(0, 2, 3, 0)):
        super().__init__()
        # stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # fast-reid: MaxPool2d(3, stride=2, ceil_mode=True), NO padding
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # layers (IBN in layers 1-3 only; layer4 uses plain BN)
        self.layer1 = self._make_layer(64, 3, stride=1, ibn=True)
        self.layer2 = self._make_layer(128, 4, stride=2, ibn=True)
        self.NL_2 = nn.ModuleList([NonLocal(512) for _ in range(non_layers[1])])
        self.layer3 = self._make_layer(256, 6, stride=2, ibn=True)
        self.NL_3 = nn.ModuleList([NonLocal(1024) for _ in range(non_layers[2])])
        self.layer4 = self._make_layer(512, 3, stride=2, ibn=False)
        # non-local positions: last `non` blocks of each stage (fast-reid indexing)
        self.NL_2_idx = sorted([4 - (i + 1) for i in range(non_layers[1])])
        self.NL_3_idx = sorted([6 - (i + 1) for i in range(non_layers[2])])

    def _make_layer(self, planes, blocks, stride, ibn):
        downsample = None
        inplanes = 64 if planes == 64 else planes * 2
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(inplanes, planes, stride, downsample, ibn=ibn)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(planes * 4, planes, ibn=ibn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self._forward_stage(self.layer1, None, None, x)
        x = self._forward_stage(self.layer2, self.NL_2, self.NL_2_idx, x)
        x = self._forward_stage(self.layer3, self.NL_3, self.NL_3_idx, x)
        x = self._forward_stage(self.layer4, None, None, x)
        return x

    @staticmethod
    def _forward_stage(stage, nl, nl_idx, x):
        counter = 0
        for i in range(len(stage)):
            x = stage[i](x)
            if nl is not None and counter < len(nl) and i == nl_idx[counter]:
                x = nl[counter](x)
                counter += 1
        return x


class FastReIDVeriExtractor(nn.Module):
    """Full model: backbone + GeM + BNNeck; returns 2048-d retrieval feature."""

    def __init__(self):
        super().__init__()
        self.backbone = FastReIDResNet50IBN()
        self.pool_layer = GeM()
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool_layer(f)
        f = f.flatten(1)
        return self.bottleneck(f)

    @classmethod
    def from_checkpoint(cls, path, device):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        net = cls()
        bb, heads = {}, {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                bb[k[len("backbone."):]] = v
            elif k.startswith("heads."):
                heads[k[len("heads."):]] = v
        net.backbone.load_state_dict(bb, strict=True)
        net.pool_layer.load_state_dict(
            {"p": heads["pool_layer.p"]}, strict=True)
        bn_sd = {
            "weight": heads["bottleneck.0.weight"],
            "bias": heads["bottleneck.0.bias"],
            "running_mean": heads["bottleneck.0.running_mean"],
            "running_var": heads["bottleneck.0.running_var"],
        }
        if "bottleneck.0.num_batches_tracked" in heads:
            bn_sd["num_batches_tracked"] = heads["bottleneck.0.num_batches_tracked"]
        net.bottleneck.load_state_dict(bn_sd, strict=True)
        net.eval().to(device)
        return net

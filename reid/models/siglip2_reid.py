"""Trainable wrapper around the external SigLIP2 NaFlex vehicle-ReID backbone.

The public release (occurra/vehicle_reid_siglip2_naflex_512d) ships a torch
bundle with a HuggingFace ``Siglip2VisionModel`` state dict in ``backbone``, a
per-dimension feature "neck" affine (``neck_scale``/``neck_shift``) applied to
the attention-pooled feature, and a 512-d projection in
``proj_weight``/``proj_bias``. We rebuild that backbone with transformers, load
the weights, and add our own BNNeck + ArcFace head so the model can be
fine-tuned end-to-end on the train split — the "external vehicle ReID pretrain"
lever allowed by the brief (public weights, URL + sha256 fixed in
``runs/<exp>/external_sources.json``).

IMPORTANT (fix): the released neck must be applied *before* ``proj``:
``feat = proj(pooler * neck_scale + neck_shift)``. Omitting it (the earlier
exp-0034 code) fed ``proj`` features ~4-8x too small and already cost ~0.13
mAP@10 vs the frozen ONNX, which masked any fine-tuning gain.

Partial fine-tuning helpers (for the W4.5 in-domain lever):
    * ``configure_trainable(n_unfreeze, lora_rank)`` freezes the backbone,
      re-enables the last ``n_unfreeze`` transformer blocks plus the neck /
      adapter / proj / head, and optionally injects LoRA into q/v projections.
    * ``Adapter`` is a zero-initialised residual bottleneck (identity at start).

NaFlex interface (same as the shipped ONNX):
    pixel_values          (B, 256, 768)  patch sequence, padded
    pixel_attention_mask  (B, 256)       int64, 1 = real patch
    spatial_shapes        (B, 2)         int64, per-image (rows, cols)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.models.head import ArcFace, BNNeck

PATCH = 16
MAX_PATCHES = 256


def patchify_batch(imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B,3,R*16,C*16) -> padded (B,256,768) + mask + spatial_shapes.

    The batch is assumed to come from a fixed square grid so every image has the
    same (rows, cols); callers that need mixed aspect ratios must patchify per
    image. Pixels must already be scaled to [-1, 1] (mean=std=0.5).
    """
    b, c, h, w = imgs.shape
    rows, cols = h // PATCH, w // PATCH
    n = rows * cols
    if n > MAX_PATCHES:
        raise ValueError(f"too many patches: {n} > {MAX_PATCHES}")
    patches = imgs.reshape(b, c, rows, PATCH, cols, PATCH)
    patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(b, n, PATCH * PATCH * c)
    pv = torch.zeros(b, MAX_PATCHES, PATCH * PATCH * c, dtype=imgs.dtype)
    pv[:, :n] = patches
    mask = torch.zeros(b, MAX_PATCHES, dtype=torch.long)
    mask[:, :n] = 1
    shapes = torch.tensor([[rows, cols]], dtype=torch.long).repeat(b, 1)
    return pv, mask, shapes


class Adapter(nn.Module):
    """Zero-initialised residual bottleneck (identity at init)."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank update (B init 0)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha if alpha is not None else rank) / max(1, self.rank)
        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(self.rank, base.in_features,
                                               device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank,
                                               device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        lora = (x @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * lora


class SigLIP2ReID(nn.Module):
    """SigLIP2 vision backbone + released neck + adapter + proj + BNNeck + ArcFace."""

    def __init__(self, backbone, emb_dim: int = 512, num_classes: int = 1,
                 margin: float = 0.3, scale: float = 30.0,
                 proj_weight=None, proj_bias=None, neck_scale=None, neck_shift=None,
                 adapter_hidden: int = 0):
        super().__init__()
        self.backbone = backbone
        feat = int(backbone.config.hidden_size)
        hidden = feat if adapter_hidden <= 0 else int(adapter_hidden)
        self.neck_scale = nn.Parameter(torch.ones(feat))
        self.neck_shift = nn.Parameter(torch.zeros(feat))
        if neck_scale is not None:
            with torch.no_grad():
                self.neck_scale.copy_(neck_scale.float())
        if neck_shift is not None:
            with torch.no_grad():
                self.neck_shift.copy_(neck_shift.float())
        self.adapter = Adapter(feat, hidden) if adapter_hidden > 0 else None
        self.proj = nn.Linear(feat, emb_dim)
        if proj_weight is not None:
            with torch.no_grad():
                self.proj.weight.copy_(proj_weight)
                if proj_bias is not None:
                    self.proj.bias.copy_(proj_bias)
        self.bnneck = BNNeck(emb_dim)
        self.arcface = ArcFace(emb_dim, num_classes, margin=margin, scale=scale)
        self.emb_dim = int(emb_dim)

    def _features(self, pixel_values, pixel_attention_mask, spatial_shapes):
        out = self.backbone(pixel_values=pixel_values,
                            pixel_attention_mask=pixel_attention_mask,
                            spatial_shapes=spatial_shapes)
        h = out.pooler_output
        h = h * self.neck_scale + self.neck_shift   # released neck (was missing)
        if self.adapter is not None:
            h = self.adapter(h)
        return self.proj(h)

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes, labels=None):
        emb = self.bnneck(self._features(pixel_values, pixel_attention_mask,
                                         spatial_shapes))
        if labels is not None:
            return self.arcface(emb, labels), emb
        return emb

    @torch.no_grad()
    def embed(self, pixel_values, pixel_attention_mask, spatial_shapes):
        self.eval()
        return F.normalize(self.forward(pixel_values, pixel_attention_mask,
                                        spatial_shapes), dim=1)

    @property
    def num_classes(self):
        return self.arcface.num_classes

    # ------------------------------------------------------------------
    # Partial fine-tuning / LoRA
    # ------------------------------------------------------------------
    def add_lora(self, rank: int = 8, alpha: float | None = None) -> int:
        """Inject LoRA into q/v projections of every block. Returns count."""
        n = 0
        for layer in self.backbone.encoder.layers:
            for name in ("q_proj", "v_proj"):
                base = getattr(layer.self_attn, name)
                if isinstance(base, LoRALinear):
                    continue
                setattr(layer.self_attn, name, LoRALinear(base, rank, alpha))
                n += 1
        return n

    def configure_trainable(self, n_unfreeze: int = 0, lora_rank: int = 0,
                            lora_alpha: float | None = None,
                            train_adapter: bool = True, train_proj: bool = True) -> dict:
        """Freeze everything, then re-enable the selected parts.

        Always trainable: ArcFace, BNNeck, neck_scale/shift, and (optionally)
        adapter / proj. Plus the last ``n_unfreeze`` transformer blocks, plus
        LoRA params when ``lora_rank > 0``.
        """
        for p in self.parameters():
            p.requires_grad_(False)

        layers = list(self.backbone.encoder.layers)
        if n_unfreeze > 0:
            for layer in layers[-int(n_unfreeze):]:
                for p in layer.parameters():
                    p.requires_grad_(True)

        n_lora = 0
        if lora_rank > 0:
            n_lora = self.add_lora(lora_rank, lora_alpha)
            for m in self.modules():
                if isinstance(m, LoRALinear):
                    m.lora_A.requires_grad_(True)
                    m.lora_B.requires_grad_(True)

        for p in (self.neck_scale, self.neck_shift):
            p.requires_grad_(True)
        trainable_mods = [self.bnneck, self.arcface]
        if train_adapter and self.adapter is not None:
            trainable_mods.append(self.adapter)
        if train_proj:
            trainable_mods.append(self.proj)
        for mod in trainable_mods:
            for p in mod.parameters():
                p.requires_grad_(True)

        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        return {"trainable": n_tr, "total": n_all,
                "trainable_pct": 100.0 * n_tr / max(1, n_all),
                "n_unfreeze": n_unfreeze, "n_lora_layers": n_lora}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_siglip2_reid(bundle_path: str, num_classes: int, emb_dim: int = 512,
                       margin: float = 0.3, scale: float = 30.0,
                       image_size: int = 256, pretrained: bool = True,
                       adapter_hidden: int = 0) -> SigLIP2ReID:
    """Rebuild the released backbone from a ``.pth`` bundle and attach our head."""
    from transformers import Siglip2VisionConfig, Siglip2VisionModel

    ck = torch.load(bundle_path, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("config", {}))
    cfg["image_size"] = int(image_size)
    backbone = Siglip2VisionModel(Siglip2VisionConfig(**cfg))
    if pretrained:
        sd = {k.replace("vision_model.", "", 1): v for k, v in ck["backbone"].items()}
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"backbone load mismatch: missing={missing[:3]} "
                               f"unexpected={unexpected[:3]}")
    return SigLIP2ReID(
        backbone, emb_dim=emb_dim, num_classes=num_classes, margin=margin,
        scale=scale, proj_weight=ck.get("proj_weight"), proj_bias=ck.get("proj_bias"),
        neck_scale=ck.get("neck_scale"), neck_shift=ck.get("neck_shift"),
        adapter_hidden=adapter_hidden)

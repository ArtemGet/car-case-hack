"""Train / eval torchvision transforms for vehicle ReID.

Hard rule: NO ``RandomHorizontalFlip`` - vehicles are not left/right symmetric
(steering side, light positions, text and windscreen layout all change).
Augmentations are applied to the aspect-preserving square crop produced by
:func:`reid.data.crop.crop_vehicle`, hence no aspect-distorting Resize here.
"""
from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Deliberately no RandomHorizontalFlip / VerticalFlip.
_TRAIN_OPS = ("ColorJitter", "RandomResizedCrop", "RandomErasing", "GaussianBlur")


def build_train_transform(size=224):
    """Train-time pipeline: geometry + photometry + tensor + erasing."""
    return T.Compose(
        [
            T.RandomResizedCrop(
                size,
                scale=(0.70, 1.0),
                ratio=(0.85, 1.18),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.30, hue=0.05),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5))], p=0.30
            ),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]
    )


def build_eval_transform(size=224):
    """Eval-time pipeline.

    The crop is an aspect-preserving square (see ``crop_vehicle``); it may come
    from the 320-px crop cache, so resize to the backbone input size here (a
    no-op when the crop is already ``size``).
    """
    size = int(size)
    return T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR,
                     antialias=True),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


train_transform = build_train_transform()
eval_transform = build_eval_transform()

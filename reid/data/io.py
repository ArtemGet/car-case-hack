"""Dataset CSV I/O with strict schema validation.

Frozen input contracts (see docs/_workspace/INTERFACES.md):
    train.csv         image_id,x,y,w,h,vehicle_id,camera_id
    test_query.csv    image_id,x,y,w,h
    test_gallery.csv  image_id,x,y,w,h

`x,y,w,h` are integer pixels of the source frame (top-left / width / height).
`image_id` is unique; one frame == one vehicle instance.
"""
from __future__ import annotations

import os

import pandas as pd

TRAIN_COLUMNS = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]
QUERY_COLUMNS = ["image_id", "x", "y", "w", "h"]

BBOX_COLUMNS = ["x", "y", "w", "h"]
ID_COLUMNS = ["vehicle_id", "camera_id"]

_SPLIT_FILES = {
    "train": "train.csv",
    "test_query": "test_query.csv",
    "test_gallery": "test_gallery.csv",
}
_SPLIT_COLUMNS = {
    "train": TRAIN_COLUMNS,
    "test_query": QUERY_COLUMNS,
    "test_gallery": QUERY_COLUMNS,
}


def read_csv(path, required=None):
    """Read a dataset CSV and validate its schema strictly.

    Parameters
    ----------
    path : str
        CSV path.
    required : list[str] | None
        Exact expected column list (order and content). Mismatch raises
        ``ValueError``. When ``None`` only the basic dtype/unique checks run.

    Returns
    -------
    pandas.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    actual = list(df.columns)
    if required is not None and actual != list(required):
        raise ValueError(
            f"schema mismatch in {path}: expected {list(required)}, got {actual}"
        )

    for col in BBOX_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int64")
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int64")
    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str)
        if df["image_id"].duplicated().any():
            dupes = df.loc[df["image_id"].duplicated(), "image_id"].tolist()[:5]
            raise ValueError(f"duplicate image_id in {path}: {dupes}")

    return df


def load_split(dataset_dir, name):
    """Load one named split (``train`` | ``test_query`` | ``test_gallery``)."""
    if name not in _SPLIT_FILES:
        raise ValueError(
            f"unknown split {name!r}; expected one of {sorted(_SPLIT_FILES)}"
        )
    path = os.path.join(dataset_dir, _SPLIT_FILES[name])
    return read_csv(path, required=_SPLIT_COLUMNS[name])


def image_path(dataset_dir, image_id):
    """Resolve the flat images directory path for an ``image_id``."""
    return os.path.join(dataset_dir, "images", f"{image_id}.jpg")

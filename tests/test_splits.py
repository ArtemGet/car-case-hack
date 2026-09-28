"""Tests for reid.data: split integrity, open-set share, crop geometry, I/O."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reid.data.crop import crop_vehicle  # noqa: E402
from reid.data.io import TRAIN_COLUMNS, load_split, read_csv  # noqa: E402
from reid.data.splits import holdout_val, to_gt_csv  # noqa: E402

DATASET_DIR = os.path.join(ROOT, "docs", "Датасет", "dataset")


def synth_train(n_vehicles=60, seed=0):
    """Synthetic train table: each vehicle has >=2 cameras and 4..8 frames."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_vehicles):
        n_cams = int(rng.integers(2, 4))
        n_frames = int(rng.integers(4, 9))
        for i in range(n_frames):
            if i == 0:
                cam = 0
            elif i == 1:
                cam = 1
            else:
                cam = int(rng.integers(0, n_cams))
            rows.append(
                {
                    "image_id": f"v{v}_f{i}",
                    "x": int(rng.integers(0, 50)),
                    "y": int(rng.integers(0, 50)),
                    "w": 60,
                    "h": 50,
                    "vehicle_id": v,
                    "camera_id": 1000 + v * 10 + cam,
                }
            )
    return pd.DataFrame(rows, columns=TRAIN_COLUMNS)


def val_open_mask(query, gallery):
    """True where a val query has NO junk-surviving positive (open-set query)."""
    gal_vid = gallery["vehicle_id"].to_numpy()
    gal_cam = gallery["camera_id"].to_numpy()
    out = []
    for vid, cam in zip(query["vehicle_id"], query["camera_id"]):
        out.append(not np.any((gal_vid == vid) & (gal_cam != cam)))
    return np.array(out, dtype=bool)


def test_no_vehicle_id_leak():
    train, q, g = holdout_val(synth_train(), val_fraction=0.2, seed=42)
    assert set(train["vehicle_id"]).isdisjoint(set(q["vehicle_id"]))
    assert set(train["vehicle_id"]).isdisjoint(set(g["vehicle_id"]))


def test_val_frames_not_lost():
    df = synth_train()
    train, q, g = holdout_val(df, val_fraction=0.2, seed=42)
    assert len(train) + len(q) + len(g) == len(df)
    val_total = len(df) - len(train)
    assert len(q) + len(g) == val_total
    assert set(train["image_id"]).isdisjoint(set(q["image_id"]))
    assert set(q["image_id"]).isdisjoint(set(g["image_id"]))


def test_val_fraction_approx():
    df = synth_train(n_vehicles=100, seed=1)
    train, q, g = holdout_val(df, val_fraction=0.2, seed=42)
    n_val_vids = df.loc[~df["vehicle_id"].isin(train["vehicle_id"]), "vehicle_id"].nunique()
    assert abs(n_val_vids / df["vehicle_id"].nunique() - 0.2) < 0.05


def test_open_set_query_fraction_approx():
    for seed in (0, 1, 2):
        df = synth_train(n_vehicles=120, seed=seed)
        train, q, g = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
        frac = val_open_mask(q, g).mean()
        assert abs(frac - 0.2) < 0.08, f"seed={seed} open query frac={frac:.3f}"


def test_closed_query_has_valid_positive():
    df = synth_train(n_vehicles=120, seed=3)
    train, q, g = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    open_mask = val_open_mask(q, g)
    closed = q[~open_mask]
    assert len(closed) > 0
    gal_by_vid = {v: sub for v, sub in g.groupby("vehicle_id")}
    for _, row in closed.iterrows():
        sub = gal_by_vid.get(row["vehicle_id"])
        assert sub is not None
        valid = (sub["camera_id"] != row["camera_id"]).any()
        assert valid, f"no valid positive for {row['image_id']}"


def test_to_gt_csv_schema():
    df = synth_train()
    train, q, g = holdout_val(df, val_fraction=0.2, seed=42)
    gt = to_gt_csv(train, q, g)
    assert list(gt.columns) == ["image_id", "vehicle_id", "camera_id", "split"]
    assert set(gt["split"]) == {"train", "val_query", "val_gallery"}
    assert len(gt) == len(df)


def test_crop_landscape_square_and_proportions():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (100, 200, 3), dtype=np.uint8))
    out = crop_vehicle(img, 0, 0, 200, 100, target=224)
    assert out.size == (224, 224)
    arr = np.asarray(out)
    # landscape: content is 224 wide x 112 tall -> top/bottom border of 56 rows
    assert np.all(arr[:56, :] == arr[0, 0])
    assert np.all(arr[168:, :] == arr[0, 0])
    # content band is not uniformly background
    assert np.any(arr[56, :] != arr[0, 0])
    assert arr.shape == (224, 224, 3)


def test_crop_portrait_square_and_proportions():
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (200, 100, 3), dtype=np.uint8))
    out = crop_vehicle(img, 0, 0, 100, 200, target=224)
    arr = np.asarray(out)
    assert out.size == (224, 224)
    assert np.all(arr[:, :56] == arr[0, 0])
    assert np.all(arr[:, 168:] == arr[0, 0])
    assert np.any(arr[:, 56] != arr[0, 0])


def test_crop_partial_bbox_keeps_aspect():
    rng = np.random.default_rng(2)
    img = Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8))
    out = crop_vehicle(img, 10, 20, 100, 50, target=224)
    arr = np.asarray(out)
    assert out.size == (224, 224)
    # crop 100x50 -> content 224x112, vertical padding 56 rows top/bottom
    assert np.all(arr[:56, :] == arr[0, 0])
    assert np.all(arr[168:, :] == arr[0, 0])


def test_crop_numpy_input():
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    out = crop_vehicle(arr, 0, 0, 200, 100, target=224)
    assert out.size == (224, 224)


def test_read_csv_strict_columns(tmp_path):
    good = tmp_path / "train.csv"
    good.write_text("image_id,x,y,w,h,vehicle_id,camera_id\na,1,2,3,4,5,6\n", encoding="utf-8")
    df = read_csv(str(good), required=TRAIN_COLUMNS)
    assert list(df.columns) == TRAIN_COLUMNS
    assert df["x"].dtype == np.int64

    bad = tmp_path / "bad.csv"
    bad.write_text("image_id,x,y,vehicle_id\na,1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv(str(bad), required=TRAIN_COLUMNS)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATASET_DIR, "train.csv")),
    reason="dataset not present",
)
def test_load_real_train_split():
    train = load_split(DATASET_DIR, "train")
    assert len(train) == 9556
    assert train["vehicle_id"].nunique() == 1541
    assert list(train.columns) == TRAIN_COLUMNS
    t, q, g = holdout_val(train, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    assert set(t["vehicle_id"]).isdisjoint(set(q["vehicle_id"]))
    assert len(t) + len(q) + len(g) == len(train)
    frac = val_open_mask(q, g).mean()
    assert abs(frac - 0.2) < 0.08

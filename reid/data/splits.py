"""Cross-camera hold-out validation split with an open-set share.

Design (W1-1, see 00_BRIEF.md §4.1/§4.2 and METRICS.md):
  * ``vehicle_id`` are partitioned into train and val - no identity leaks.
  * Within val, whole vehicles are chosen as **open-set**: ALL of their frames
    go to the query set, none to the gallery (no valid positive exists).
  * Every other val vehicle contributes exactly ONE query frame; the remaining
    frames form the gallery. The query camera is the most populated one, so at
    least one gallery frame comes from a DIFFERENT camera and survives the
    official junk filter (same ``vehicle_id`` AND same ``camera_id``).
  * ``open_set_fraction`` targets the share of val **queries** that are open-set
    (the official closed test marks ~20% of queries as pair-less). Whole
    vehicles are selected greedily to approach that share, matching the metric
    the validation harness actually reports.

Junk filter (official): a gallery item is junk for a query iff it shares BOTH
``vehicle_id`` and ``camera_id`` with the query.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from reid.data.io import BBOX_COLUMNS, ID_COLUMNS, TRAIN_COLUMNS

_QUERY_COLUMNS = ["image_id"] + BBOX_COLUMNS + ID_COLUMNS


def _require_train_columns(df):
    missing = [c for c in TRAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"train_df missing columns: {missing}")


def _valid_positive_mask(query, gallery):
    """Return a boolean mask of gallery rows that are valid positives for query.

    Valid == same ``vehicle_id`` AND different ``camera_id`` (i.e. survives the
    junk filter as a true match).
    """
    if gallery.empty:
        return np.zeros(0, dtype=bool)
    same_vid = gallery["vehicle_id"].to_numpy() == query["vehicle_id"]
    diff_cam = gallery["camera_id"].to_numpy() != query["camera_id"]
    return same_vid & diff_cam


def holdout_val(train_df, val_fraction=0.2, open_set_fraction=0.2, seed=42):
    """Split ``train_df`` into ``(train_df, val_query_df, val_gallery_df)``."""
    _require_train_columns(train_df)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if not 0.0 <= open_set_fraction < 1.0:
        raise ValueError("open_set_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)
    vids = np.array(sorted(train_df["vehicle_id"].unique()))
    rng.shuffle(vids)

    n_val = int(round(val_fraction * len(vids)))
    if len(vids) > 1:
        n_val = min(max(n_val, 1), len(vids) - 1)
    val_vids = list(vids[:n_val])

    train_out = train_df[~train_df["vehicle_id"].isin(val_vids)].reset_index(drop=True)
    val_df = train_df[train_df["vehicle_id"].isin(val_vids)].copy()
    groups = {int(v): sub for v, sub in val_df.groupby("vehicle_id")}

    # Greedy selection of whole open-set vehicles to approach the target query
    # share. Closed vehicles each contribute exactly one query.
    open_vids: set = set()
    if open_set_fraction > 0 and n_val > 0:
        order = list(val_vids)
        rng.shuffle(order)
        open_q = 0
        closed_q = n_val
        eps = 0.02
        for v in order:
            v = int(v)
            frames = len(groups[v])
            new_open = open_q + frames
            new_closed = closed_q - 1
            if new_closed <= 0:
                break
            if new_open / (new_open + new_closed) <= open_set_fraction + eps:
                open_vids.add(v)
                open_q = new_open
                closed_q = new_closed
        if not open_vids:  # honor a non-zero request
            open_vids.add(int(order[0]))

    query_parts = []
    gallery_parts = []
    for v in val_vids:
        v = int(v)
        sub = groups[v]
        # A vehicle with a single camera cannot yield a junk-surviving positive;
        # route it to the open-set side instead of a broken positive.
        if v in open_vids or sub["camera_id"].nunique() < 2:
            query_parts.append(sub)
            continue

        counts = sub["camera_id"].value_counts()
        query_cam = int(counts.index[0])
        qsub = sub[sub["camera_id"] == query_cam]
        qrow = qsub.iloc[[0]]
        gsub = sub.drop(qrow.index)
        query_parts.append(qrow)
        if not gsub.empty:
            gallery_parts.append(gsub)

    val_query = (
        pd.concat(query_parts, ignore_index=True)
        if query_parts
        else val_df.iloc[0:0].copy()
    )
    val_gallery = (
        pd.concat(gallery_parts, ignore_index=True)
        if gallery_parts
        else val_df.iloc[0:0].copy()
    )

    val_query = val_query[_QUERY_COLUMNS].reset_index(drop=True)
    val_gallery = val_gallery[_QUERY_COLUMNS].reset_index(drop=True)
    return train_out, val_query, val_gallery


def to_gt_csv(train_df, val_query_df, val_gallery_df, path=None):
    """Concatenate splits into a ground-truth table ``image_id,vehicle_id,camera_id,split``."""
    parts = []
    for df, split in (
        (train_df, "train"),
        (val_query_df, "val_query"),
        (val_gallery_df, "val_gallery"),
    ):
        part = df[["image_id"] + ID_COLUMNS].copy()
        part["split"] = split
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    if path is not None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        out.to_csv(path, index=False)
    return out

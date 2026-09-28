"""Set data-pipeline keys in all training configs (idempotent)."""
from __future__ import annotations

import glob
import os
import re

ROOT = r"C:\car-case"

WANT = {
    "num_workers": "12",
    "draft_factor": "1.2",
    "cache_size": "320",
    "eval_every": "2",
}


def set_key(text: str, key: str, value: str) -> str:
    if re.search(rf"(?m)^{key}\s*:", text):
        return re.sub(rf"(?m)^{key}\s*:.*$", f"{key}: {value}", text)
    return text.rstrip() + f"\n{key}: {value}\n"


for path in glob.glob(os.path.join(ROOT, "configs", "*.yaml")):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for k, v in WANT.items():
        text = set_key(text, k, v)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("updated", os.path.basename(path))

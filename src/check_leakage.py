#!/usr/bin/env python3
"""Quantify near-duplicate overlap across the development/test boundary.

The FFE images were captured by photographing stored fish repeatedly over six
days, so images taken seconds apart within one session, and images of the same
specimen on consecutive days, are near-duplicates that carry the same label. A
split drawn at the image level can place such a pair on opposite sides.

Similarity is measured with an ImageNet-pretrained Swin-Tiny that has never seen
FFE, so the measurement is independent of anything trained here. For every test
image we find its nearest development neighbour and record the cosine similarity
and whether that neighbour comes from the same capture session (same species,
same class, same date).

    python check_leakage.py
"""

from __future__ import annotations

import platform
import re
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader

import core

SEED = 123


def session_of(p: str) -> str:
    """Species, grade, and capture date -- one photography session."""
    folder = Path(p).parent.name
    m = re.search(r"(20\d{6})", Path(p).name)
    return f"{folder}|{m.group(1) if m else 'unknown'}"


@torch.no_grad()
def embed(paths, tf) -> np.ndarray:
    model = timm.create_model(core.MODEL_NAME, pretrained=True, num_classes=0)
    model.eval().to(core.DEVICE)
    ds = core.FishDataset(paths, np.zeros(len(paths), dtype=np.int64), tf)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=core.NUM_WORKERS)
    out = []
    for x, _ in dl:
        f = model(x.to(core.DEVICE))
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.vstack(out)


def main() -> int:
    core.set_seed(SEED)
    dev, test, paths, labels, strata = core.make_split()
    _, eval_tf = core.build_transforms(core.DEFAULT_HP)

    print("=" * 74)
    print("NEAR-DUPLICATE OVERLAP ACROSS THE DEVELOPMENT/TEST BOUNDARY")
    print(f"host      : {socket.gethostname()}  |  python {platform.python_version()}")
    print(f"encoder   : ImageNet-pretrained {core.MODEL_NAME} (never trained on FFE)")
    print(f"seed      : {SEED}   dev={len(dev)}  test={len(test)}")
    print(f"timestamp : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    sessions = np.array([session_of(p) for p in paths])
    print(f"capture sessions in the benchmark: {len(set(sessions))}")
    per_class_dates = Counter()
    for s in set(sessions):
        per_class_dates[s.split("|")[0]] += 1
    print(f"dates per (species, grade): "
          f"{sorted(set(per_class_dates.values()))}  "
          f"-> every grade spans multiple capture days")

    zd = embed(paths[dev], eval_tf)
    zt = embed(paths[test], eval_tf)
    sim = zt @ zd.T                                   # cosine, both normalised
    nn = sim.argmax(1)
    nn_sim = sim[np.arange(len(test)), nn]
    same_session = sessions[test] == sessions[dev][nn]
    same_label = labels[test] == labels[dev][nn]

    print(f"\nnearest development neighbour of each test image")
    print(f"  mean cosine similarity      : {nn_sim.mean():.4f}")
    print(f"  median                      : {np.median(nn_sim):.4f}")
    for thr in (0.95, 0.98, 0.99):
        n = int((nn_sim > thr).sum())
        print(f"  test images with NN > {thr}   : {n:>4} ({n/len(test)*100:5.1f}%)")
    print(f"  NN from the same session    : {int(same_session.sum()):>4} "
          f"({same_session.mean()*100:5.1f}%)")
    print(f"  NN carries the same label   : {int(same_label.sum()):>4} "
          f"({same_label.mean()*100:5.1f}%)")

    # a within-session baseline says what "near-duplicate" looks like on this data
    rng = np.random.default_rng(SEED)
    within = []
    for s in rng.choice(list(set(sessions[dev])), size=min(40, len(set(sessions[dev]))),
                        replace=False):
        idx = np.where(sessions[dev] == s)[0]
        if len(idx) > 1:
            sub = zd[idx] @ zd[idx].T
            np.fill_diagonal(sub, -1)
            within.append(sub.max(1).mean())
    print(f"\n  within-session NN similarity (reference): {np.mean(within):.4f}")
    print(f"  cross-boundary NN similarity            : {nn_sim.mean():.4f}")
    print(f"  -> test images sit {'AS CLOSE AS' if nn_sim.mean() >= np.mean(within) - 0.01 else 'further than'}"
          f" images from the same session")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

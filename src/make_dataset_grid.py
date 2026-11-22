#!/usr/bin/env python3
"""Sample grid of the FFE benchmark, one image per species and freshness grade.

Laid out as two blocks of four species so that every panel and label stays
legible at the width of the journal text block, rather than compressing eight
species into a single row.

    python make_dataset_grid.py
"""

from __future__ import annotations

import platform
import socket
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import core

OUT = Path("outputs/figures")
SEED = 123          # picks which exemplar is shown; no bearing on any result
# printed sizes in inches: the figure is included unscaled, so text stays at 8 pt
IMG, GUT, ROWLAB, HEAD, GAP = 0.86, 0.03, 0.2, 0.3, 0.08
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def main() -> int:
    dev_idx, test_idx, paths, labels, strata = core.make_split()
    species = np.array([Path(p).parent.name.split(" - ")[0] for p in paths])
    names = sorted(set(species))

    print("=" * 70)
    print("FFE SAMPLE GRID")
    print(f"host      : {socket.gethostname()}  |  python {platform.python_version()}")
    print(f"seed      : {SEED} (exemplar choice only)")
    print(f"species   : {len(names)}   classes: {core.NUM_CLASSES}")
    print(f"timestamp : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    blocks = [names[:4], names[4:]]
    k_rows = core.NUM_CLASSES
    block_h = HEAD + k_rows * IMG + (k_rows - 1) * GUT
    fw, fh = ROWLAB + 4 * IMG + 3 * GUT, 2 * block_h + GAP
    fig = plt.figure(figsize=(fw, fh))

    shown = 0
    for bi, block in enumerate(blocks):
        top = bi * (block_h + GAP) + HEAD
        for ci, sp in enumerate(block):
            for k in range(k_rows):
                x, y = ROWLAB + ci * (IMG + GUT), top + k * (IMG + GUT)
                ax = fig.add_axes([x / fw, 1 - (y + IMG) / fh, IMG / fw, IMG / fh])
                pool = np.where((species == sp) & (labels == k))[0]
                img = Image.open(paths[rng.choice(pool)]).convert("RGB").resize((640, 640))
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                shown += 1
                if k == 0:
                    genus, epithet = sp.split()[0], sp.split()[-1].lower()
                    ax.set_title(f"{genus}\n{epithet}", fontsize=8, style="italic", pad=2,
                                 linespacing=1.1)
                if ci == 0:
                    ax.set_ylabel(core.CLASS_NAMES[k], fontsize=8, labelpad=3)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F_dataset_grid.{ext}", bbox_inches="tight", pad_inches=0.02,
                    dpi=400)
    plt.close(fig)
    print(f"{shown} panels ({len(names)} species x {core.NUM_CLASSES} grades) "
          f"-> {OUT}/F_dataset_grid.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate F3 — focal-weight curve for the focal CORN loss.

Run: python figures/make_focal_weight.py
Plots the focal weight (1 - p)^gamma against the probability p assigned to the
correct binary outcome, for gamma in {0, 1, 2.41}. gamma=0 is the unweighted
(threshold-averaged) CORN loss (flat weight 1); gamma=2.41 is the value selected by the search for the final model.
No data or model weights needed — pure matplotlib. Drawn at its printed size and
included unscaled; the legend sits outside the axes, so no text covers a curve.
Outputs: F3_focal_weight.pdf (+ .png) next to this script.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8,
                     "legend.frameon": False, "pdf.fonttype": 42})
p = np.linspace(0.0, 1.0, 400)

GAMMAS = [
    (0.0, "#8c8c8c", "-",  r"$\gamma = 0$ (unweighted)"),
    (1.0, "#1baf7a", "--", r"$\gamma = 1$"),
    (2.41, "#2a78d6", "-",  r"$\gamma = 2.41$ (selected)"),
]

fig, ax = plt.subplots(figsize=(4.1, 2.15), layout="constrained")
for g, color, ls, label in GAMMAS:
    ax.plot(p, (1.0 - p) ** g, ls=ls, color=color, lw=1.8 if g > 2.0 else 1.3, label=label)

ax.set_xlabel(r"Probability of the correct outcome $p_{i,k}$")
ax.set_ylabel(r"Weight $(1-p_{i,k})^{\gamma}$")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.04)
ax.grid(True, color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.legend(loc="outside right center", handlelength=2.0)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"F3_focal_weight.{ext}"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=400, facecolor="white")
    print(f"Saved: {out}")
plt.close(fig)

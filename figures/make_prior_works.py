#!/usr/bin/env python3
"""Generate F12 — accuracy on the FFE benchmark across published studies.

Run: python figures/make_prior_works.py

The upper five bars are accuracies as reported by each study under its own
evaluation protocol, shown for context only. The lower two share one protocol:
the locked species-stratified split of this work, ten training seeds, mean and
standard deviation. Only those two are directly comparable; a gap and a rule
separate them, and the caption says so, so no explanatory text sits in the plot.

  Prasetyo et al. 2022 (MB-BE)          63.21          [prasetyo2022b]
  Yildiz et al. 2024                    77.30          [yildiz2024]
  Hoang et al. 2026b (handcrafted)      77.56          [hoang2026handcrafted]
  P.K. et al. 2026                      81.03          [pk2026llm]
  Hoang et al. 2026a (as published)     85.99          [hoang2026ffe]
  Hoang et al. 2026a (matched re-run)   84.43 +/- 0.91  (RF, ten seeds)
  Proposed model                        90.01 +/- 0.64  (ten seeds)

Drawn at its printed size (text width 4.77 in) and included unscaled.
Outputs: F12_prior_works.pdf (+ .png) next to this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42})

# (label, accuracy, sd or None, colour); the last two share the matched protocol
GREY, HOANG, OURS = "#b4b4b4", "#eda100", "#2a78d6"
ROWS = [
    ("Prasetyo et al. (2022), MB-BE",        63.21, None, GREY),
    ("Yildiz et al. (2024)",                 77.30, None, GREY),
    ("Hoang et al. (2026b), handcrafted",    77.56, None, GREY),
    ("P.K. et al. (2026)",                   81.03, None, GREY),
    ("Hoang et al. (2026a), as published",   85.99, None, GREY),
    ("Hoang et al. (2026a), matched re-run", 84.43, 0.91, HOANG),
    ("Proposed",                             90.01, 0.64, OURS),
]
Y = [0, 1, 2, 3, 4, 5.6, 6.6]          # a gap between the two groups

fig, ax = plt.subplots(figsize=(4.77, 2.2), layout="constrained")
for y, (lab, v, sd, col) in zip(Y, ROWS):
    ax.barh(y, v, height=0.62, color=col, zorder=3)
    if sd is not None:
        ax.errorbar(v, y, xerr=sd, fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5,
                    zorder=4)
    txt = f"{v:.2f}" if sd is None else f"{v:.2f} $\\pm$ {sd:.2f}"
    ax.annotate(txt, (v + (sd or 0), y), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8)
ax.axhline(4.8, color="#8c8c8c", lw=0.7, ls=(0, (2, 2)))
ax.set_yticks(Y)
ax.set_yticklabels([r[0] for r in ROWS])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Test accuracy (%)")
ax.tick_params(axis="y", length=0)
ax.spines[["top", "right"]].set_visible(False)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"F12_prior_works.{ext}"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=400, facecolor="white")
    print(f"Saved: {out}")
plt.close(fig)

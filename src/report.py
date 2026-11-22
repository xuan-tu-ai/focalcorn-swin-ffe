#!/usr/bin/env python3
"""Generate every LaTeX table and figure the manuscript needs.

    python report.py dev     # selection, calibration, and safety on the development folds
    python report.py test    # final per-seed results, confusion matrices, reliability
    python report.py compute # parameter and latency profile of the single final model

Development-set artefacts come from the out-of-fold predictions written by
``phases.py refit``; test artefacts come from ``phases.py evaluate``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

import core
import metrics as M

OUT = Path("outputs")
OOF_DIR, TEST_DIR = OUT / "oof", OUT / "test"
TBL, FIG = OUT / "tables", OUT / "figures"
BS = chr(92)
CLS = ["Highly Fresh", "Fresh", "Not Fresh"]
SHORT = ["HF", "Fresh", "NF"]
LABELS = {
    "ce_gap": "CE + GAP", "ce_gapgmp": "CE + GAP+GMP",
    "corn_gap_std": "CORN + GAP (standard)", "corn_gap_ours": "CORN + GAP (threshold-averaged)",
    "corn_gapgmp_std": "CORN + GAP+GMP (standard)",
    "focalcorn_gap_std": "Focal CORN + GAP (standard)",
    "focalcorn_gap_ours": "Focal CORN + GAP (threshold-averaged)",
    "focalcorn_gapgmp_std": "Focal CORN + GAP+GMP (standard)",
    "focalce_gap": "Focal CE + GAP",
}
BASE_DIR = OUT / "baseline"          # matched re-run of the prior-best pipeline
ALT_DIR = OUT / "alt_branch"          # extra seeds (0-4) were trained by alt_branch_run.py
SEEDS = None                          # set from --seeds; None means final_config.json
FIGW = 4.77                           # text-block width of the manuscript (344 pt), inches
# Figures are drawn at their printed size and included unscaled, so these are the printed
# sizes: 9 pt labels and 8 pt ticks against the 10 pt body text.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                     "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
                     "legend.frameon": False, "pdf.fonttype": 42,
                     "savefig.facecolor": "white", "axes.facecolor": "white",
                     "figure.facecolor": "white"})
# one colour per system in every figure; the order passes the colour-vision check
SYS_COL = {"Proposed": "#2a78d6", "CE + GAP+GMP": "#eb6834", "CE + GAP": "#1baf7a",
           "Hoang RF": "#eda100", "Hoang ET": "#e87ba4"}
GREY = "#b4b4b4"
CM_CMAP = "Blues"


def _seeds(fc):
    return SEEDS if SEEDS else fc["seeds"]


def test_npz(s):
    """Final-model predictions of one seed, wherever that seed was trained."""
    f = TEST_DIR / f"seed{s}.npz"
    return f if f.exists() else ALT_DIR / "focalcorn_gap_ours" / f"seed{s}.npz"


def ce_npz(s, config="ce_gapgmp"):
    """Cross-entropy comparator predictions of one seed."""
    f = OUT / "archive_ce_gapgmp" / f"seed{s}.npz"
    if config == "ce_gapgmp" and f.exists():
        return f
    return ALT_DIR / config / f"seed{s}.npz"


def _w(name, lines):
    TBL.mkdir(parents=True, exist_ok=True)
    (TBL / name).write_text("\n".join(lines))
    print(f"  table  {TBL/name}")


def _f(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)
    print(f"  figure {FIG/name}.pdf")


def _xlabel(ax, label, panel=None):
    """Axis label, with the panel letter centred on its own line beneath it."""
    ax.set_xlabel("\n".join(t for t in (label, panel) if t), linespacing=1.7)


def _cm_plot(ax, cm, title=None, hazard=True, short=True, fs=8):
    """Row-normalised confusion matrix with the count and row percentage in each cell."""
    cn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cn, cmap=CM_CMAP, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n{cn[i,j]*100:.1f}%", ha="center", va="center",
                    fontsize=fs, linespacing=1.15,
                    color="white" if cn[i, j] > 0.5 else "black")
    if hazard:
        ax.add_patch(plt.Rectangle((-0.46, 1.54), 0.92, 0.92, fill=False,
                                   edgecolor="#d62728", lw=1.6))
    names = SHORT if short else CLS
    ax.set_xticks(range(3)); ax.set_xticklabels(names)
    ax.set_yticks(range(3)); ax.set_yticklabels(names, rotation=90, va="center")
    ax.tick_params(length=0)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)
    return im


# ── development-set artefacts ───────────────────────────────────────────────
def _baseline(seeds):
    """Per-classifier metrics of the matched prior-best re-run, or {} if absent."""
    if not (BASE_DIR / "results.json").exists():
        return {}
    out = {}
    for kind, name in (("rf", "RF + LGBM-FS"), ("et", "ET + LGBM-FS")):
        ds = []
        for s in seeds:
            f = BASE_DIR / f"{kind}_seed{s}.npz"
            if f.exists():
                d = dict(np.load(f))
                ds.append(M.all_metrics(d["y_true"], d["y_pred"], d["probs"], d["conf"]))
        if ds:
            out[name] = {"per_seed": ds, "npz": BASE_DIR / f"{kind}_seed{seeds[0]}.npz"}
    return out


def dev():
    sel = {r["config"]: r for r in json.loads((OUT / "selection.json").read_text())}
    order = [c for c in core.CONFIGS if c in sel]
    win = max(order, key=lambda c: sel[c]["fold_qwk_mean"])
    data = {c: dict(np.load(OOF_DIR / f"{c}.npz")) for c in order}
    mm = {c: M.all_metrics(d["y_true"], d["y_pred"], d["probs"], d["conf"])
          for c, d in data.items()}

    # Table: architecture and loss selection (validation)
    best_q = max(order, key=lambda c: sel[c]["fold_qwk_mean"])
    best_e = min(order, key=lambda c: mm[c]["ece"])
    L = [f"{BS}begin{{tabular}}{{lrrrr}}", f"{BS}toprule",
         f"Configuration & Val QWK & Val Acc (\\%) & Val MAE & Val ECE {BS}{BS}",
         f"{BS}midrule"]
    for c in order:
        q = f"{sel[c]['fold_qwk_mean']:.4f} $\\pm$ {sel[c]['fold_qwk_sd']:.4f}"
        if c == best_q:
            q = f"{BS}textbf{{{q}}}"
        e = f"{mm[c]['ece']:.4f}"
        if c == best_e:
            e = f"{BS}textbf{{{e}}}"
        L.append(f"{LABELS[c]} & {q} & {mm[c]['accuracy']*100:.2f} & "
                 f"{mm[c]['mae']:.4f} & {e} {BS}{BS}")
    L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_selection.tex", L)

    # Table: calibration on the development folds
    L = [f"{BS}begin{{tabular}}{{lrrrr}}", f"{BS}toprule",
         f"Configuration & ECE & MCE & Brier & NLL {BS}{BS}", f"{BS}midrule"]
    bests = {k: min(order, key=lambda c: mm[c][k]) for k in ("ece", "mce", "brier", "nll")}
    for c in order:
        cells = []
        for k in ("ece", "mce", "brier", "nll"):
            s = f"{mm[c][k]:.4f}"
            cells.append(f"{BS}textbf{{{s}}}" if bests[k] == c else s)
        L.append(f"{LABELS[c]} & " + " & ".join(cells) + f" {BS}{BS}")
    L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_calibration_dev.tex", L)

    # Table: safety-critical confusions on the development folds
    L = [f"{BS}begin{{tabular}}{{lrrr}}", f"{BS}toprule",
         f"Configuration & true NF $\\rightarrow$ pred HF & true HF $\\rightarrow$ pred NF & "
         f"Total jumps {BS}{BS}", f"{BS}midrule"]
    bn = min(order, key=lambda c: mm[c]["nf_to_hf"])
    for c in order:
        n = f"{BS}textbf{{{mm[c]['nf_to_hf']}}}" if c == bn else str(mm[c]["nf_to_hf"])
        L.append(f"{LABELS[c]} & {n} & {mm[c]['hf_to_nf']} & {mm[c]['total_jumps']} {BS}{BS}")
    L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_safety_dev.tex", L)

    # Figure: selection, kappa as points with fold SD (its axis cannot start at zero, so no
    # bars) and calibration error as bars; the numbers themselves are in the tables
    fcp = OUT / "final_config.json"
    chosen = json.loads(fcp.read_text())["config"] if fcp.exists() else win
    cols = [SYS_COL["Proposed"] if c == chosen else
            SYS_COL["CE + GAP+GMP"] if c == win else GREY for c in order]
    y = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 2.7), sharey=True, layout="constrained")
    for i, c in enumerate(order):
        a1.errorbar(sel[c]["fold_qwk_mean"], i, xerr=sel[c]["fold_qwk_sd"], fmt="o",
                    ms=4.5, color=cols[i], ecolor=cols[i], elinewidth=1.1, capsize=2.5)
    _xlabel(a1, "Validation QWK", "(a)")
    a1.set_xlim(0.85, 0.905)
    a2.barh(y, [mm[c]["ece"] for c in order], color=cols, height=0.62)
    _xlabel(a2, "Validation ECE", "(b)")
    a2.set_xlim(0, 0.105)
    a1.set_yticks(y); a1.set_yticklabels([LABELS[c] for c in order])
    a1.invert_yaxis()
    for ax in (a1, a2):
        ax.xaxis.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    _f(fig, "F_selection")

    # Figure: per-configuration confusion matrices (development folds)
    fig, axes = plt.subplots(2, 5, figsize=(FIGW, 2.4))
    flat = np.atleast_1d(axes).ravel()
    for ax, c in zip(flat, order):
        _cm_plot(ax, M.cm(data[c]["y_true"], data[c]["y_pred"]), LABELS[c])
    for ax in flat[len(order):]:
        ax.axis("off")
    fig.tight_layout(); _f(fig, "F_cm_by_config_dev")

    print(f"\n  highest validation QWK: {win}  "
          f"(val QWK {sel[win]['fold_qwk_mean']:.4f} +/- {sel[win]['fold_qwk_sd']:.4f}, "
          f"ECE {mm[win]['ece']:.4f})")
    fcp = OUT / "final_config.json"
    if fcp.exists():
        fc = json.loads(fcp.read_text())
        crit = fc.get("selection_criterion", "highest mean validation QWK")
        print(f"  selected configuration: {fc['config']}  ({crit})")


# ── test-set artefacts ──────────────────────────────────────────────────────
def test():
    fc = json.loads((OUT / "final_config.json").read_text())
    seeds = _seeds(fc)
    data = {s: dict(np.load(test_npz(s))) for s in seeds}
    mm = {s: M.all_metrics(d["y_true"], d["y_pred"], d["probs"], d["conf"])
          for s, d in data.items()}
    keys = ["accuracy", "f1_macro", "f1_weighted", "mae", "qwk"]

    # Table: per-seed test performance
    L = [f"{BS}begin{{tabular}}{{lrrrrr}}", f"{BS}toprule",
         f"Seed & Acc (\\%) & F1-mac (\\%) & F1-wtd (\\%) & MAE & QWK {BS}{BS}",
         f"{BS}midrule"]
    for s in seeds:
        m = mm[s]
        L.append(f"{s} & {m['accuracy']*100:.2f} & {m['f1_macro']*100:.2f} & "
                 f"{m['f1_weighted']*100:.2f} & {m['mae']:.4f} & {m['qwk']:.4f} {BS}{BS}")
    cells = []
    for k in keys:
        mu, sd = M.agg([mm[s][k] for s in seeds])
        sc = 100 if k.startswith(("accuracy", "f1")) else 1
        dp = 2 if sc == 100 else 4
        cells.append(f"{mu*sc:.{dp}f} $\\pm$ {sd*sc:.{dp}f}")
    L += [f"{BS}midrule",
          f"{BS}textbf{{Mean $\\pm$ SD}} & " + " & ".join(cells) + f" {BS}{BS}",
          f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_test_per_seed.tex", L)

    # Table: comparison with the prior best
    mu = {k: M.agg([mm[s][k] for s in seeds]) for k in keys}
    L = [f"{BS}begin{{tabular}}{{lrrrrr}}", f"{BS}toprule",
         f"Method & Acc (\\%) & F1-mac (\\%) & F1-w (\\%) & MAE & QWK {BS}{BS}",
         f"{BS}midrule",
         f"Ours ({LABELS.get(fc['config'], fc['config'])}) & "
         + " & ".join(f"{BS}textbf{{{mu[k][0]*(100 if k.startswith(('accuracy','f1')) else 1):.{2 if k.startswith(('accuracy','f1')) else 4}f} $\\pm$ "
                      f"{mu[k][1]*(100 if k.startswith(('accuracy','f1')) else 1):.{2 if k.startswith(('accuracy','f1')) else 4}f}}}"
                      for k in keys) + f" {BS}{BS}"]
    for n, b in _baseline(seeds).items():
        cs = []
        for k in keys:
            a, sd = M.agg([r[k] for r in b["per_seed"]])
            sc = 100 if k.startswith(("accuracy", "f1")) else 1
            dp = 2 if sc == 100 else 4
            cs.append(f"{a*sc:.{dp}f} $\\pm$ {sd*sc:.{dp}f}")
        L.append(f"Hoang et al.{BS} {n} (matched re-run) & "
                 + " & ".join(cs) + f" {BS}{BS}")
    L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_vs_prior.tex", L)

    # Table: calibration and safety of the final model
    L = [f"{BS}begin{{tabular}}{{lrrrr}}", f"{BS}toprule",
         f"Metric & ECE & MCE & Brier & NLL {BS}{BS}", f"{BS}midrule"]
    cells = []
    for k in ("ece", "mce", "brier", "nll"):
        a, b = M.agg([mm[s][k] for s in seeds])
        cells.append(f"{a:.4f} $\\pm$ {b:.4f}")
    L += [f"Mean $\\pm$ SD over seeds & " + " & ".join(cells) + f" {BS}{BS}",
          f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_calibration_test.tex", L)

    # representative seed = median QWK (no pooling across seeds)
    med = sorted(seeds, key=lambda s: mm[s]["qwk"])[len(seeds) // 2]
    d = data[med]
    cmm = M.cm(d["y_true"], d["y_pred"])

    fig, ax = plt.subplots(figsize=(3.1, 2.75), layout="constrained")
    im = _cm_plot(ax, cmm, hazard=True, short=False, fs=9)
    fig.colorbar(im, ax=ax, shrink=0.9, aspect=18, ticks=[0, 0.25, 0.5, 0.75, 1])
    _f(fig, "F_confusion_matrix")

    base = _baseline(seeds)
    if base:
        bn = "RF + LGBM-FS" if "RF + LGBM-FS" in base else list(base)[0]
        bfile = BASE_DIR / f"{'rf' if bn.startswith('RF') else 'et'}_seed{med}.npz"
        bd = dict(np.load(bfile if bfile.exists() else base[bn]["npz"]))
        fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.65), layout="constrained")
        _cm_plot(axes[0], M.cm(bd["y_true"], bd["y_pred"]), short=False)
        im = _cm_plot(axes[1], cmm, short=False)
        _xlabel(axes[0], "Predicted label", f"(a) Hoang et al. {bn}")
        _xlabel(axes[1], "Predicted label", "(b) Proposed")
        axes[1].set_ylabel("")
        fig.colorbar(im, ax=axes, shrink=0.8, aspect=18, ticks=[0, 0.25, 0.5, 0.75, 1])
        _f(fig, "F_cm_vs_prior")

    counts = M.ordinal_errors(d["y_true"], d["y_pred"]); n = len(d["y_true"])
    fig, ax = plt.subplots(figsize=(3.3, 2.2), layout="constrained")
    ax.bar(range(3), counts, color=SYS_COL["Proposed"], width=0.55)
    for i, c in enumerate(counts):
        ax.annotate(f"{c} ({c/n*100:.2f}%)", (i, c), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["$|e| = 0$\ncorrect", "$|e| = 1$\nadjacent grade",
                        "$|e| = 2$\nopposite ends"])
    ax.set_ylabel("Test predictions"); ax.set_ylim(0, max(counts) * 1.15)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _f(fig, "F_ordinal_error")

    # Reliability against the cross-entropy comparator, with bin occupancy, since
    # a departure in a bin holding four images means less than one holding four hundred.
    comp = dict(np.load(ce_npz(med))) if ce_npz(med).exists() else None
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(3.3, 3.55), sharex=True, layout="constrained",
                                  gridspec_kw={"height_ratios": [3, 1.1]})
    ax.plot([0, 1], [0, 1], "--", color="#8c8c8c", lw=0.9, label="Perfect calibration")
    series = [(d, SYS_COL["Proposed"], "o", "Proposed")]
    if comp is not None:
        series.append((comp, SYS_COL["CE + GAP+GMP"], "s", "CE + GAP+GMP"))
    for src, col, mk, lab in series:
        xs, ys, ns = [], [], []
        for c, a, f in M._bins(src["y_true"], src["y_pred"], src["conf"]):
            xs.append(f); ys.append(a); ns.append(c)
        e = M.ece(src["y_true"], src["y_pred"], src["conf"])
        ax.plot(xs, ys, marker=mk, color=col, lw=1.3, ms=3.5, label=f"{lab} (ECE {e:.4f})")
        axn.plot(xs, ns, marker=mk, color=col, lw=1.1, ms=3)
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    axn.set_yscale("log"); axn.set_ylabel("Images")
    axn.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    axn.yaxis.set_minor_formatter(NullFormatter())
    axn.set_xlabel("Mean predicted confidence")
    for a in (ax, axn):
        a.grid(True, color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1)
    _f(fig, "F_reliability")

    print(f"\n  representative seed (median QWK): {med}   N = {n}")
    for k in keys + ["ece", "mce", "brier", "nll"]:
        a, b = M.agg([mm[s][k] for s in seeds])
        sc = 100 if k.startswith(("accuracy", "f1")) else 1
        print(f"  {k:<12} {a*sc:.4f} +/- {b*sc:.4f}")


# ── compute profile of the single final model ───────────────────────────────
def compute():
    import time, statistics as st, torch
    fc = json.loads((OUT / "final_config.json").read_text())
    m = core.build_model(fc["cfg"], fc["hp"], pretrained=False).eval()
    tot = sum(p.numel() for p in m.parameters())
    head = sum(p.numel() for p in m.fc.parameters())
    macs = None
    try:
        from thop import profile
        macs, _ = profile(m, inputs=(torch.randn(1, 3, 224, 224).to(core.DEVICE),),
                          verbose=False)
    except Exception:
        pass
    x = torch.randn(1, 3, 224, 224).to(core.DEVICE)
    with torch.no_grad():
        for _ in range(20):
            m(x)
        if core.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t = []
        for _ in range(100):
            s = time.perf_counter(); m(x)
            if core.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t.append((time.perf_counter() - s) * 1000)
    L = [f"{BS}begin{{tabular}}{{lr}}", f"{BS}toprule", f"Metric & Value {BS}{BS}",
         f"{BS}midrule",
         f"Total parameters & {tot:,} {BS}{BS}".replace(",", "{,}"),
         f"{BS}quad Backbone & {tot-head:,} {BS}{BS}".replace(",", "{,}"),
         f"{BS}quad Head & {head:,} {BS}{BS}".replace(",", "{,}")]
    if macs:
        L.append(f"MACs & {macs/1e9:.3f} GMACs {BS}{BS}")
    L += [f"Inference / image & {st.mean(t):.2f} $\\pm$ {st.stdev(t):.2f} ms {BS}{BS}",
          f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_compute.tex", L)
    print(f"  params {tot:,} | head {head:,} | "
          f"{st.mean(t):.2f} ms/image on {core.DEVICE}")


# ── artefacts introduced by the matched baseline and the ablation ───────────
def extra():
    """Tables and figures that the reframed manuscript needs.

    T_ablation_2x2   the loss x head factorial, showing what each part does
    T_baseline_full  the matched re-run on every metric, calibration included
    T_significance   paired-bootstrap intervals for the test-set comparisons
    F_ablation_2x2   the same factorial as a figure
    F_safety         ordinal-error profile of the final model against comparators
    """
    fc = json.loads((OUT / "final_config.json").read_text())
    seeds = _seeds(fc)
    rows = {r["config"]: r for r in json.loads((OUT / "selection.json").read_text())}

    # ---- 2x2 ablation: does calibration come from the loss or the head? ----
    grid = [("Nominal (softmax)", "ce_gap", "focalce_gap"),
            ("Ordinal (CORN)", "corn_gap_std", "focalcorn_gap_std")]
    L = [f"{BS}begin{{tabular}}{{lrrrr}}", f"{BS}toprule",
         f" & {BS}multicolumn{{2}}{{c}}{{Cross-entropy weighting}} & "
         f"{BS}multicolumn{{2}}{{c}}{{Focal weighting}} {BS}{BS}",
         f"{BS}cmidrule(lr){{2-3}} {BS}cmidrule(lr){{4-5}}",
         f"Head & QWK & ECE & QWK & ECE {BS}{BS}", f"{BS}midrule"]
    for name, plain, focal in grid:
        if plain not in rows or focal not in rows:
            continue
        a, b = rows[plain], rows[focal]
        L.append(f"{name} & {a['fold_qwk_mean']:.4f} & {a['oof']['ece']:.4f} & "
                 f"{b['fold_qwk_mean']:.4f} & {BS}textbf{{{b['oof']['ece']:.4f}}} {BS}{BS}")
    L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_ablation_2x2.tex", L)

    if all(c in rows for c in ("ce_gap", "focalce_gap", "corn_gap_std", "focalcorn_gap_std")):
        fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.5), layout="constrained")
        names = ["CE+GAP", "Focal CE+GAP", "CORN+GAP", "Focal CORN+GAP"]
        keys = ["ce_gap", "focalce_gap", "corn_gap_std", "focalcorn_gap_std"]
        cols = ["#9ecae1", "#3182bd", "#a1d99b", "#31a354"]
        for ax, met, lab in ((axes[0], "ece", "Expected calibration error"),
                             (axes[1], "qwk", "Quadratic-weighted kappa")):
            v = [rows[k]["oof"][met] if met == "ece" else rows[k]["fold_qwk_mean"]
                 for k in keys]
            ax.bar(range(4), v, color=cols, width=0.62)
            for i, x in enumerate(v):
                ax.text(i, x + max(v) * 0.02, f"{x:.4f}", ha="center", fontsize=7.5)
            ax.set_xticks(range(4))
            ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7.5)
            ax.set_ylabel(lab); ax.set_ylim(0, max(v) * 1.18)
            ax.spines[["top", "right"]].set_visible(False)
        _f(fig, "F_ablation_2x2")

    # ---- matched baseline on every metric ---------------------------------
    base = _baseline(seeds)
    if base:
        data = {s: dict(np.load(test_npz(s))) for s in seeds}
        ours = [M.all_metrics(d["y_true"], d["y_pred"], d["probs"], d["conf"])
                for d in data.values()]
        mk = ["accuracy", "f1_macro", "mae", "qwk", "ece", "brier", "nll"]
        hdr = ["Acc (\\%)", "F1-mac (\\%)", "MAE", "QWK", "ECE", "Brier", "NLL"]
        L = [f"{BS}begin{{tabular}}{{l" + "r" * len(mk) + "}", f"{BS}toprule",
             "Method & " + " & ".join(hdr) + f" {BS}{BS}", f"{BS}midrule"]

        def row(label, per_seed, bold=False):
            cs = []
            for k in mk:
                a, sd = M.agg([r[k] for r in per_seed])
                sc = 100 if k.startswith(("accuracy", "f1")) else 1
                dp = 2 if sc == 100 else 4
                t = f"{a*sc:.{dp}f} ${BS}pm$ {sd*sc:.{dp}f}"
                cs.append(f"{BS}textbf{{{t}}}" if bold else t)
            return f"{label} & " + " & ".join(cs) + f" {BS}{BS}"

        L.append(row(f"Ours ({LABELS.get(fc['config'], fc['config'])})", ours, True))
        for n, b in base.items():
            L.append(row(f"Hoang et al.{BS} {n}", b["per_seed"]))
        L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
        _w("T_baseline_full.tex", L)

    # ---- paired-bootstrap intervals ---------------------------------------
    sb = sorted(p for p in Path("results").glob("session_bootstrap_2*.json")) \
        if Path("results").exists() else []
    bt = sorted(Path("results").glob("bootstrap_test_*.json")) if Path("results").exists() else []
    names = {**LABELS, "hoang_rf": "Hoang RF + LGBM-FS", "hoang_et": "Hoang ET + LGBM-FS"}
    metric = {"accuracy": "Accuracy (pp)", "QWK": "QWK", "ECE": "ECE",
              "NF->HF": f"NF ${BS}rightarrow$ HF", "two-grade": "Two-grade errors"}
    if sb:
        # session x seed cluster bootstrap with Holm correction (session_bootstrap.py)
        blob = json.loads(sb[-1].read_text())
        L = [f"{BS}begin{{tabular}}{{llrrrl}}", f"{BS}toprule",
             f"Comparison & Metric & ${BS}Delta$ & 95{BS}%~CI & $p_{{{BS}text{{Holm}}}}$ & "
             f"Outcome {BS}{BS}", f"{BS}midrule"]
        prev = None
        for c in blob["test_session_seed"]:
            if prev is not None and c["vs"] != prev:
                L.append(f"{BS}midrule")
            prev = c["vs"]
            sc, dp = (100, 2) if c["metric"] == "accuracy" else \
                     (1, 1) if c["metric"] in ("NF->HF", "two-grade") else (1, 4)
            p = c["p_holm"]
            ps = f"$<0.001$" if p < 0.001 else f"{p:.3f}"
            v = f"{BS}textbf{{significant}}" if c["significant_holm"] else "not significant"
            L.append(f"vs {names.get(c['vs'], c['vs'])} & {metric[c['metric']]} & "
                     f"${c['delta']*sc:+.{dp}f}$ & $[{c['lo']*sc:+.{dp}f}, {c['hi']*sc:+.{dp}f}]$ & "
                     f"{ps} & {v} {BS}{BS}")
        L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
        _w("T_significance.tex", L)
        print(f"  (bootstrap source: {sb[-1].name})")
    elif bt:
        blob = json.loads(bt[-1].read_text())
        L = [f"{BS}begin{{tabular}}{{llrrl}}", f"{BS}toprule",
             f"Comparison & Metric & ${BS}Delta$ & 95{BS}%~CI & Outcome {BS}{BS}",
             f"{BS}midrule"]
        for c in blob["comparisons"]:
            v = "significant" if c["significant"] else "not significant"
            m = c["metric"].replace("->", f"${BS}rightarrow$")
            L.append(f"vs {LABELS.get(c['vs'], c['vs']).replace('_', BS + '_')} & {m} & "
                     f"{c['delta']:+.4f} & [{c['lo']:+.4f}, {c['hi']:+.4f}] & {v} {BS}{BS}")
        L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
        _w("T_significance.tex", L)
        print(f"  (bootstrap source: {bt[-1].name})")

    # ---- ordinal-error profile against the comparators --------------------
    data = {s: dict(np.load(test_npz(s))) for s in seeds}
    series = {"Proposed": [M.ordinal_errors(d["y_true"], d["y_pred"]) for d in data.values()]}
    if all(ce_npz(s).exists() for s in seeds):
        ds = [dict(np.load(ce_npz(s))) for s in seeds]
        series["CE + GAP+GMP"] = [M.ordinal_errors(d["y_true"], d["y_pred"]) for d in ds]
    for n, b in _baseline(seeds).items():
        ds = [dict(np.load(BASE_DIR / f"{'rf' if n.startswith('RF') else 'et'}_seed{s}.npz"))
              for s in seeds
              if (BASE_DIR / f"{'rf' if n.startswith('RF') else 'et'}_seed{s}.npz").exists()]
        if ds:
            series[f"Hoang {n.split()[0]}"] = [M.ordinal_errors(d["y_true"], d["y_pred"])
                                               for d in ds]
    # one panel per error magnitude, each on its own linear axis from zero, so bar heights
    # stay proportional to counts that differ by two orders of magnitude
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 2.3), layout="constrained")
    titles = ["Correct", "One grade off", "Two grades off"]
    for k, (ax, t) in enumerate(zip(axes, titles)):
        for i, (n, runs) in enumerate(series.items()):
            a = np.array(runs, float)[:, k]
            ax.bar(i, a.mean(), 0.7, yerr=a.std(ddof=1) if len(a) > 1 else 0, capsize=2.5,
                   color=SYS_COL[n], error_kw={"elinewidth": 0.9, "capthick": 0.9}, label=n)
        ax.set_xticks([]); _xlabel(ax, "", f"({'abc'[k]}) {t}")
        ax.yaxis.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test predictions per seed")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
               ncol=len(series))
    _f(fig, "F_safety")

    # ---- spoiled fish passed as saleable (true Not Fresh graded otherwise) ----
    systems = {"Proposed (focal CORN + GAP)": [test_npz(s) for s in seeds],
               "CE + GAP+GMP": [ce_npz(s) for s in seeds],
               "CE + GAP": [ce_npz(s, "ce_gap") for s in seeds],
               "Hoang RF + LGBM-FS": [BASE_DIR / f"rf_seed{s}.npz" for s in seeds],
               "Hoang ET + LGBM-FS": [BASE_DIR / f"et_seed{s}.npz" for s in seeds]}
    L = [f"{BS}begin{{tabular}}{{lrrrr}}", f"{BS}toprule",
         f"System & NF recall ({BS}%) & NF $\\rightarrow$ Fresh & NF $\\rightarrow$ HF & "
         f"Passed as saleable ({BS}%) {BS}{BS}", f"{BS}midrule"]
    for name, fs in systems.items():
        if not all(f.exists() for f in fs):
            continue
        rec, pf, ph = [], [], []
        for f in fs:
            d = dict(np.load(f))
            nf = d["y_true"] == 2
            rec.append(float((d["y_pred"][nf] == 2).mean()))
            pf.append(float(np.sum(nf & (d["y_pred"] == 1))))
            ph.append(float(np.sum(nf & (d["y_pred"] == 0))))
        r, rsd = M.agg(rec)
        L.append(f"{name} & {r*100:.2f} $\\pm$ {rsd*100:.2f} & {M.agg(pf)[0]:.1f} & "
                 f"{M.agg(ph)[0]:.1f} & {(1-r)*100:.2f} {BS}{BS}")
    L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_saleable.tex", L)


def reliability_tables():
    """Tables the calibration analysis feeds: comparator metrics, temperature
    scaling, and the accuracy available at a given coverage."""
    fc = json.loads((OUT / "final_config.json").read_text())
    seeds = _seeds(fc)
    src = {"Proposed (focal CORN + GAP)": test_npz,
           "CE + GAP+GMP": ce_npz,
           "CE + GAP": lambda s: ce_npz(s, "ce_gap"),
           "Hoang RF + LGBM-FS": lambda s: BASE_DIR / f"rf_seed{s}.npz",
           "Hoang ET + LGBM-FS": lambda s: BASE_DIR / f"et_seed{s}.npz"}
    got = {}
    for name, where in src.items():
        fs = [where(s) for s in seeds]
        if all(f.exists() for f in fs):
            got[name] = [dict(np.load(f)) for f in fs]
    if len(got) < 2:
        print("  (comparator predictions absent; skipping reliability tables)")
        return

    # every system on every metric, so the comparator is not quoted only in prose
    mk = ["accuracy", "f1_macro", "mae", "qwk", "ece", "brier", "nll"]
    hdr = ["Acc (" + BS + "%)", "F1-mac (" + BS + "%)", "MAE", "QWK", "ECE", "Brier", "NLL"]
    L = [f"{BS}begin{{tabular}}{{l" + "r" * len(mk) + "}", f"{BS}toprule",
         "System & " + " & ".join(hdr) + f" {BS}{BS}", f"{BS}midrule"]
    for name, ds in got.items():
        ms = [M.all_metrics(d["y_true"], d["y_pred"], d["probs"], d["conf"]) for d in ds]
        cs = []
        for k in mk:
            a, sd = M.agg([m[k] for m in ms])
            sc = 100 if k.startswith(("accuracy", "f1")) else 1
            dp = 2 if sc == 100 else 4
            cs.append(f"{a*sc:.{dp}f} ${BS}pm$ {sd*sc:.{dp}f}")
        L.append(f"{name} & " + " & ".join(cs) + f" {BS}{BS}")
    L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_all_systems.tex", L)

    ca = OUT / "calibration_analysis.json"
    if ca.exists():
        blob = json.loads(ca.read_text())
        t = blob.get("temperature", {})
        if t:
            L = [f"{BS}begin{{tabular}}{{lrrrrr}}", f"{BS}toprule",
                 f"System & $T$ & ECE before & ECE after & Acc.\\ after ({BS}%) & "
                 f"Labels changed ({BS}%) {BS}{BS}", f"{BS}midrule"]
            for name, (tt, before, after, acc, changed) in t.items():
                L.append(f"{name} & {tt:.3f} & {before:.4f} & {after:.4f} & "
                         f"{acc*100:.2f} & {changed*100:.2f} {BS}{BS}")
            L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
            _w("T_temperature.tex", L)

        rc, covs = blob.get("risk_coverage", {}), blob.get("coverages", [])
        if rc and covs:
            L = [f"{BS}begin{{tabular}}{{l" + "r" * len(covs) + "}", f"{BS}toprule",
                 "System & " + " & ".join(f"{int(c*100)}{BS}%" for c in covs) + f" {BS}{BS}",
                 f"{BS}midrule"]
            for name, vals in rc.items():
                L.append(f"{name} & " + " & ".join(f"{v*100:.2f}" for v in vals) + f" {BS}{BS}")
            L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
            _w("T_risk_coverage.tex", L)

            # as-trained systems and temperature-scaled ones in separate panels, one colour
            # and marker per system in both, legend outside the data
            short = {"Proposed (focal CORN + GAP)": "Proposed", "Hoang RF + LGBM-FS": "Hoang RF",
                     "Hoang ET + LGBM-FS": "Hoang ET"}
            marker = {"Proposed": "o", "CE + GAP+GMP": "s", "CE + GAP": "^",
                      "Hoang RF": "D", "Hoang ET": "v"}
            fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.75), sharey=True,
                                     layout="constrained")
            x = [c * 100 for c in covs]
            for name, vals in rc.items():
                scaled = name.endswith(", temperature-scaled")
                base = name.removesuffix(", temperature-scaled")
                sysname = short.get(base, base)
                axes[int(scaled)].plot(x, [v * 100 for v in vals], marker=marker[sysname],
                                       ms=3.5, lw=1.2, color=SYS_COL[sysname], label=sysname)
            _xlabel(axes[0], "Coverage (%)", "(a) As trained")
            _xlabel(axes[1], "Coverage (%)", "(b) Temperature-scaled")
            axes[0].set_ylabel("Accuracy on retained (%)")
            axes[0].set_yticks([85, 90, 95, 100])
            for ax in axes:
                ax.set_xlim(102, 48); ax.set_xticks([100, 90, 80, 70, 60, 50])
                ax.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
            fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
                       ncol=5, handlelength=1.4, handletextpad=0.4, columnspacing=0.8)
            _f(fig, "F_risk_coverage")


def per_species():
    """Accuracy and calibration for each species, on the held-out test set.

    Stratifying the split by species is only worth doing if performance is
    reported that way, and an even spread across species is evidence against a
    model that has learned one species and guessed at the rest.
    """
    if not Path(core.DATA_ROOT).exists():
        print("  (dataset not present; skipping the per-species breakdown)")
        return
    fc = json.loads((OUT / "final_config.json").read_text())
    seeds = _seeds(fc)
    dev, test, paths, labels, strata = core.make_split()
    species = np.array([Path(p).parent.name.split(" - ")[0] for p in paths[test]])
    ds = [dict(np.load(test_npz(s))) for s in seeds]

    names = sorted(set(species))
    L = [f"{BS}begin{{tabular}}{{lrrrr}}", f"{BS}toprule",
         f"Species & $n$ & Acc (" + BS + "%) & QWK & ECE " + BS + BS, f"{BS}midrule"]
    rows = []
    for sp in names:
        m = species == sp
        accs, qwks, eces = [], [], []
        for d in ds:
            mm = M.all_metrics(d["y_true"][m], d["y_pred"][m], d["probs"][m], d["conf"][m])
            accs.append(mm["accuracy"]); qwks.append(mm["qwk"]); eces.append(mm["ece"])
        a, asd = M.agg(accs); q, _ = M.agg(qwks); e, _ = M.agg(eces)
        rows.append((sp, int(m.sum()), a, asd, q, e))
        parts = sp.split()
        short = f"{BS}emph{{{parts[0][0]}. {parts[1].lower()}}}" if len(parts) > 1 else sp
        L.append(f"{short} & {int(m.sum())} & {a*100:.2f} $" + BS + "pm$ "
                 f"{asd*100:.2f} & {q:.4f} & {e:.4f} " + BS + BS)
    L += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    _w("T_per_species.tex", L)
    lo = min(rows, key=lambda r: r[2]); hi = max(rows, key=lambda r: r[2])
    print(f"  per-species accuracy spans {lo[2]*100:.2f}% ({lo[0]}) to "
          f"{hi[2]*100:.2f}% ({hi[0]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["dev", "test", "compute", "extra", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="test-set seeds to report (default: those in final_config.json)")
    a = ap.parse_args()
    SEEDS = a.seeds
    print(f"seeds: {SEEDS or 'from final_config.json'}")
    if a.what in ("dev", "all"):
        print("[dev artefacts]"); dev()
    if a.what in ("test", "all"):
        print("[test artefacts]"); test()
    if a.what in ("compute", "all"):
        print("[compute profile]"); compute()
    if a.what in ("extra", "all"):
        print("[baseline and ablation artefacts]"); extra()
        print("[reliability artefacts]"); reliability_tables()
        print("[per-species breakdown]"); per_species()

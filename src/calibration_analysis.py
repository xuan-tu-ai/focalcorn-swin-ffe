#!/usr/bin/env python3
"""Calibration analysis: decoding convention, temperature scaling, and triage.

Four questions that the headline calibration numbers do not by themselves answer.

1. The ordinal head labels by cumulative-product decoding while the nominal head
   labels by argmax, and each reports the probability of its own predicted label.
   How often does the ordinal decode disagree with the argmax of its own
   probability vector, and does the calibration gap survive scoring every system
   under one common convention?
2. Temperature scaling fits a single parameter on held-out data and is the
   standard remedy for over-confidence. Does the focal-weighted objective still
   help once the cross-entropy model is given that remedy?
3. Deployment routes low-confidence cases to an inspector. What accuracy does
   each system deliver at a given coverage?
4. Expected calibration error depends on the binning. Is the ordering stable
   across bin counts and across equal-mass bins?

Reads saved predictions only; no training. Temperature scaling follows each
model's own decision rule: the softmax logits are divided by T and relabelled by
argmax, while the two CORN logits are divided by T and re-decoded by the
cumulative-product rule, so the scaled ordinal model can change labels
(session_bootstrap.corn_scaled).

    python calibration_analysis.py
"""

from __future__ import annotations

import json
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import metrics as M
import session_bootstrap as B

OUT = Path("outputs")
SEEDS = B.ORIG_SEEDS + B.EXTRA_SEEDS
# name -> (test-prediction locations, development OOF file, temperature scaler)
SYSTEMS = {
    "Proposed (focal CORN + GAP)": (B.TEST["proposed"], "outputs/oof/focalcorn_gap_ours.npz",
                                    B.corn_scaled),
    "CE + GAP+GMP": (B.TEST["ce_gapgmp"], "outputs/oof/ce_gapgmp.npz", B.soft_scaled),
    "CE + GAP": (B.TEST["ce_gap"], "outputs/oof/ce_gap.npz", B.soft_scaled),
    "Hoang RF + LGBM-FS": (B.TEST["hoang_rf"], None, None),
    "Hoang ET + LGBM-FS": (B.TEST["hoang_et"], None, None),
}


def ece_common(d, n_bins=15):
    """ECE under one convention for every system: argmax label, its own probability."""
    p = d["probs"]
    yp = p.argmax(1)
    conf = p[np.arange(len(yp)), yp]
    return M.ece(d["y_true"], yp, conf, n_bins), float((yp == d["y_true"]).mean())


def ece_equal_mass(y_true, y_pred, conf, n_bins=15):
    """Bins holding equal numbers of predictions rather than equal width."""
    order = np.argsort(conf)
    correct = (y_pred == y_true).astype(float)[order]
    c = conf[order]
    n = len(c)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            e += (b - a) / n * abs(correct[a:b].mean() - c[a:b].mean())
    return float(e)


def coverage_acc(d, c):
    """Accuracy on the most confident fraction c of the predictions."""
    k = max(1, int(round(len(d["conf"]) * c)))
    idx = np.argsort(-d["conf"])[:k]
    return float((d["y_pred"][idx] == d["y_true"][idx]).mean())


def main() -> int:
    print("=" * 78)
    print("CALIBRATION ANALYSIS")
    print(f"host      : {socket.gethostname()}  |  python {platform.python_version()}")
    print(f"numpy     : {np.__version__}   seeds: {SEEDS}")
    print(f"invoked   : calibration_analysis.py")
    print(f"timestamp : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    data = {}
    for name, (where, oof, scaler) in SYSTEMS.items():
        data[name] = {"test": B.load_runs(where, SEEDS), "oof": oof, "scaler": scaler}
    print(f"systems loaded: {len(data)} x {len(SEEDS)} seeds\n")

    # ---- 1. does the ordinal decode disagree with its own argmax? -----------
    print("-" * 78)
    print("1. DECODING CONVENTION")
    print("-" * 78)
    ours = data["Proposed (focal CORN + GAP)"]["test"]
    dis = [float((d["y_pred"] != d["probs"].argmax(1)).mean()) for d in ours]
    mu, sd = M.agg(dis)
    print(f"cumulative-product decode vs argmax of the same probability vector:")
    print(f"  disagreement rate: {mu*100:.2f}% +/- {sd*100:.2f}% of test predictions")
    # pooled over every seed, so the split between the two rules is not one seed's
    yt = np.concatenate([d["y_true"] for d in ours])
    yd = np.concatenate([d["y_pred"] for d in ours])
    ya = np.concatenate([d["probs"].argmax(1) for d in ours])
    mism = yd != ya
    dec_ok = float((yd[mism] == yt[mism]).mean())
    arg_ok = float((ya[mism] == yt[mism]).mean())
    print(f"  on the {int(mism.sum())} disagreeing predictions (all seeds) the decoded label "
          f"is correct {dec_ok*100:.1f}% of the time, the argmax {arg_ok*100:.1f}%")

    print(f"\n{'system':<30}{'ECE (own rule)':>16}{'ECE (argmax)':>14}{'acc (argmax)':>14}")
    common = {}
    for name, blob in data.items():
        own = M.agg([M.ece(d["y_true"], d["y_pred"], d["conf"]) for d in blob["test"]])[0]
        com = M.agg([ece_common(d)[0] for d in blob["test"]])[0]
        acc = M.agg([ece_common(d)[1] for d in blob["test"]])[0]
        common[name] = {"ece_own": own, "ece_argmax": com, "acc_argmax": acc}
        print(f"{name:<30}{own:>16.4f}{com:>14.4f}{acc*100:>13.2f}%")

    # ---- 2. temperature scaling --------------------------------------------
    print("\n" + "-" * 78)
    print("2. TEMPERATURE SCALING (fitted on development out-of-fold predictions)")
    print("-" * 78)
    print(f"{'system':<30}{'T':>7}{'ECE before':>12}{'ECE after':>11}{'acc after':>11}"
          f"{'changed':>9}")
    temp_rows, scaled = {}, {}
    for name, blob in data.items():
        if not blob["oof"]:
            continue
        o = dict(np.load(blob["oof"]))
        t = B.fit_T(o["probs"], o["y_true"], blob["scaler"])
        before, after, accs, changed, runs = [], [], [], [], []
        for d in blob["test"]:
            pr, yp, cf = blob["scaler"](d["probs"], t)
            runs.append({"y_true": d["y_true"], "y_pred": yp, "conf": cf, "probs": pr})
            before.append(M.ece(d["y_true"], d["y_pred"], d["conf"]))
            after.append(M.ece(d["y_true"], yp, cf))
            accs.append(float((yp == d["y_true"]).mean()))
            changed.append(float((yp != d["y_pred"]).mean()))
        scaled[f"{name}, temperature-scaled"] = runs
        temp_rows[name] = (t, M.agg(before)[0], M.agg(after)[0], M.agg(accs)[0],
                           M.agg(changed)[0])
        print(f"{name:<30}{t:>7.3f}{M.agg(before)[0]:>12.4f}{M.agg(after)[0]:>11.4f}"
              f"{M.agg(accs)[0]*100:>10.2f}%{M.agg(changed)[0]*100:>8.2f}%")

    # ---- 3. selective prediction -------------------------------------------
    print("\n" + "-" * 78)
    print("3. RISK-COVERAGE (accuracy on the retained fraction, mean over seeds)")
    print("-" * 78)
    covs = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    print(f"{'system':<44}" + "".join(f"{int(c*100):>8}%" for c in covs))
    rc = {}
    views = {n: b["test"] for n, b in data.items()}
    views.update(scaled)
    for name, runs in views.items():
        rc[name] = [M.agg([coverage_acc(d, c) for d in runs])[0] for c in covs]
        print(f"{name:<44}" + "".join(f"{a*100:>8.2f}" for a in rc[name]))

    # threshold view: what does a 95% confidence gate actually retain?
    print(f"\n{'system':<44}{'coverage at conf>=0.95':>24}{'accuracy there':>16}")
    gate = {}
    for name, runs in views.items():
        cov, acc = [], []
        for d in runs:
            m = d["conf"] >= 0.95
            cov.append(float(m.mean()))
            acc.append(float((d["y_pred"][m] == d["y_true"][m]).mean()) if m.any() else np.nan)
        gate[name] = (M.agg(cov)[0], M.agg(acc)[0])
        print(f"{name:<44}{M.agg(cov)[0]*100:>23.1f}%{M.agg(acc)[0]*100:>15.2f}%")

    # ---- 4. binning sensitivity --------------------------------------------
    print("\n" + "-" * 78)
    print("4. ECE UNDER DIFFERENT BINNINGS")
    print("-" * 78)
    print(f"{'system':<30}{'M=10':>9}{'M=15':>9}{'M=20':>9}{'equal-mass':>12}")
    binning = {}
    for name, blob in data.items():
        r = []
        for nb in (10, 15, 20):
            r.append(M.agg([M.ece(d["y_true"], d["y_pred"], d["conf"], nb)
                            for d in blob["test"]])[0])
        em = M.agg([ece_equal_mass(d["y_true"], d["y_pred"], d["conf"])
                    for d in blob["test"]])[0]
        binning[name] = r + [em]
        print(f"{name:<30}{r[0]:>9.4f}{r[1]:>9.4f}{r[2]:>9.4f}{em:>12.4f}")

    blob = {"seeds": SEEDS, "decode_disagreement": [mu, sd],
            "disagreeing_cases": {"n": int(mism.sum()), "decoded_correct": dec_ok,
                                  "argmax_correct": arg_ok},
            "common_convention": common, "temperature": temp_rows,
            "risk_coverage": rc, "coverages": covs, "gate_095": gate, "binning": binning}
    f = OUT / "calibration_analysis.json"
    f.write_text(json.dumps(blob, indent=2, default=float))
    print(f"\nverified: {len(data)} systems x {len(SEEDS)} seeds; written to {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

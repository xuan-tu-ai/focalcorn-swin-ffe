#!/usr/bin/env python3
"""Paired bootstrap with the capture session as the resampling unit.

Images from one photography session share background, lighting, and staging, so
they are not independent draws. Every comparison in the paper is recomputed here
with a cluster bootstrap: each replicate draws sessions with replacement and takes
all images of the drawn sessions, and (for the multi-seed test systems) draws each
system's training runs with replacement as well. All systems are scored on the same
session draw and the same seed draw.

The reported difference is the observed full-data difference (mean over seeds),
not the mean of the bootstrap draws: binned calibration error is biased upwards on
resamples that repeat images, so the bootstrap mean of an ECE difference drifts
away from the observed one. The interval is the bootstrap percentile interval and
the p-value is twice the smaller tail mass at zero, Holm-adjusted within each
family of comparisons.

Temperature scaling is also redone here. For the softmax models the logits are
divided by T; for the CORN model the two ordinal logits are recovered from the
saved class probabilities (f0 = 1 - P0, f1 = P2 / (P1 + P2)), divided by T, and
re-decoded by the cumulative-product rule, so the scaled model keeps its own
decision rule and its labels may change.

No GPU, no training. Bootstrap seed 123; the session definition is the one used
by check_leakage.py and session_split_run.py (species, grade, capture date).

    python session_bootstrap.py --n-boot 10000
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import core
import metrics as M
import session_split_run as S

OUT = Path("outputs")
ORIG_SEEDS = [42, 123, 999, 7, 2024]
EXTRA_SEEDS = [0, 1, 2, 3, 4]          # fixed on 2026-09-28 before any of them was trained
BOOT_SEED = 123
NB = M.N_BINS
W = np.array([[(i - j) ** 2 for j in range(3)] for i in range(3)], float) / 4.0

# (location of the original five seeds, location of the extra five)
TEST = {
    "proposed": (OUT / "test" / "seed{}.npz",
                 OUT / "alt_branch" / "focalcorn_gap_ours" / "seed{}.npz"),
    "ce_gapgmp": (OUT / "archive_ce_gapgmp" / "seed{}.npz",
                  OUT / "alt_branch" / "ce_gapgmp" / "seed{}.npz"),
    "ce_gap": (OUT / "alt_branch" / "ce_gap" / "seed{}.npz",) * 2,
    "hoang_rf": (OUT / "baseline" / "rf_seed{}.npz",) * 2,
    "hoang_et": (OUT / "baseline" / "et_seed{}.npz",) * 2,
}
ALT = {"alt_gapgmp": (OUT / "alt_branch" / "focalcorn_gapgmp_std" / "seed{}.npz",) * 2}
SESSION_SPLIT = {
    "proposed": (OUT / "session_split" / "seed{}.npz",) * 2,
    "ce_gap": (OUT / "session_split" / "ce_gap" / "seed{}.npz",) * 2,
    "hoang_rf": (OUT / "baseline_session" / "rf_seed{}.npz",) * 2,
    "hoang_et": (OUT / "baseline_session" / "et_seed{}.npz",) * 2,
}


def load_runs(where, seeds):
    """Every requested seed of one system; a missing file is an error, never skipped."""
    return [dict(np.load(str(where[0 if s in ORIG_SEEDS else 1]).format(s))) for s in seeds]
KEYS = ["accuracy", "QWK", "ECE", "NF->HF", "two-grade"]


# ── per-session sufficient statistics ───────────────────────────────────────
def bin_of(conf: np.ndarray) -> np.ndarray:
    """Bin index matching metrics._bins: bin i holds edges[i] < conf <= edges[i+1]."""
    edges = np.linspace(0.0, 1.0, NB + 1)
    b = np.searchsorted(edges, conf, side="left") - 1
    return np.clip(b, 0, NB - 1)


def suff(yt, yp, conf, sess_idx, n_sess) -> np.ndarray:
    """Per-session vector: [n, 9 confusion cells, NB bin counts, NB sum-correct, NB sum-conf]."""
    b = bin_of(conf)
    cor = (yt == yp).astype(float)
    F = np.zeros((n_sess, 1 + 9 + 3 * NB))
    np.add.at(F[:, 0], sess_idx, 1.0)
    np.add.at(F, (sess_idx, 1 + yt * 3 + yp), 1.0)
    np.add.at(F, (sess_idx, 10 + b), 1.0)
    np.add.at(F, (sess_idx, 10 + NB + b), cor)
    np.add.at(F, (sess_idx, 10 + 2 * NB + b), conf)
    return F


def metrics_from(T: np.ndarray, n_ref: int) -> np.ndarray:
    """Metrics from aggregated statistics, rows = replicates. Counts are per n_ref images."""
    n = T[:, 0]
    O = T[:, 1:10].reshape(-1, 3, 3)
    acc = np.trace(O, axis1=1, axis2=2) / n
    E = O.sum(2)[:, :, None] * O.sum(1)[:, None, :] / n[:, None, None]
    den = (W * E).sum((1, 2))
    qwk = 1.0 - (W * O).sum((1, 2)) / np.where(den > 0, den, np.nan)
    sc, sf = T[:, 10 + NB:10 + 2 * NB], T[:, 10 + 2 * NB:]
    ece = np.abs(sc - sf).sum(1) / n
    haz = O[:, 2, 0] / n * n_ref
    two = (O[:, 2, 0] + O[:, 0, 2]) / n * n_ref
    return np.stack([acc, qwk, ece, haz, two], 1)


def holm(p: np.ndarray) -> np.ndarray:
    idx = np.argsort(p)
    adj, run, m = np.empty(len(p)), 0.0, len(p)
    for r, i in enumerate(idx):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def summarise(obs_d, boot_d):
    lo, hi = np.percentile(boot_d, [2.5, 97.5])
    p = min(2 * min((boot_d > 0).mean(), (boot_d < 0).mean()), 1.0)
    return {"delta": float(obs_d), "lo": float(lo), "hi": float(hi), "p": float(p),
            "boot_mean": float(boot_d.mean())}


def family(rows):
    adj = holm(np.array([r["p"] for r in rows]))
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
        r["significant_holm"] = bool(a < 0.05)
    return rows


# ── temperature scaling ─────────────────────────────────────────────────────
def corn_logits(p):
    f0 = np.clip(1.0 - p[:, 0].astype(float), 1e-7, 1 - 1e-7)
    f1 = np.clip(p[:, 2] / np.clip(p[:, 1] + p[:, 2], 1e-12, None), 1e-7, 1 - 1e-7)
    return np.log(f0 / (1 - f0)), np.log(f1 / (1 - f1))


def corn_scaled(p, T):
    u0, u1 = corn_logits(p)
    f0, f1 = 1 / (1 + np.exp(-u0 / T)), 1 / (1 + np.exp(-u1 / T))
    pr = np.stack([1 - f0, f0 * (1 - f1), f0 * f1], 1)
    yp = (f0 > 0.5).astype(int) + (f0 * f1 > 0.5).astype(int)
    return pr, yp, pr[np.arange(len(yp)), yp]


def soft_scaled(p, T):
    z = np.log(np.clip(p.astype(float), 1e-12, None)) / T
    z -= z.max(1, keepdims=True)
    pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
    yp = pr.argmax(1)
    return pr, yp, pr[np.arange(len(yp)), yp]


def fit_T(p, y, scaler):
    def nll(lt):
        pr, _, _ = scaler(p, np.exp(lt))
        return -np.mean(np.log(np.clip(pr[np.arange(len(y)), y], 1e-12, None)))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


# ── main ────────────────────────────────────────────────────────────────────
def seed_system_boot(data, sess_idx, n_sess, n_ref, n_boot, rng, two_stage=False,
                     strata=None):
    """Bootstrap draws of every metric for every system; returns (observed, draws).

    Sessions are drawn once per replicate and shared by all systems; with `strata`
    (one stratum id per session) they are drawn with replacement within each stratum,
    so every replicate keeps the species-by-grade composition of the test set. Seeds
    are drawn with replacement from each system's own runs (one draw per run count,
    shared by systems with the same number of runs).
    """
    if strata is None:
        counts = rng.multinomial(n_sess, np.full(n_sess, 1 / n_sess), size=n_boot)
    else:
        counts = np.zeros((n_boot, n_sess), dtype=np.int64)
        for g in np.unique(strata):
            mem = np.where(strata == g)[0]
            counts[:, mem] = rng.multinomial(len(mem), np.full(len(mem), 1 / len(mem)),
                                             size=n_boot)
    seed_w = {}
    for k in sorted({len(r) for r in data.values()}):
        seed_w[k] = rng.multinomial(k, np.full(k, 1 / k), size=n_boot) / k
    obs, draws = {}, {}
    if two_stage:
        # image weights: each drawn copy of a session re-draws its images with replacement
        members = [np.where(sess_idx == s)[0] for s in range(n_sess)]
        img_w = np.zeros((n_boot, len(sess_idx)))
        for b in range(n_boot):
            for s in np.nonzero(counts[b])[0]:
                m = members[s]
                img_w[b, m] += rng.multinomial(len(m) * counts[b, s],
                                               np.full(len(m), 1 / len(m)))
    for t, runs in data.items():
        per_seed_obs, per_seed_draw = [], []
        for d in runs:
            if two_stage:
                F = suff(d["y_true"], d["y_pred"], d["conf"],
                         np.arange(len(sess_idx)), len(sess_idx))
                Tb = img_w @ F
            else:
                F = suff(d["y_true"], d["y_pred"], d["conf"], sess_idx, n_sess)
                Tb = counts @ F
            per_seed_obs.append(metrics_from(F.sum(0, keepdims=True), n_ref)[0])
            per_seed_draw.append(metrics_from(Tb, n_ref))
        obs[t] = np.mean(per_seed_obs, 0)
        draws[t] = (np.einsum("bs,sbk->bk", seed_w[len(runs)], np.stack(per_seed_draw))
                    if len(runs) > 1 else per_seed_draw[0])
    return obs, draws


def session_permutation(a_runs, b_runs, sess_idx, n_sess, n_ref, n_perm, rng):
    """Paired permutation test with the session as the exchangeable unit.

    Each system's per-session statistics are averaged over its training runs; a
    permutation swaps the two systems within a random subset of sessions. Returns the
    observed difference and the two-sided p = (1 + #{|d*| >= |d|}) / (1 + n_perm) per metric.
    """
    Fa = np.mean([suff(d["y_true"], d["y_pred"], d["conf"], sess_idx, n_sess) for d in a_runs], 0)
    Fb = np.mean([suff(d["y_true"], d["y_pred"], d["conf"], sess_idx, n_sess) for d in b_runs], 0)
    obs = (metrics_from(Fa.sum(0, keepdims=True), n_ref)[0]
           - metrics_from(Fb.sum(0, keepdims=True), n_ref)[0])
    flip = rng.integers(0, 2, size=(n_perm, n_sess)).astype(float)
    d = (metrics_from(flip @ Fa + (1 - flip) @ Fb, n_ref)
         - metrics_from(flip @ Fb + (1 - flip) @ Fa, n_ref))
    p = (1 + (np.abs(d) >= np.abs(obs) - 1e-12).sum(0)) / (1 + n_perm)
    return obs, p


def gate_boot(data, sess_idx, n_sess, counts, seed_w, thr=0.95):
    """Coverage and accuracy of an absolute confidence gate, observed and bootstrapped."""
    obs, draws = {}, {}
    for t, runs in data.items():
        po, pd_ = [], []
        for d in runs:
            g = d["conf"] >= thr
            F = np.zeros((n_sess, 3))
            np.add.at(F[:, 0], sess_idx, 1.0)
            np.add.at(F[:, 1], sess_idx, g.astype(float))
            np.add.at(F[:, 2], sess_idx, (g & (d["y_pred"] == d["y_true"])).astype(float))
            for Fx, acc in ((F.sum(0, keepdims=True), po), (counts @ F, pd_)):
                acc.append(np.stack([Fx[:, 1] / Fx[:, 0],
                                     Fx[:, 2] / np.maximum(Fx[:, 1], 1)], 1))
        obs[t] = np.mean([x[0] for x in po], 0)
        draws[t] = np.einsum("bs,sbk->bk", seed_w[len(runs)], np.stack(pd_))
    return obs, draws


def compare(obs, draws, ref, others, keys=KEYS):
    rows = []
    for t in others:
        for k in keys:
            j = KEYS.index(k)
            r = summarise(obs[ref][j] - obs[t][j], draws[ref][:, j] - draws[t][:, j])
            rows.append({"vs": t, "metric": k, **r})
    return family(rows)


def show(rows, title):
    print(f"\n{title}")
    print(f"  {'vs':<22}{'metric':<11}{'delta':>10}{'95% CI':>24}{'p':>7}{'p_holm':>8}")
    for r in rows:
        sc = 100 if r["metric"] == "accuracy" else 1
        print(f"  {r.get('vs', ''):<22}{r['metric']:<11}{r['delta']*sc:>10.4f}"
              f"  [{r['lo']*sc:+.4f},{r['hi']*sc:+.4f}]{r['p']:>7.3f}"
              f"{r.get('p_holm', float('nan')):>8.3f}"
              f"  {'SIG' if r.get('significant_holm') else 'n.s.'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=ORIG_SEEDS + EXTRA_SEEDS,
                    help="training seeds of the locked-split test systems")
    ap.add_argument("--session-seeds", type=int, nargs="+", default=ORIG_SEEDS,
                    help="training seeds of the session-disjoint systems")
    args = ap.parse_args()
    rng = np.random.default_rng(BOOT_SEED)
    stamp = {"script": "session_bootstrap.py", "boot_seed": BOOT_SEED, "n_boot": args.n_boot,
             "python": platform.python_version(), "numpy": np.__version__,
             "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "args": vars(args)}
    print("=" * 78)
    print("SESSION-CLUSTERED PAIRED BOOTSTRAP")
    print(json.dumps(stamp))
    print("=" * 78)

    dev, test, paths, labels, _ = core.make_split()
    sess_all = S.sessions_of(paths)
    names = sorted(set(sess_all))
    sid = {s: i for i, s in enumerate(names)}
    sess_test = np.array([sid[s] for s in sess_all[test]])
    sess_dev = np.array([sid[s] for s in sess_all[dev]])
    blob = {"stamp": stamp, "n_sessions": len(names),
            "test_sessions": int(len(set(sess_test))), "dev_sessions": int(len(set(sess_dev)))}
    print(f"sessions: {len(names)} total | test {len(set(sess_test))} | dev {len(set(sess_dev))}")

    # sanity: vectorised metrics reproduce metrics.py on every loaded run
    data = {t: load_runs(w, args.seeds) for t, w in TEST.items()}
    alt = {t: load_runs(w, args.seeds) for t, w in ALT.items()}
    for t, runs in list(data.items()) + list(alt.items()):
        for d in runs:
            assert np.array_equal(d["y_true"], labels[test]), t
            F = suff(d["y_true"], d["y_pred"], d["conf"], sess_test, len(names))
            v = metrics_from(F.sum(0, keepdims=True), len(test))[0]
            # metrics.py averages float32 confidences (rounding ~1e-7); binning must match exactly
            assert abs(v[2] - M.ece(d["y_true"], d["y_pred"], d["conf"].astype(float))) < 1e-12, t
            assert abs(v[2] - M.ece(d["y_true"], d["y_pred"], d["conf"])) < 1e-6, t
            assert abs(v[0] - (d["y_pred"] == d["y_true"]).mean()) < 1e-12, t
    n_runs = sum(len(r) for r in data.values()) + sum(len(r) for r in alt.values())
    print(f"check: vectorised accuracy/ECE match metrics.py on all {n_runs} test runs "
          f"(seeds {args.seeds})")

    # ---- 1. test set, session x seed -------------------------------------
    obs, dr = seed_system_boot(data, sess_test, len(names), len(test), args.n_boot, rng)
    rows = compare(obs, dr, "proposed", [t for t in data if t != "proposed"])
    show(rows, "1. TEST SET — sessions x seeds, Holm over 20")
    blob["test_session_seed"] = rows
    blob["test_observed"] = {t: dict(zip(KEYS, map(float, v))) for t, v in obs.items()}

    # selected configuration vs the branch the rule would have chosen without parsimony
    both = {"proposed": data["proposed"], **alt}
    obsA, drA = seed_system_boot(both, sess_test, len(names), len(test), args.n_boot,
                                 np.random.default_rng(BOOT_SEED))
    rowsA = compare(obsA, drA, "proposed", list(alt))
    show(rowsA, "1c. SELECTED vs ALTERNATIVE BRANCH (Focal CORN + GAP+GMP), Holm over 5")
    blob["test_vs_alt_branch"] = rowsA
    blob["alt_observed"] = {t: dict(zip(KEYS, map(float, v))) for t, v in obsA.items()}

    # sensitivity: two-stage (sessions, then images within session) x seeds
    obs2, dr2 = seed_system_boot(data, sess_test, len(names), len(test),
                                 min(args.n_boot, 4000), np.random.default_rng(BOOT_SEED),
                                 two_stage=True)
    rows2 = compare(obs2, dr2, "proposed", [t for t in data if t != "proposed"])
    show(rows2, "1b. SENSITIVITY — two-stage (session, then image) x seeds")
    blob["test_two_stage"] = rows2

    # sensitivity: sessions resampled within species-by-grade strata
    strata = np.array([n.split("|")[0] for n in names])
    obs3, dr3 = seed_system_boot(data, sess_test, len(names), len(test), args.n_boot,
                                 np.random.default_rng(BOOT_SEED), strata=strata)
    rows3 = compare(obs3, dr3, "proposed", [t for t in data if t != "proposed"])
    show(rows3, f"1d. SENSITIVITY — stratified by species x grade ({len(set(strata))} strata)")
    blob["test_stratified"] = rows3

    # sensitivity: paired permutation test, session as the exchangeable unit
    prng = np.random.default_rng(BOOT_SEED)
    rows4 = []
    for t in [t for t in data if t != "proposed"]:
        d_obs, p = session_permutation(data["proposed"], data[t], sess_test, len(names),
                                       len(test), args.n_boot, prng)
        for k in KEYS:
            j = KEYS.index(k)
            rows4.append({"vs": t, "metric": k, "delta": float(d_obs[j]),
                          "lo": float("nan"), "hi": float("nan"), "p": float(p[j])})
    rows4 = family(rows4)
    print(f"\n1e. SENSITIVITY — paired session permutation test ({args.n_boot} permutations)")
    for r in rows4:
        sc = 100 if r["metric"] == "accuracy" else 1
        print(f"  {r['vs']:<22}{r['metric']:<11}{r['delta']*sc:>10.4f}  p={r['p']:.4f}  "
              f"p_holm={r['p_holm']:.4f}  {'SIG' if r['significant_holm'] else 'n.s.'}")
    blob["test_permutation"] = rows4

    # ---- 2. temperature scaling ------------------------------------------
    print("\n2. TEMPERATURE SCALING (fitted on development OOF)")
    temp, scaled = {}, {}
    for tag, oof, scaler in (("proposed", "focalcorn_gap_ours", corn_scaled),
                             ("ce_gapgmp", "ce_gapgmp", soft_scaled),
                             ("ce_gap", "ce_gap", soft_scaled)):
        o = dict(np.load(OUT / "oof" / f"{oof}.npz"))
        Tv = fit_T(o["probs"], o["y_true"], scaler)
        runs, before, after, acc_a, changed = [], [], [], [], []
        for d in data[tag]:
            pr, yp, cf = scaler(d["probs"], Tv)
            runs.append({"y_true": d["y_true"], "y_pred": yp, "conf": cf, "probs": pr})
            before.append(M.ece(d["y_true"], d["y_pred"], d["conf"]))
            after.append(M.ece(d["y_true"], yp, cf))
            acc_a.append((yp == d["y_true"]).mean())
            changed.append(float((yp != d["y_pred"]).mean()))
        scaled[tag] = runs
        temp[tag] = {"T": Tv, "ece_before": M.agg(before), "ece_after": M.agg(after),
                     "acc_after": M.agg(acc_a), "labels_changed_frac": M.agg(changed),
                     "nll_after": M.agg([M.nll(r["y_true"], r["probs"]) for r in runs]),
                     "brier_after": M.agg([M.brier(r["y_true"], r["probs"]) for r in runs])}
        print(f"  {tag:<10} T={Tv:6.3f}  ECE {M.agg(before)[0]:.4f} -> {M.agg(after)[0]:.4f}"
              f"  acc after {M.agg(acc_a)[0]*100:.2f}%  labels changed "
              f"{M.agg(changed)[0]*100:.2f}%")
    blob["temperature"] = temp

    # selective prediction at an absolute 0.95 gate for the scaled systems
    gate = {}
    for tag, runs in list(scaled.items()) + [("proposed_raw", data["proposed"])]:
        cov = [float((r["conf"] >= 0.95).mean()) for r in runs]
        acc = [float((r["y_pred"] == r["y_true"])[r["conf"] >= 0.95].mean()) for r in runs]
        gate[tag] = {"coverage": M.agg(cov)[0], "accuracy": M.agg(acc)[0]}
        print(f"  gate 0.95  {tag:<13} coverage {M.agg(cov)[0]*100:5.1f}%  "
              f"accuracy {M.agg(acc)[0]*100:6.2f}%")
    blob["gate_095"] = gate

    tsys = {"proposed": data["proposed"], "proposed_T": scaled["proposed"],
            "ce_gapgmp_T": scaled["ce_gapgmp"], "ce_gap_T": scaled["ce_gap"]}
    obsT, drT = seed_system_boot(tsys, sess_test, len(names), len(test), args.n_boot,
                                 np.random.default_rng(BOOT_SEED))
    rowsT = compare(obsT, drT, "proposed", ["ce_gapgmp_T", "ce_gap_T"], ["ECE"])
    rowsT += compare(obsT, drT, "proposed_T", ["ce_gapgmp_T", "ce_gap_T"], ["ECE"])
    rowsT[2]["ref"] = rowsT[3]["ref"] = "proposed_T"
    show(rowsT, "2b. RESIDUAL ECE AFTER SCALING (first two: raw proposed; last two: scaled)")
    blob["temperature_residual"] = rowsT

    # 2c. intervals for the absolute 0.95 gate (coverage, accuracy on what is kept)
    grng = np.random.default_rng(BOOT_SEED)
    gcounts = grng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=args.n_boot)
    gsys = {"proposed": data["proposed"], "ce_gapgmp": data["ce_gapgmp"],
            "ce_gap": data["ce_gap"], "ce_gapgmp_T": scaled["ce_gapgmp"],
            "ce_gap_T": scaled["ce_gap"], "hoang_rf": data["hoang_rf"]}
    gseed = {k: grng.multinomial(k, np.full(k, 1 / k), size=args.n_boot) / k
             for k in sorted({len(r) for r in gsys.values()})}
    gobs, gdr = gate_boot(gsys, sess_test, len(names), gcounts, gseed)
    print("\n2c. GATE AT 0.95 — coverage and accuracy with 95% intervals")
    gate = {}
    for t in gsys:
        cl, ch = np.percentile(gdr[t][:, 0], [2.5, 97.5])
        al, ah = np.percentile(gdr[t][:, 1], [2.5, 97.5])
        gate[t] = {"coverage": [float(gobs[t][0]), float(cl), float(ch)],
                   "accuracy": [float(gobs[t][1]), float(al), float(ah)]}
        print(f"  {t:<12} coverage {gobs[t][0]*100:5.1f}% [{cl*100:5.1f},{ch*100:5.1f}]  "
              f"accuracy {gobs[t][1]*100:6.2f}% [{al*100:6.2f},{ah*100:6.2f}]")
    gdiff = []
    for t in [t for t in gsys if t != "proposed"]:
        for j, k in enumerate(("coverage", "accuracy")):
            gdiff.append({"vs": t, "metric": k, **summarise(gobs["proposed"][j] - gobs[t][j],
                                                             gdr["proposed"][:, j] - gdr[t][:, j])})
    gdiff = family(gdiff)
    for r in gdiff:
        print(f"  proposed - {r['vs']:<12}{r['metric']:<9}{r['delta']*100:+7.2f} pp "
              f"[{r['lo']*100:+.2f},{r['hi']*100:+.2f}]  p_holm={r['p_holm']:.3f}")
    blob["gate_095"] = {"systems": gate, "differences": gdiff}

    # ---- 3. session-disjoint split: the 24 held-out sessions x seeds --------
    _, stest, ssess, held = S.grouped_split(paths, labels)
    hs = {s: i for i, s in enumerate(held)}
    sidx = np.array([hs[s] for s in ssess[stest]])
    sdata = {t: load_runs(w, args.session_seeds) for t, w in SESSION_SPLIT.items()}
    for t, runs in sdata.items():
        for d in runs:
            assert np.array_equal(d["y_true"], labels[stest]), t
    # temperature fitted on session-grouped out-of-fold predictions of this split's
    # development set, so the correction is estimated under the same kind of shift
    stemp = {}
    for tag, sub, scaler in (("proposed", "", corn_scaled), ("ce_gap", "ce_gap", soft_scaled)):
        o = dict(np.load(OUT / "session_split" / sub / "oof.npz"))
        Tv = fit_T(o["probs"], o["y_true"], scaler)
        runs = []
        for d in sdata[tag]:
            pr, yp, cf = scaler(d["probs"], Tv)
            runs.append({"y_true": d["y_true"], "y_pred": yp, "conf": cf, "probs": pr})
        sdata[f"{tag}_T"] = runs
        stemp[tag] = {"T": Tv, "oof_ece": M.ece(o["y_true"], o["y_pred"], o["conf"]),
                      "oof_acc": float((o["y_pred"] == o["y_true"]).mean())}
    blob["session_temperature"] = stemp

    print(f"\n3. SESSION-DISJOINT SPLIT — {len(held)} held-out sessions, "
          f"{len(args.session_seeds)} seeds")
    table = {}
    for t, runs in sdata.items():
        ms = [M.all_metrics(d["y_true"], d["y_pred"], d["probs"], d["conf"]) for d in runs]
        table[t] = {k: M.agg([m[k] for m in ms]) for k in ("accuracy", "qwk", "mae", "ece")}
        a, q, e, ma = (table[t][k] for k in ("accuracy", "qwk", "ece", "mae"))
        print(f"  {t:<12} acc {a[0]*100:5.2f} +/- {a[1]*100:4.2f}  qwk {q[0]:.4f} +/- {q[1]:.4f}  "
              f"mae {ma[0]:.4f}  ece {e[0]:.4f} +/- {e[1]:.4f}"
              + (f"   T = {stemp[t[:-2]]['T']:.3f}" if t.endswith("_T") else ""))
    blob["session_split_table"] = table

    obsS, drS = seed_system_boot(sdata, sidx, len(held), len(stest), args.n_boot,
                                 np.random.default_rng(BOOT_SEED))
    rowsS = compare(obsS, drS, "proposed", ["ce_gap", "ce_gap_T", "hoang_rf", "hoang_et"],
                    ["accuracy", "QWK", "ECE"])
    show(rowsS, "3b. proposed (as trained) vs each system, Holm over 12")
    rowsST = compare(obsS, drS, "proposed_T", ["ce_gap_T"], ["accuracy", "QWK", "ECE"])
    show(rowsST, "3c. proposed, scaled vs CE + GAP, scaled")
    blob["session_split"] = rowsS
    blob["session_split_scaled"] = rowsST
    blob["session_split_observed"] = {t: dict(zip(KEYS, map(float, v))) for t, v in obsS.items()}

    # ---- 4. development OOF, sessions only (one model per fold) ------------
    oof = {f.stem: dict(np.load(f)) for f in sorted((OUT / "oof").glob("*.npz"))}
    for c, d in oof.items():
        assert np.array_equal(d["y_true"], labels[dev]), c
    odata = {c: [d] for c, d in oof.items()}
    obsD, drD = seed_system_boot(odata, sess_dev, len(names), len(dev), args.n_boot,
                                 np.random.default_rng(BOOT_SEED))

    def dpair(a, b, k):
        j = KEYS.index(k)
        return {"a": a, "b": b, "metric": k,
                **summarise(obsD[a][j] - obsD[b][j], drD[a][:, j] - drD[b][:, j])}

    abl = family([dpair("focalce_gap", "ce_gap", "ECE"),
                  dpair("focalcorn_gap_std", "corn_gap_std", "ECE"),
                  dpair("focalce_gap", "ce_gap", "QWK"),
                  dpair("focalcorn_gap_std", "corn_gap_std", "QWK"),
                  dpair("focalce_gap", "ce_gap", "accuracy"),
                  dpair("focalcorn_gap_std", "corn_gap_std", "accuracy"),
                  dpair("corn_gap_std", "ce_gap", "ECE"),
                  dpair("corn_gap_std", "ce_gap", "QWK"),
                  dpair("focalcorn_gap_std", "focalce_gap", "QWK"),
                  dpair("focalcorn_gap_std", "focalce_gap", "ECE")])
    for r in abl:
        r["vs"] = f"{r['a']}-{r['b']}"
    show(abl, "4a. FACTORIAL ABLATION (dev, sessions), Holm over 10")
    blob["dev_ablation"] = abl

    kp = family([dpair(a, b, "QWK") for a, b in combinations(sorted(oof), 2)])
    raw = sum(1 for r in kp if r["lo"] > 0 or r["hi"] < 0)
    hsig = sum(r["significant_holm"] for r in kp)
    print(f"\n4b. KAPPA, 36 PAIRS: uncorrected CI excludes 0 in {raw}; after Holm {hsig}")
    for r in sorted(kp, key=lambda z: z["p"])[:4]:
        print(f"   {r['a']:<22}{r['b']:<22}{r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]"
              f" p={r['p']:.3f} holm={r['p_holm']:.3f}")
    blob["dev_kappa_pairs"] = {"raw": raw, "holm": hsig, "pairs": kp}

    lead = max(oof, key=lambda c: obsD[c][0])
    ac = family([dpair(lead, c, "accuracy") for c in sorted(oof) if c != lead])
    for r in ac:
        r["vs"] = r["b"]
    show(ac, f"4c. DEV ACCURACY vs most accurate ({lead}), Holm over 8")
    blob["dev_accuracy_vs_lead"] = {"lead": lead, "rows": ac,
                                    "raw_sig": sum(1 for r in ac if r["lo"] > 0 or r["hi"] < 0),
                                    "holm_sig": sum(r["significant_holm"] for r in ac)}

    focal = [c for c in oof if c.startswith("focal")]
    plain = [c for c in oof if not c.startswith("focal")]
    props = []
    for f in focal:
        for c in plain:
            j = KEYS.index("ECE")
            props.append(float((drD[f][:, j] < drD[c][:, j]).mean()))
    j = KEYS.index("ECE")
    thr = float((drD["focalcorn_gap_ours"][:, j] < drD["focalcorn_gap_std"][:, j]).mean())
    qk = dpair("focalcorn_gap_ours", "ce_gapgmp", "QWK")
    print(f"\n4d. focal-vs-plain ECE: min resample proportion {min(props):.4f} over {len(props)}")
    print(f"    thr-avg better than standard on ECE in {thr*100:.1f}% of resamples")
    print(f"    dev kappa proposed - ce_gapgmp {qk['delta']:+.4f} [{qk['lo']:+.4f},{qk['hi']:+.4f}]")
    blob["dev_focal_vs_plain_min_prop"] = min(props)
    blob["dev_thr_vs_std_ece_prop"] = thr
    blob["dev_kappa_proposed_vs_cegapgmp"] = qk

    Path("results").mkdir(exist_ok=True)
    f = Path(f"results/session_bootstrap_{datetime.now():%Y-%m-%d}.json")
    f.write_text(json.dumps(blob, indent=2, default=float))
    print(f"\nverified: {len(names)} sessions, {n_runs} test runs, "
          f"{sum(len(r) for r in sdata.values())} session-split runs, "
          f"{len(oof)} OOF sets; written {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write the split out as plain index files so others can reuse it exactly.

A protocol is only reproducible if the partition itself is published. This emits
the development and test image lists, the five development folds, and the
session-disjoint partition, each as newline-separated relative paths, plus a
manifest recording how they were produced.

    python export_split.py --out release_split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

import core


def write(path: Path, paths) -> str:
    rel = [str(Path(p).parent.name + "/" + Path(p).name) for p in paths]
    body = "\n".join(sorted(rel)) + "\n"
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release_split")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dev, test, paths, labels, strata = core.make_split()
    folds = core.make_folds(dev, strata)

    man = {
        "split_seed": core.SPLIT_SEED,
        "test_fraction": core.TEST_FRACTION,
        "strata": "species x freshness class (24 strata)",
        "n_total": int(len(paths)), "n_dev": int(len(dev)), "n_test": int(len(test)),
        "n_folds": core.N_FOLDS,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "numpy": np.__version__, "files": {},
    }
    man["files"]["development.txt"] = write(out / "development.txt", paths[dev])
    man["files"]["test.txt"] = write(out / "test.txt", paths[test])
    for k, (tr, va) in enumerate(folds, 1):
        man["files"][f"fold{k}_train.txt"] = write(out / f"fold{k}_train.txt", paths[tr])
        man["files"][f"fold{k}_val.txt"] = write(out / f"fold{k}_val.txt", paths[va])

    try:
        import session_split_run as S
        sdev, stest, sess, held = S.grouped_split(paths, labels)
        man["session_split"] = {
            "n_sessions": int(len(set(sess))), "n_held_out": len(held),
            "n_dev": int(len(sdev)), "n_test": int(len(stest)),
            "held_out_sessions": held,
        }
        man["files"]["session_development.txt"] = write(
            out / "session_development.txt", paths[sdev])
        man["files"]["session_test.txt"] = write(out / "session_test.txt", paths[stest])
    except Exception as exc:                       # the session split is optional
        print(f"  (session split not exported: {exc})")

    (out / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print("=" * 70)
    print("SPLIT EXPORT")
    print(f"seed {core.SPLIT_SEED} | dev {len(dev)} | test {len(test)} | "
          f"{core.N_FOLDS} folds | 24 strata")
    if "session_split" in man:
        ss = man["session_split"]
        print(f"session-disjoint: {ss['n_held_out']} of {ss['n_sessions']} sessions held "
              f"out, dev {ss['n_dev']} test {ss['n_test']}")
    print(f"{len(man['files'])} index files + MANIFEST.json written to {out}/")
    for k, v in sorted(man["files"].items()):
        print(f"  {k:<28} sha256:{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

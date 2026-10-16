#!/usr/bin/env python3
"""
validate_features.py   (read-only; works for resnet50, conch, UNI ... any folder made by the extract_features*.py scripts)
=========================================================================================================================
Checks that a feature folder is complete, correctly mapped to tiles/patients, and safe to train on.
It NEVER modifies anything (it only writes a report if you pass --report-csv).

For every patient file it checks
  * required arrays exist: features, file_names, x, y, patient, encoder
  * features shape = (tiles in tile_inventory.csv, D), float32, all finite, not constant, no duplicated tile vectors
  * file_names == the inventory's tile names for that patient, in the SAME order (so every vector is traceable to its tile)
  * x, y == the inventory's tile coordinates; the stored patient id and encoder name are correct
Folder-level checks
  * only known patients; no leftover *.tmp.npz files from an interrupted run
  * feature_manifest.csv exists and agrees with the files (rows, tile counts, dimension, finite flag)
  * every patient present (or --allow-partial for a one-patient test)
  * tile ORDER identical to a reference encoder folder (default features\\resnet50), so encoders are compared tile-for-tile
  * extraction_info.json present (model snapshot / preprocessing recorded)
Information only: feature statistics and a sanity check that tiles of the SAME patient are more similar to each other
than to tiles of OTHER patients (a representation that carries no tissue information would not show this).

RUN
    python validate_features.py --features-dir features\\conch --allow-partial     # after the one-patient test
    python validate_features.py --features-dir features\\conch                      # after the full extraction
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\LENOVO\OneDrive\Documents\capstone")
PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
KEYS = {"features", "file_names", "x", "y", "patient", "encoder"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=BASE / "features" / "conch")
    ap.add_argument("--inventory", type=Path, default=BASE / "er_label_outputs" / "tile_inventory.csv")
    ap.add_argument("--reference-dir", type=Path, default=BASE / "features" / "resnet50", help="tile order must match this folder")
    ap.add_argument("--encoder", default=None, help="expected encoder name inside the files (default: the folder name)")
    ap.add_argument("--expected-dim", type=int, default=None, help="default: taken from the first file, then required to be equal everywhere")
    ap.add_argument("--allow-partial", action="store_true", help="accept a folder that has only some patients (one-patient test)")
    ap.add_argument("--report-csv", type=Path, default=None)
    args = ap.parse_args()
    enc_name = args.encoder or args.features_dir.name
    rows: list[dict] = []

    def add(check, status, observed="", detail=""):
        rows.append(dict(check=check, status=status, observed=observed, detail=detail))
        print(f"  [{status}] {check}  {observed}  {detail}".rstrip(), flush=True)

    print("=" * 70 + f"\nFEATURE VALIDATION: {args.features_dir}\n" + "=" * 70)
    if not args.features_dir.is_dir() or not args.inventory.is_file():
        print("ERROR: features folder or inventory not found")
        return 2
    inv = pd.read_csv(args.inventory, encoding="utf-8-sig", dtype={"PATIENT": str, "file_name": str})
    inv = inv[inv.readable.astype(bool)]
    inv_patients = sorted(inv.PATIENT.unique())
    inv_by = {p: g.sort_values("file_name") for p, g in inv.groupby("PATIENT")}

    files = sorted(f for f in args.features_dir.glob("TCGA-*.npz") if ".tmp" not in f.name)
    tmp = sorted(f.name for f in args.features_dir.glob("*.tmp*"))
    have = [f.stem for f in files]
    add("no leftover temporary files", PASS if not tmp else FAIL, observed=f"{len(tmp)} found", detail=", ".join(tmp[:3]))
    unknown = [p for p in have if p not in inv_by]
    add("every feature file belongs to a patient in the inventory", PASS if not unknown else FAIL, observed=f"{len(unknown)} unknown", detail=", ".join(unknown[:5]))
    missing = [p for p in inv_patients if p not in have]
    if args.allow_partial:
        add("patient coverage", INFO, observed=f"{len(have)} of {len(inv_patients)} patients present (partial run allowed)")
    else:
        add("every patient has a feature file", PASS if not missing else FAIL, observed=f"{len(have)} of {len(inv_patients)}", detail=", ".join(missing[:5]))

    dim_seen, bad, stats, per_pat = set(), [], [], {}
    for f in files:
        pid = f.stem
        if pid not in inv_by:
            continue
        g, probs = inv_by[pid], []
        try:
            with np.load(f, allow_pickle=False) as z:
                if not KEYS <= set(z.files):
                    bad.append((pid, f"missing arrays {sorted(KEYS - set(z.files))}"))
                    continue
                a, names, x, y = z["features"], [str(n) for n in z["file_names"]], z["x"], z["y"]
                pf, ef_ = str(z["patient"]), str(z["encoder"])
        except Exception as e:  # noqa: BLE001
            bad.append((pid, f"cannot load: {type(e).__name__}"))
            continue
        dim_seen.add(a.shape[1])
        if a.ndim != 2 or a.shape[0] != len(g):
            probs.append(f"shape {a.shape}, inventory has {len(g)} tiles")
        if a.dtype != np.float32:
            probs.append(f"dtype {a.dtype}")
        if not np.isfinite(a).all():
            probs.append("non-finite values")
        if names != g.file_name.tolist():
            probs.append("tile names/order differ from inventory")
        if list(x) != g.x.fillna(-1).astype(int).tolist() or list(y) != g.y.fillna(-1).astype(int).tolist():
            probs.append("x/y differ from inventory")
        if pf != pid:
            probs.append(f"stored patient id {pf}")
        if ef_ != enc_name:
            probs.append(f"stored encoder {ef_} != {enc_name}")
        if a.ndim == 2 and len(a) > 1:
            if float(a.std(axis=0).max()) < 1e-5:
                probs.append("features are constant")
            if len(np.unique(a, axis=0)) != len(a):
                probs.append("duplicated tile vectors")
        if probs:
            bad.append((pid, "; ".join(probs)))
        elif a.shape[1] and np.isfinite(a).all():
            per_pat[pid] = a
    add("every file loads and passes the per-patient checks", PASS if not bad else FAIL, observed=f"{len(bad)} of {len(files)} files with problems",
        detail="; ".join(f"{p}: {w}" for p, w in bad[:3]))
    exp = args.expected_dim or (next(iter(dim_seen)) if len(dim_seen) == 1 else None)
    add("one feature dimension everywhere" + (f" (= {exp})" if exp else ""), PASS if len(dim_seen) <= 1 and (args.expected_dim is None or dim_seen <= {args.expected_dim}) else FAIL,
        observed=sorted(dim_seen))

    man_path = args.features_dir / "feature_manifest.csv"
    if man_path.is_file():
        man = pd.read_csv(man_path, dtype={"PATIENT": str})
        ok = (set(man.PATIENT) == set(have) and man.all_finite.all() and (man.feature_dim.nunique() <= 1)
              and all(int(r.n_tiles) == len(inv_by[r.PATIENT]) for r in man.itertuples() if r.PATIENT in inv_by))
        add("feature_manifest.csv agrees with the files", PASS if ok else FAIL, observed=f"{len(man)} rows", detail="" if ok else "rows / tile counts / dimension / finite flag differ")
    else:
        add("feature_manifest.csv exists", FAIL)

    ref = args.reference_dir
    if ref.is_dir() and ref.resolve() != args.features_dir.resolve():
        diff, n_cmp = [], 0
        for pid in have:
            rf = ref / f"{pid}.npz"
            if rf.is_file():
                with np.load(rf, allow_pickle=False) as z:
                    n_cmp += 1
                    with np.load(args.features_dir / f"{pid}.npz", allow_pickle=False) as zz:
                        if [str(n) for n in z["file_names"]] != [str(n) for n in zz["file_names"]]:
                            diff.append(pid)
        add(f"tile order identical to {ref.name} (same tiles, same order)", PASS if not diff and n_cmp else (FAIL if diff else WARN),
            observed=f"{n_cmp} patients compared, {len(diff)} differ")
    info = args.features_dir / "extraction_info.json"
    add("extraction_info.json (model snapshot / preprocessing record)", PASS if info.is_file() else WARN, observed="present" if info.is_file() else "missing")

    if per_pat:
        allf = np.concatenate(list(per_pat.values()))
        add("feature statistics", INFO, observed=f"{len(allf)} tiles x {allf.shape[1]} dims | mean {allf.mean():.4f}, sd {allf.std():.4f}, min {allf.min():.3f}, max {allf.max():.3f}, zeros {(allf == 0).mean():.3f}")
    if len(per_pat) >= 2:
        rng = np.random.default_rng(0)
        unit = {p: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12) for p, a in per_pat.items()}
        within = float(np.mean([(u @ u.T)[np.triu_indices(len(u), 1)].mean() for u in unit.values()]))
        ids = list(unit)
        btw = []
        for _ in range(300):
            p, q = rng.choice(len(ids), 2, replace=False)
            btw.append(float((unit[ids[p]] @ unit[ids[q]].T).mean()))
        add("sanity: tiles of the same patient vs different patients (mean cosine)", INFO if within > np.mean(btw) else WARN,
            observed=f"within-patient {within:.3f} vs between-patient {np.mean(btw):.3f}",
            detail="expected: within > between" if within > np.mean(btw) else "within is NOT higher than between - look at the features")

    n_fail = sum(r["status"] == FAIL for r in rows)
    print("\nRESULT: " + ("ALL CHECKS PASSED" + (" (partial folder)" if args.allow_partial else "") if not n_fail else f"{n_fail} CHECK(S) FAILED"))
    if args.report_csv:
        pd.DataFrame(rows).to_csv(args.report_csv, index=False, encoding="utf-8-sig")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

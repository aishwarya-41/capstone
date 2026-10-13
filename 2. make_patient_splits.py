#!/usr/bin/env python3
"""
make_patient_splits.py
======================
Step 2 of the capstone pipeline: create PATIENT-LEVEL cross-validation splits
for ER prediction. No model is trained and no image is opened.

DESIGN (agreed design):
  * Repeated stratified K-fold cross-validation, split by PATIENT (never by tile).
    Default: 5 folds x 3 repeats = 15 train/val/test splits.
  * Inside every split, the patients that are not in the test fold are divided into
    train and validation (default 15% validation), stratified by ER_label.
    The test fold is not touched until final evaluation.
  * Fixed random seed, so rerunning gives exactly the same splits.
  * Leakage checks run automatically. If any check fails the script stops with an
    error and still writes the report, but marks the outputs as NOT SAFE.

INPUTS (read only):
  patient_er_labels.csv   made by validate_er_labels.py   (required)
  tile_inventory.csv      made by validate_er_labels.py   (recommended)
  clinical Excel          only to PRINT PR / subtype balance per fold (optional,
                          information only, never used to build the splits)

OUTPUTS (new folder, old files are renamed with a timestamp, never overwritten):
  patient_cv_splits.csv        one row per patient per split (long format)
  tile_splits.csv              one row per tile, one column per split
  cv_balance_report.csv        patients / positives / negatives per split and role
  cv_split_report.csv          every check with PASS / FAIL, plus the settings used
  cv_covariate_balance_info.csv  (only if --clinical-xlsx is given) PR / subtype
                               counts per test fold, information only
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# =============================================================================
# SETTINGS YOU MAY EDIT (or override with command-line options, see --help)
# =============================================================================
# Folder created by validate_er_labels.py (contains patient_er_labels.csv)
INPUT_DIR = Path(r"C:\Users\YOUR_NAME\Documents\capstone\er_label_outputs")
# New folder for the split files:
OUTPUT_DIR = Path(r"C:\Users\YOUR_NAME\Documents\capstone\cv_splits")

N_FOLDS = 5          # test fold = 1/5 of patients (about 20)
N_REPEATS = 3        # repeat the whole 5-fold procedure with 3 different shuffles
VAL_FRACTION = 0.15  # share of the non-test patients used for validation
SEED = 42            # keep fixed for the whole project; do not change it later

LABEL_COL = "ER_label"
PASS, FAIL, INFO = "PASS", "FAIL", "INFO"


def say(msg: str = "") -> None:
    print(msg, flush=True)


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, check, status, observed="", expected="", detail=""):
        self.rows.append(dict(check=check, status=status, expected=expected,
                              observed=observed, detail=detail))

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["check", "status", "expected", "observed", "detail"])

    def n_fail(self) -> int:
        return sum(r["status"] == FAIL for r in self.rows)


def file_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]


def safe_write_csv(df: pd.DataFrame, path: Path, stamp: str) -> None:
    """Write CSV; if the file exists, rename the old one first and say whether the
    new content is identical (useful to confirm splits did not change)."""
    if path.exists():
        old_hash = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").lstrip(b"\xef\xbb\xbf")).hexdigest()[:16]
        new_hash = file_hash(df)
        backup = path.with_name(f"{path.stem}_previous_{stamp}{path.suffix}")
        path.rename(backup)
        same = "IDENTICAL to" if old_hash == new_hash else "DIFFERENT from"
        say(f"  ({path.name}: new content is {same} the previous file; old one kept as {backup.name})")
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =============================================================================
# BUILD THE SPLITS
# =============================================================================
def build_splits(pat: pd.DataFrame, n_folds: int, n_repeats: int, val_fraction: float, seed: int):
    """Returns long table: split_id, repeat, fold, PATIENT, ER_label, role."""
    pat = pat.sort_values("PATIENT").reset_index(drop=True)   # fixed order -> reproducible
    y = pat[LABEL_COL].astype(int).to_numpy()
    ids = pat["PATIENT"].to_numpy()
    rows = []
    for r in range(1, n_repeats + 1):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for f, (pool_idx, test_idx) in enumerate(skf.split(ids, y), start=1):
            train_idx, val_idx = train_test_split(
                pool_idx, test_size=val_fraction, stratify=y[pool_idx],
                random_state=seed + 1000 * r + f)
            sid = f"r{r}_f{f}"
            for role, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
                for i in idx:
                    rows.append((sid, r, f, ids[i], int(y[i]), role))
    return pd.DataFrame(rows, columns=["split_id", "repeat", "fold", "PATIENT", LABEL_COL, "role"])


# =============================================================================
# LEAKAGE / SANITY CHECKS
# =============================================================================
def check_splits(splits: pd.DataFrame, pat: pd.DataFrame, rep: Report, n_folds, n_repeats):
    all_ids = set(pat["PATIENT"])
    n = len(all_ids)

    # 1. each split contains every patient exactly once
    ok = True
    for sid, g in splits.groupby("split_id"):
        if len(g) != n or set(g["PATIENT"]) != all_ids or g["PATIENT"].duplicated().any():
            ok = False
            rep.add("every patient appears exactly once in each split", FAIL, observed=sid,
                    expected=f"{n} unique patients")
    if ok:
        rep.add("every patient appears exactly once in each split", PASS,
                observed=f"{splits['split_id'].nunique()} splits x {n} patients")

    # 2. train / val / test disjoint (follows from 1, checked explicitly)
    bad = 0
    for sid, g in splits.groupby("split_id"):
        sets = {k: set(v["PATIENT"]) for k, v in g.groupby("role")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            bad += len(sets.get(a, set()) & sets.get(b, set()))
    rep.add("train, validation and test patients never overlap", PASS if bad == 0 else FAIL,
            observed=bad, expected=0)

    # 3. inside one repeat every patient is in the test set exactly once
    bad = 0
    for r, g in splits[splits.role == "test"].groupby("repeat"):
        c = g["PATIENT"].value_counts()
        bad += int((c != 1).sum()) + (n - len(c))
    rep.add("within each repeat every patient is tested exactly once", PASS if bad == 0 else FAIL,
            observed=bad, expected=0)

    # 4. test-fold sizes balanced
    sizes = splits[splits.role == "test"].groupby("split_id").size()
    rep.add("test folds are similar in size", PASS if sizes.max() - sizes.min() <= 1 else FAIL,
            observed=f"min={sizes.min()}, max={sizes.max()}", expected="difference <= 1")

    # 5. every role has both classes (needed for AUC)
    miss = []
    for (sid, role), g in splits.groupby(["split_id", "role"]):
        if g[LABEL_COL].nunique() < 2:
            miss.append(f"{sid}/{role}")
    rep.add("every train / val / test set contains both ER classes", PASS if not miss else FAIL,
            observed=len(miss), expected=0, detail=", ".join(miss[:10]))

    # 6. class ratio similar across test folds
    ratio = splits[splits.role == "test"].groupby("split_id")[LABEL_COL].mean()
    overall = pat[LABEL_COL].astype(int).mean()
    dev = (ratio - overall).abs().max()
    rep.add("ER-positive share in every test fold is close to the overall share",
            PASS if dev <= 0.10 else FAIL,
            observed=f"largest deviation {dev:.3f} (overall {overall:.3f})", expected="<= 0.10")

    # 7. labels in the splits equal labels in the input table
    m = splits.merge(pat[["PATIENT", LABEL_COL]], on="PATIENT", suffixes=("", "_src"))
    rep.add("labels in the splits equal the validated label table",
            PASS if (m[LABEL_COL].astype(int) == m[f"{LABEL_COL}_src"].astype(int)).all() and len(m) == len(splits) else FAIL)


def check_tiles(tile_splits: pd.DataFrame, splits: pd.DataFrame, rep: Report, tiles_per_patient):
    """Tile-level leakage check: every tile inherits the role of its patient."""
    split_cols = [c for c in tile_splits.columns if c.startswith("split_")]
    bad = 0
    for col in split_cols:
        sid = col.replace("split_", "")
        role_by_pat = splits[splits.split_id == sid].set_index("PATIENT")["role"]
        # a patient's tiles must carry exactly one role, equal to the patient's role
        per_pat = tile_splits.groupby("PATIENT")[col].nunique()
        bad += int((per_pat != 1).sum())
        mapped = tile_splits["PATIENT"].map(role_by_pat)
        bad += int((mapped != tile_splits[col]).sum())
    rep.add("all tiles of a patient have the same role in every split (no tile-level leakage)",
            PASS if bad == 0 else FAIL, observed=bad, expected=0)
    counts = tile_splits.groupby("PATIENT").size()
    rep.add("tiles per patient in tile_inventory", INFO,
            observed=f"min={counts.min()}, max={counts.max()}", expected=tiles_per_patient)


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Create patient-level repeated stratified CV splits for ER prediction.")
    ap.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="folder with patient_er_labels.csv (and tile_inventory.csv)")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="folder for the split files")
    ap.add_argument("--clinical-xlsx", type=Path, default=None, help="optional: clinical Excel, only to print PR/subtype balance")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if "YOUR_NAME" in str(args.input_dir) or "YOUR_NAME" in str(args.out_dir):
        say("ERROR: edit INPUT_DIR and OUTPUT_DIR at the top of the script, or use --input-dir / --out-dir.")
        return 2
    lab_path = args.input_dir / "patient_er_labels.csv"
    tile_path = args.input_dir / "tile_inventory.csv"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = Report()

    say("=" * 70)
    say("PATIENT-LEVEL CROSS-VALIDATION SPLITS  (no training, no images opened)")
    say("=" * 70)
    say(f"Label table : {lab_path}")
    say(f"Tile table  : {tile_path}")
    say(f"Output dir  : {args.out_dir}")
    say(f"Settings    : {args.folds} folds x {args.repeats} repeats, validation={args.val_fraction:.0%} of non-test patients, seed={args.seed}")

    if not lab_path.is_file():
        say(f"\nERROR: {lab_path} not found. Run validate_er_labels.py first.")
        return 2
    pat = pd.read_csv(lab_path, encoding="utf-8-sig", dtype={"PATIENT": str})

    # ---- inputs must already be clean: nothing is excluded silently ---------
    for col in ("PATIENT", LABEL_COL, "all_checks_passed"):
        if col not in pat.columns:
            say(f"\nERROR: column '{col}' missing from {lab_path.name}. Columns: {list(pat.columns)}")
            return 2
    problems = []
    if pat["PATIENT"].duplicated().any():
        problems.append("duplicate PATIENT ids")
    if pat[LABEL_COL].isna().any():
        problems.append(f"{int(pat[LABEL_COL].isna().sum())} patient(s) without ER_label")
    if not set(pat[LABEL_COL].dropna().unique()) <= {0, 1}:
        problems.append("ER_label contains values other than 0/1")
    if not pat["all_checks_passed"].astype(str).str.lower().isin(["true"]).all():
        bad = pat.loc[~pat["all_checks_passed"].astype(str).str.lower().eq("true"), "PATIENT"].tolist()
        problems.append(f"patients that failed validation: {', '.join(bad[:10])}")
    if problems:
        say("\nERROR: the label table is not clean, so no splits were made:")
        for p in problems:
            say(f"  - {p}")
        say("Fix these with validate_er_labels.py first. This script never drops patients on its own.")
        return 2
    rep.add("input label table is clean (unique ids, valid 0/1 labels, all checks passed)", PASS,
            observed=f"{len(pat)} patients")
    n_pos, n_neg = int((pat[LABEL_COL] == 1).sum()), int((pat[LABEL_COL] == 0).sum())
    rep.add("class counts in input", INFO, observed=f"ER_label=1: {n_pos}, ER_label=0: {n_neg}")
    if min(n_pos, n_neg) < args.folds * 2:
        say(f"\nERROR: smallest class has {min(n_pos, n_neg)} patients; too few for {args.folds} folds.")
        return 2

    for k, v in dict(folds=args.folds, repeats=args.repeats, val_fraction=args.val_fraction,
                     seed=args.seed, stratified_by=LABEL_COL, split_unit="patient").items():
        rep.add(f"setting: {k}", INFO, observed=v)

    # ---- splits -------------------------------------------------------------
    splits = build_splits(pat, args.folds, args.repeats, args.val_fraction, args.seed)
    check_splits(splits, pat, rep, args.folds, args.repeats)

    # ---- balance report -----------------------------------------------------
    bal = (splits.groupby(["split_id", "repeat", "fold", "role"])[LABEL_COL]
           .agg(n_patients="size", n_ER_positive="sum").reset_index())
    bal["n_ER_negative"] = bal["n_patients"] - bal["n_ER_positive"]
    bal["pct_ER_positive"] = (100 * bal["n_ER_positive"] / bal["n_patients"]).round(1)
    bal = bal.sort_values(["repeat", "fold", "role"]).reset_index(drop=True)

    # ---- tile-level table ---------------------------------------------------
    tile_splits = None
    if tile_path.is_file():
        tiles = pd.read_csv(tile_path, encoding="utf-8-sig", dtype={"PATIENT": str})
        unknown = set(tiles["PATIENT"]) - set(pat["PATIENT"])
        missing = set(pat["PATIENT"]) - set(tiles["PATIENT"])
        rep.add("every tile belongs to a patient in the label table", PASS if not unknown else FAIL,
                observed=len(unknown), expected=0, detail=", ".join(sorted(unknown)[:5]))
        rep.add("every patient has tiles in tile_inventory", PASS if not missing else FAIL,
                observed=len(missing), expected=0, detail=", ".join(sorted(missing)[:5]))
        keep = [c for c in ("PATIENT", "tile_folder_name", "file_name", "full_path", "x", "y", "readable") if c in tiles.columns]
        tile_splits = tiles[keep].copy()
        wide = splits.pivot(index="PATIENT", columns="split_id", values="role")
        wide.columns = [f"split_{c}" for c in wide.columns]
        order = [f"split_r{r}_f{f}" for r in range(1, args.repeats + 1) for f in range(1, args.folds + 1)]
        tile_splits = tile_splits.merge(wide[order], left_on="PATIENT", right_index=True, how="left")
        tile_splits = tile_splits.sort_values(["PATIENT"] + (["file_name"] if "file_name" in tile_splits else [])).reset_index(drop=True)
        check_tiles(tile_splits, splits, rep, "100 expected")
    else:
        rep.add("tile_inventory.csv found", INFO, detail="Not found; tile_splits.csv not created. "
                "Tiles can still be assigned later through PATIENT.")

    # ---- optional information-only covariate balance -------------------------
    cov = None
    if args.clinical_xlsx:
        try:
            clin = pd.read_excel(args.clinical_xlsx, dtype=str, engine="openpyxl")
            want = [c for c in ("PATIENT", "PR status by ihc", "TCGA Subtype") if c in clin.columns]
            if "PATIENT" in want and len(want) > 1:
                c2 = splits[splits.role == "test"].merge(clin[want], on="PATIENT", how="left")
                parts = []
                for col in want[1:]:
                    t = (c2.assign(value=c2[col].fillna("(missing)"))
                           .groupby(["split_id", "value"]).size().rename("n_test_patients").reset_index())
                    t.insert(1, "variable", col)
                    parts.append(t)
                cov = pd.concat(parts, ignore_index=True)
                rep.add("PR / subtype balance written (information only)", INFO, detail="Not used for splitting.")
        except Exception as e:  # noqa: BLE001
            rep.add("clinical Excel read for balance info", INFO, detail=f"{type(e).__name__}: {e}")

    # ---- write ---------------------------------------------------------------
    rep_df = rep.df()
    safe = rep.n_fail() == 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    say("\nWriting output files...")
    safe_write_csv(splits, args.out_dir / "patient_cv_splits.csv", stamp)
    safe_write_csv(bal, args.out_dir / "cv_balance_report.csv", stamp)
    if tile_splits is not None:
        safe_write_csv(tile_splits, args.out_dir / "tile_splits.csv", stamp)
    if cov is not None:
        safe_write_csv(cov, args.out_dir / "cv_covariate_balance_info.csv", stamp)
    rep.add("fingerprint of patient_cv_splits.csv (same value = same splits)", INFO,
            observed=file_hash(splits))
    rep_df = rep.df()
    safe_write_csv(rep_df, args.out_dir / "cv_split_report.csv", stamp)
    say(f"  saved to: {args.out_dir.resolve()}")

    # ---- summary -------------------------------------------------------------
    say("\n" + "=" * 70)
    say("SUMMARY")
    say("=" * 70)
    say(f"Patients: {len(pat)}   (ER+ {n_pos}, ER- {n_neg})")
    say(f"Splits  : {splits['split_id'].nunique()}  ({args.repeats} repeats x {args.folds} folds)")
    t = bal[bal.role == "test"]
    say(f"Test fold: {t.n_patients.min()}-{t.n_patients.max()} patients, "
        f"ER-negative per test fold: {t.n_ER_negative.min()}-{t.n_ER_negative.max()}")
    v = bal[bal.role == "val"]
    say(f"Validation: {v.n_patients.min()}-{v.n_patients.max()} patients, "
        f"ER-negative per validation set: {v.n_ER_negative.min()}-{v.n_ER_negative.max()}")
    tr = bal[bal.role == "train"]
    say(f"Training  : {tr.n_patients.min()}-{tr.n_patients.max()} patients, "
        f"ER-negative per training set: {tr.n_ER_negative.min()}-{tr.n_ER_negative.max()}")
    say("\nChecks:")
    for _, r in rep_df[rep_df.status.isin([PASS, FAIL])].iterrows():
        say(f"  [{r['status']}] {r['check']}" + (f"  ({r['observed']})" if r['observed'] != "" else ""))
    say("\nExample (split r1_f1):")
    ex = bal[bal.split_id == "r1_f1"][["role", "n_patients", "n_ER_positive", "n_ER_negative"]]
    say(ex.to_string(index=False))
    say(f"\nFingerprint: {file_hash(splits)}   (same value on every rerun = splits unchanged)")
    say("\nRESULT: " + ("ALL CHECKS PASSED - splits are safe to use" if safe else "CHECKS FAILED - do NOT use these splits"))
    return 0 if safe else 1


if __name__ == "__main__":
    sys.exit(main())

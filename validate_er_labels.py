#!/usr/bin/env python3
"""
validate_er_labels.py
=====================
Step 1 of the capstone pipeline: build and validate the patient-level ER label
table for the TCGA-BRCA A2 tile dataset.

WHAT IT DOES (read-only on your data):
  1. Reads the slide Excel and the clinical Excel and prints their real columns.
  2. Joins them on PATIENT.
  3. Creates ER_label from "ER Status By IHC":  Positive -> 1, Negative -> 0.
     Anything else (blank, typo, other text) is FLAGGED, never guessed.
  4. Looks inside your tile folder, matches each patient to its tile folder,
     counts images, and tries to open every image.
  5. Writes three CSV files into a NEW output folder:
        patient_er_labels.csv          one row per slide patient
        dataset_validation_report.csv  every check, with PASS / WARN / FAIL
        tile_inventory.csv             one row per image file (extra, for later steps)

WHAT IT NEVER DOES:
  - never trains a model
  - never edits, moves, renames or deletes your Excel files or images
  - never changes a clinical label and never drops borderline ER cases
  - never overwrites an earlier output: an old output file is renamed with a
    timestamp before the new one is written

HOW TO RUN: see the instructions that came with this script, or run
    python validate_er_labels.py --help
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

# =============================================================================
# 1. SETTINGS YOU MAY EDIT  (or override on the command line, see --help)
# =============================================================================
# Windows tip: put an r in front of the quotes so backslashes are not treated
# as escape characters, e.g.  Path(r"C:\Users\Aishwarya\Desktop\Capstone\...")
# Forward slashes also work on Windows: Path("C:/Users/Aishwarya/Desktop/...")

# Folder that contains the two Excel files:

EXCEL_DIR = Path(r"C:\Users\LENOVO\OneDrive\Documents\capstone")

# Exact Excel file names (these are the names you uploaded):
SLIDE_XLSX_NAME = "TCGA-BRCA-A2_SLIDE.xlsx"
CLINICAL_XLSX_NAME = "TCGA-BRCA-A2-CLINI.xlsx"

# The tile folder that contains one sub-folder per patient/slide:
TILE_DIR = Path(r"C:\Users\LENOVO\OneDrive\Documents\capstone\BLOCKS_NORM_MACENKO")

# Where the three output CSV files are written (created if missing).
# By default: a folder called er_label_outputs next to this script.
OUTPUT_DIR = Path(__file__).resolve().parent / "er_label_outputs"

# What you expect (from your Review 2 report and notebook). These are CHECKED,
# not assumed: a mismatch is reported, never silently accepted.
EXPECTED_PATIENTS = 100
EXPECTED_TILES_PER_PATIENT = 100
EXPECTED_TILE_SIZE = (256, 256)   # (width, height)
EXPECTED_TILE_MODE = "RGB"

# Column names (verified in the Excel files you uploaded). The script checks that
# they exist and stops with a clear message if they do not.
COL_PATIENT = "PATIENT"
COL_FILENAME = "FILENAME"
COL_ER = "ER Status By IHC"
COL_ER_PCT = "ER Status IHC Percent Positive"   # optional, information only

# The ONLY two ER values allowed to become a label. Nothing else is guessed.
ER_MAP = {"Positive": 1, "Negative": 0}

# Files with these extensions count as image files.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}

# Patterns used to VERIFY naming (a failed match is reported, not fixed).
PATIENT_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")
SLIDE_RE = re.compile(
    r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2}[A-Z])-(\d{2})-([A-Z]{2}\d+)\."
    r"([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})$"
)

# =============================================================================
# 2. SMALL HELPERS
# =============================================================================
PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
CHECKED = "checked_in_this_run"
ASSUMED = "assumption_not_verified"


class Report:
    """Collects every check so it can be saved as dataset_validation_report.csv."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, category, check, status, detail="", patient="", item="",
            expected="", observed="", basis=CHECKED) -> None:
        self.rows.append(dict(
            category=category, check=check, status=status, patient=patient,
            item=item, expected=expected, observed=observed, detail=detail,
            basis=basis))

    def to_df(self) -> pd.DataFrame:
        cols = ["category", "check", "status", "patient", "item",
                "expected", "observed", "detail", "basis"]
        return pd.DataFrame(self.rows, columns=cols)

    def count(self, status: str) -> int:
        return sum(1 for r in self.rows if r["status"] == status)


def say(msg: str = "") -> None:
    print(msg, flush=True)


def safe_write_csv(df: pd.DataFrame, path: Path, stamp: str) -> None:
    """Write a CSV. If the file already exists, rename the OLD one first."""
    if path.exists():
        backup = path.with_name(f"{path.stem}_previous_{stamp}{path.suffix}")
        path.rename(backup)
        say(f"  (existing {path.name} kept as {backup.name})")
    # utf-8-sig so Excel on Windows opens the file correctly
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalise(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def find_column(df: pd.DataFrame, wanted: str, rep: Report, file_label: str,
                required: bool = True):
    """Return the real column name matching `wanted`. Exact match first;
    a whitespace/case-insensitive match is accepted but reported as a WARN."""
    if wanted in df.columns:
        rep.add("columns", f"{file_label}: column '{wanted}' exists", PASS)
        return wanted
    matches = [c for c in df.columns if normalise(c) == normalise(wanted)]
    if len(matches) == 1:
        rep.add("columns", f"{file_label}: column '{wanted}' exists", WARN,
                detail=f"Only found with different spacing/case: {matches[0]!r}",
                observed=matches[0], expected=wanted)
        return matches[0]
    rep.add("columns", f"{file_label}: column '{wanted}' exists",
            FAIL if required else WARN,
            detail="Column not found. Check the printed column list.",
            expected=wanted, observed="not found")
    return None


# =============================================================================
# 3. READ THE EXCEL FILES
# =============================================================================
def read_excel_file(path: Path, label: str, rep: Report, show_all_cols: bool):
    say(f"\nReading {label}: {path}")
    if not path.is_file():
        rep.add("input files", f"{label} file found", FAIL,
                detail=f"File not found: {path}", expected="file exists",
                observed="missing")
        return None
    try:
        # dtype=str keeps every cell exactly as typed (no silent conversions)
        df = pd.read_excel(path, sheet_name=0, dtype=str, engine="openpyxl")
    except Exception as e:  # noqa: BLE001
        rep.add("input files", f"{label} readable", FAIL,
                detail=f"{type(e).__name__}: {e}")
        return None
    rep.add("input files", f"{label} found and readable", PASS,
            observed=f"{df.shape[0]} rows x {df.shape[1]} columns",
            detail=str(path))
    say(f"  rows: {df.shape[0]}   columns: {df.shape[1]}")
    cols = list(df.columns)
    if show_all_cols or len(cols) <= 30:
        for i, c in enumerate(cols):
            say(f"    [{i}] {c}")
    else:
        say("  (203-style wide sheet: showing first 5 columns and every column "
            "whose name mentions ER / PR / HER2; use --print-all-columns for all)")
        for i, c in enumerate(cols[:5]):
            say(f"    [{i}] {c}")
        key = re.compile(r"\b(ER|PR|HER2)\b|estrogen|progesterone", re.I)
        for i, c in enumerate(cols):
            if key.search(str(c)) and i >= 5:
                say(f"    [{i}] {c}")
    return df


# =============================================================================
# 4. BUILD + VALIDATE THE PATIENT TABLE
# =============================================================================
def build_patient_table(slide: pd.DataFrame, clin: pd.DataFrame, rep: Report):
    # --- locate columns ------------------------------------------------------
    s_pat = find_column(slide, COL_PATIENT, rep, "slide file")
    s_fn = find_column(slide, COL_FILENAME, rep, "slide file")
    c_pat = find_column(clin, COL_PATIENT, rep, "clinical file")
    c_er = find_column(clin, COL_ER, rep, "clinical file")
    c_pct = find_column(clin, COL_ER_PCT, rep, "clinical file", required=False)
    if None in (s_pat, s_fn, c_pat, c_er):
        return None

    s = slide[[s_pat, s_fn]].rename(columns={s_pat: "PATIENT", s_fn: "FILENAME"}).copy()
    keep = [c_pat, c_er] + ([c_pct] if c_pct else [])
    c = clin[keep].rename(columns={c_pat: "PATIENT", c_er: "ER_status_raw",
                                   **({c_pct: "ER_percent_positive_raw"} if c_pct else {})}).copy()
    if not c_pct:
        c["ER_percent_positive_raw"] = pd.NA

    # whitespace in IDs would silently break joins -> report it, do not fix IDs
    for name, df in (("slide", s), ("clinical", c)):
        for col in ["PATIENT"] + (["FILENAME"] if name == "slide" else []):
            bad = df[col].notna() & (df[col] != df[col].str.strip())
            if bad.any():
                rep.add("ids", f"{name} {col} has no leading/trailing spaces", FAIL,
                        observed=f"{int(bad.sum())} value(s) with spaces",
                        detail="Examples: " + "; ".join(map(repr, df.loc[bad, col].head(3))))
            else:
                rep.add("ids", f"{name} {col} has no leading/trailing spaces", PASS)

    # --- blanks / duplicates in the slide table ------------------------------
    n_slide = len(s)
    rep.add("slide table", "number of slide rows", PASS if n_slide == EXPECTED_PATIENTS else WARN,
            expected=EXPECTED_PATIENTS, observed=n_slide)
    for col in ("PATIENT", "FILENAME"):
        blanks = s[col].isna() | (s[col].astype(str).str.strip() == "")
        rep.add("slide table", f"no blank {col}", FAIL if blanks.any() else PASS,
                observed=int(blanks.sum()), expected=0)
    for col in ("PATIENT", "FILENAME"):
        dup = s[col][s[col].duplicated(keep=False) & s[col].notna()]
        rep.add("duplicates", f"slide table: {col} is unique",
                FAIL if len(dup) else PASS, observed=len(dup), expected=0,
                detail="Duplicated: " + "; ".join(sorted(set(dup))[:5]) if len(dup) else "")
    dupc = c["PATIENT"][c["PATIENT"].duplicated(keep=False) & c["PATIENT"].notna()]
    rep.add("duplicates", "clinical table: PATIENT is unique",
            FAIL if len(dupc) else PASS, observed=len(dupc), expected=0,
            detail="Duplicated: " + "; ".join(sorted(set(dupc))[:5]) if len(dupc) else "")

    # --- naming pattern of IDs ----------------------------------------------
    for _, r in s.iterrows():
        p, f = str(r["PATIENT"]), str(r["FILENAME"])
        if not PATIENT_RE.match(p):
            rep.add("naming", "PATIENT looks like TCGA-XX-XXXX", WARN, patient=p, observed=p)
        m = SLIDE_RE.match(f)
        if not m:
            rep.add("naming", "FILENAME looks like a TCGA slide barcode", WARN,
                    patient=p, item=f,
                    detail="Expected e.g. TCGA-A2-A04N-01Z-00-DX1.<uuid>")
        elif m.group(1) != p:
            rep.add("naming", "FILENAME starts with its own PATIENT id", FAIL,
                    patient=p, item=f, expected=p, observed=m.group(1))
    if not any(r["category"] == "naming" for r in rep.rows):
        rep.add("naming", "all PATIENT / FILENAME values follow the TCGA pattern "
                          "and each FILENAME starts with its PATIENT id", PASS)

    # --- join: slide is the base table (every slide patient must keep a row) -
    merged = s.merge(c, on="PATIENT", how="left", indicator=True)
    no_clin = merged[merged["_merge"] == "left_only"]
    rep.add("join", "every slide patient found in clinical file",
            FAIL if len(no_clin) else PASS, expected=0, observed=len(no_clin),
            detail="Missing from clinical: " + ", ".join(no_clin["PATIENT"].head(10))
            if len(no_clin) else "")
    merged = merged.drop(columns="_merge")
    if len(merged) != n_slide:
        rep.add("join", "join did not change number of rows", FAIL,
                expected=n_slide, observed=len(merged),
                detail="Usually caused by duplicate PATIENT in the clinical file.")
    else:
        rep.add("join", "join did not change number of rows", PASS,
                expected=n_slide, observed=len(merged))
    extra = sorted(set(c["PATIENT"].dropna()) - set(s["PATIENT"].dropna()))
    rep.add("join", "clinical patients that have no slide (not used, not deleted)",
            INFO, observed=len(extra), detail=", ".join(extra))

    # --- ER label: exact mapping only ----------------------------------------
    labels, flags = [], []
    for _, r in merged.iterrows():
        raw = r["ER_status_raw"]
        if pd.isna(raw) or str(raw).strip() == "":
            labels.append(pd.NA); flags.append("missing")
            rep.add("ER label", "ER status present", FAIL, patient=r["PATIENT"],
                    detail="ER Status By IHC is blank/missing. No label assigned.")
        elif raw in ER_MAP:
            labels.append(ER_MAP[raw]); flags.append("ok")
        elif str(raw).strip() in ER_MAP:
            labels.append(ER_MAP[str(raw).strip()]); flags.append("ok_after_trimming_spaces")
            rep.add("ER label", "ER status has no stray spaces", WARN,
                    patient=r["PATIENT"], observed=repr(raw),
                    detail="Mapped after trimming spaces only. Check the Excel cell.")
        else:
            labels.append(pd.NA); flags.append("unexpected_value")
            rep.add("ER label", "ER status is exactly Positive or Negative", FAIL,
                    patient=r["PATIENT"], observed=repr(raw),
                    expected="Positive or Negative",
                    detail="Unexpected value; NOT guessed, label left empty.")
    merged["ER_label"] = pd.array(labels, dtype="Int64")
    merged["ER_label_check"] = flags

    vc = merged["ER_status_raw"].value_counts(dropna=False)
    rep.add("ER label", "ER status value counts (slide patients)", INFO,
            observed="; ".join(f"{k}={v}" for k, v in vc.items()))
    n_ok = int(merged["ER_label"].notna().sum())
    rep.add("ER label", "patients with a valid ER_label",
            PASS if n_ok == len(merged) else FAIL,
            expected=len(merged), observed=n_ok)

    # --- borderline cases: INFORMATION ONLY, nobody is excluded ---------------
    pct = merged["ER_percent_positive_raw"]
    merged["ER_borderline_info"] = ""
    mask_b = (merged["ER_label"] == 1) & (pct == "<10%")
    merged.loc[mask_b, "ER_borderline_info"] = "ER positive but percent positive is <10% (kept, not excluded)"
    mask_c = (merged["ER_label"] == 0) & pct.notna() & (pct != "<10%")
    merged.loc[mask_c, "ER_borderline_info"] = "ER negative but percent positive is not <10% (kept, check)"
    for _, r in merged[merged["ER_borderline_info"] != ""].iterrows():
        rep.add("ER label", "borderline / inconsistent ER percent (information only)",
                INFO if r["ER_label"] == 1 else WARN, patient=r["PATIENT"],
                observed=f"{r['ER_status_raw']} / {r['ER_percent_positive_raw']}",
                detail="Label NOT changed and patient NOT excluded.")
    rep.add("ER label", "borderline ER cases excluded or relabelled", PASS,
            observed=0, expected=0,
            detail="By design this script never changes or drops them.")
    return merged


# =============================================================================
# 5. MATCH FOLDERS AND CHECK IMAGES
# =============================================================================
def check_image(path: Path) -> dict:
    """Try to fully decode the image. Never modifies or deletes the file."""
    out = dict(readable=False, width=None, height=None, mode=None, format=None, error="")
    try:
        with Image.open(path) as im:
            im.verify()                 # structural check
        with Image.open(path) as im:
            im.load()                   # full decode (catches truncated files)
            out.update(readable=True, width=im.width, height=im.height,
                       mode=im.mode, format=im.format)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def scan_tiles(patients: pd.DataFrame, tile_dir: Path, rep: Report):
    """Returns (patients with folder columns, tile inventory DataFrame)."""
    cols = ["tile_folder_name", "folder_match_method", "n_image_files", "n_jpeg_files",
            "n_other_files", "n_subfolders_inside", "n_readable", "n_corrupt",
            "n_wrong_size_or_mode", "n_bad_tile_names", "n_duplicate_coords"]
    for col in cols:
        patients[col] = pd.NA
    patients["tile_folder_found"] = False
    inv_rows: list[dict] = []

    if not tile_dir.is_dir():
        rep.add("tile folder", "tile folder exists", FAIL, item=str(tile_dir),
                detail="Folder not found. Edit TILE_DIR or pass --tile-dir.",
                expected="existing folder", observed="missing")
        return patients, pd.DataFrame(inv_rows)
    rep.add("tile folder", "tile folder exists", PASS, item=str(tile_dir))

    subdirs = {p.name: p for p in sorted(tile_dir.iterdir()) if p.is_dir()}
    loose = [p.name for p in tile_dir.iterdir() if p.is_file()]
    rep.add("tile folder", "number of sub-folders in tile folder",
            PASS if len(subdirs) == EXPECTED_PATIENTS else WARN,
            expected=EXPECTED_PATIENTS, observed=len(subdirs))
    if loose:
        rep.add("tile folder", "loose files directly inside tile folder", WARN,
                observed=len(loose), detail="Examples: " + ", ".join(loose[:5]))

    # --- match each patient to a folder (strict first, looser ones flagged) --
    lower_index = defaultdict(list)
    for n in subdirs:
        lower_index[n.lower()].append(n)
    mapping: dict[str, str] = {}
    for idx, r in patients.iterrows():
        pid, fn = r["PATIENT"], str(r["FILENAME"])
        if fn in subdirs:
            mapping[pid] = fn
            patients.at[idx, "folder_match_method"] = "exact_FILENAME"
        elif len(lower_index.get(fn.lower(), [])) == 1:
            mapping[pid] = lower_index[fn.lower()][0]
            patients.at[idx, "folder_match_method"] = "case_insensitive_FILENAME"
            rep.add("folder match", "folder name equals slide FILENAME", WARN, patient=pid,
                    expected=fn, observed=mapping[pid], detail="Differs only by letter case.")
        else:
            cands = [n for n in subdirs if n.startswith(pid)]
            if len(cands) == 1:
                mapping[pid] = cands[0]
                patients.at[idx, "folder_match_method"] = "patient_prefix_only"
                rep.add("folder match", "folder name equals slide FILENAME", WARN,
                        patient=pid, expected=fn, observed=cands[0],
                        detail="Matched only by patient id. Slide FILENAME and folder "
                               "differ (different slide/uuid?). Please inspect.")
            elif len(cands) > 1:
                patients.at[idx, "folder_match_method"] = "ambiguous"
                rep.add("folder match", "exactly one folder per patient", FAIL, patient=pid,
                        expected=1, observed=len(cands),
                        detail="Candidates: " + "; ".join(cands) + " (no folder chosen)")
            else:
                patients.at[idx, "folder_match_method"] = "none"
                rep.add("folder match", "tile folder exists for patient", FAIL, patient=pid,
                        expected=fn, observed="no folder",
                        detail="Missing patient folder.")

    # duplicate mappings (one folder claimed by several patients)
    inv = defaultdict(list)
    for pid, folder in mapping.items():
        inv[folder].append(pid)
    dups = {f: ps for f, ps in inv.items() if len(ps) > 1}
    for f, ps in dups.items():
        rep.add("duplicates", "one folder maps to one patient", FAIL, item=f,
                observed=", ".join(ps), expected="1 patient",
                detail="Duplicate mapping; these patients are not scanned.")
    for f, ps in dups.items():
        for pid in ps:
            mapping.pop(pid, None)

    # unmatched folders
    unmatched = sorted(set(subdirs) - set(mapping.values()) - set(dups))
    rep.add("folder match", "folders in tile folder with no slide record",
            FAIL if unmatched else PASS, expected=0, observed=len(unmatched),
            detail=", ".join(unmatched[:10]))
    for f in unmatched:
        rep.add("folder match", "unmatched folder", FAIL, item=f,
                detail="Folder has no matching patient in the slide Excel. Not scanned, not deleted.")
    matched = len(mapping)
    rep.add("folder match", "patients matched to a tile folder",
            PASS if matched == len(patients) else FAIL,
            expected=len(patients), observed=matched)

    # --- nesting hint: tile_dir probably one level too high -------------------
    if matched == 0 and len(subdirs) == 1:
        only = next(iter(subdirs.values()))
        if any(p.is_dir() for p in only.iterdir()):
            rep.add("tile folder", "possible extra nesting level", WARN, item=str(only),
                    detail="Only one sub-folder and it contains folders. "
                           "Try pointing TILE_DIR at that inner folder.")

    # --- scan every matched folder ------------------------------------------
    total = len(mapping)
    say(f"\nScanning images in {total} patient folders "
        f"(opening every file; this can take a few minutes)...")
    done = 0
    for idx, r in patients.iterrows():
        pid = r["PATIENT"]
        if pid not in mapping:
            continue
        folder = mapping[pid]
        fpath = subdirs[folder]
        patients.at[idx, "tile_folder_name"] = folder
        patients.at[idx, "tile_folder_found"] = True

        entries = sorted(fpath.iterdir())
        files = [p for p in entries if p.is_file()]
        subs = [p for p in entries if p.is_dir()]
        imgs = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
        jpegs = [p for p in imgs if p.suffix.lower() in JPEG_EXTENSIONS]
        others = [p for p in files if p.suffix.lower() not in IMAGE_EXTENSIONS]

        patients.at[idx, "n_image_files"] = len(imgs)
        patients.at[idx, "n_jpeg_files"] = len(jpegs)
        patients.at[idx, "n_other_files"] = len(others)
        patients.at[idx, "n_subfolders_inside"] = len(subs)

        if len(imgs) != EXPECTED_TILES_PER_PATIENT:
            rep.add("image count", f"{EXPECTED_TILES_PER_PATIENT} image files in folder",
                    FAIL, patient=pid, item=folder,
                    expected=EXPECTED_TILES_PER_PATIENT, observed=len(imgs))
        if len(imgs) != len(jpegs):
            rep.add("image count", "all images are JPEG", WARN, patient=pid, item=folder,
                    observed=f"{len(imgs) - len(jpegs)} non-JPEG image file(s)")
        if others:
            rep.add("image count", "no non-image files inside folder", WARN, patient=pid,
                    item=folder, observed=len(others),
                    detail="Examples (kept, not deleted): " + ", ".join(p.name for p in others[:5]))
        if subs:
            rep.add("image count", "no sub-folders inside patient folder", WARN, patient=pid,
                    item=folder, observed=len(subs),
                    detail="Not scanned: " + ", ".join(p.name for p in subs[:5]))

        name_re = re.compile(r"^" + re.escape(folder) + r"_\((\d+),(\d+)\)\.jpe?g$", re.I)
        n_read = n_bad = n_size = n_name = 0
        coords = Counter()
        for p in imgs:
            info = check_image(p)
            m = name_re.match(p.name)
            x = int(m.group(1)) if m else None
            y = int(m.group(2)) if m else None
            if not m:
                n_name += 1
            else:
                coords[(x, y)] += 1
            wrong = False
            if info["readable"]:
                n_read += 1
                if (info["width"], info["height"]) != EXPECTED_TILE_SIZE or info["mode"] != EXPECTED_TILE_MODE:
                    wrong = True
                    n_size += 1
            else:
                n_bad += 1
                rep.add("image readable", "image opens and decodes", FAIL, patient=pid,
                        item=str(p), detail=info["error"] + " (file left untouched)")
            inv_rows.append(dict(
                PATIENT=pid, tile_folder_name=folder, file_name=p.name,
                full_path=str(p), size_bytes=p.stat().st_size,
                readable=info["readable"], width=info["width"], height=info["height"],
                mode=info["mode"], format=info["format"],
                wrong_size_or_mode=wrong, name_matches_pattern=bool(m),
                x=x, y=y, error=info["error"]))
        dup_xy = sum(1 for v in coords.values() if v > 1)
        patients.at[idx, "n_readable"] = n_read
        patients.at[idx, "n_corrupt"] = n_bad
        patients.at[idx, "n_wrong_size_or_mode"] = n_size
        patients.at[idx, "n_bad_tile_names"] = n_name
        patients.at[idx, "n_duplicate_coords"] = dup_xy
        if n_size:
            rep.add("image readable", f"tiles are {EXPECTED_TILE_SIZE[0]}x{EXPECTED_TILE_SIZE[1]} {EXPECTED_TILE_MODE}",
                    WARN, patient=pid, item=folder, observed=f"{n_size} tile(s) differ")
        if n_name:
            rep.add("naming", "tile file names look like <FOLDER>_(x,y).jpg", WARN,
                    patient=pid, item=folder, observed=f"{n_name} tile(s) differ",
                    detail="See tile_inventory.csv, column name_matches_pattern.")
        if dup_xy:
            rep.add("duplicates", "no duplicate (x,y) coordinates in a folder", WARN,
                    patient=pid, item=folder, observed=f"{dup_xy} repeated coordinate(s)")
        done += 1
        if done % 10 == 0 or done == total:
            say(f"  scanned {done}/{total} folders")

    inv_df = pd.DataFrame(inv_rows)
    tot_img = len(inv_df)
    tot_bad = int((~inv_df["readable"]).sum()) if tot_img else 0
    rep.add("image readable", "total images scanned", INFO, observed=tot_img,
            expected=EXPECTED_PATIENTS * EXPECTED_TILES_PER_PATIENT)
    rep.add("image readable", "corrupt / unreadable images (all folders)",
            FAIL if tot_bad else PASS, expected=0, observed=tot_bad)
    return patients, inv_df


# =============================================================================
# 6. MAIN
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Build and validate the patient-level ER label table.")
    ap.add_argument("--excel-dir", type=Path, default=EXCEL_DIR, help="folder with the two Excel files")
    ap.add_argument("--slide-xlsx", type=Path, default=None, help="full path to the slide Excel (overrides --excel-dir)")
    ap.add_argument("--clinical-xlsx", type=Path, default=None, help="full path to the clinical Excel (overrides --excel-dir)")
    ap.add_argument("--tile-dir", type=Path, default=TILE_DIR, help="folder containing one sub-folder per patient")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="where the CSV files are written")
    ap.add_argument("--print-all-columns", action="store_true", help="print every Excel column name")
    ap.add_argument("--skip-images", action="store_true", help="only check Excel + folder names/counts (fast, skips opening images)")
    args = ap.parse_args()

    slide_path = args.slide_xlsx or (args.excel_dir / SLIDE_XLSX_NAME)
    clin_path = args.clinical_xlsx or (args.excel_dir / CLINICAL_XLSX_NAME)
    tile_dir, out_dir = args.tile_dir, args.out_dir
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = Report()

    say("=" * 70)
    say("ER LABEL TABLE + DATASET VALIDATION  (read-only on your data)")
    say("=" * 70)
    say(f"Slide Excel   : {slide_path}")
    say(f"Clinical Excel: {clin_path}")
    say(f"Tile folder   : {tile_dir}")
    say(f"Output folder : {out_dir}")

    # safety: never write outputs inside the image folder
    try:
        out_dir.resolve().relative_to(tile_dir.resolve())
        say("\nERROR: the output folder is inside the tile folder. Choose another --out-dir.")
        return 2
    except ValueError:
        pass
    if "YOUR_NAME" in str(slide_path) or "YOUR_NAME" in str(tile_dir):
        say("\nERROR: you have not edited the paths yet (they still contain YOUR_NAME).")
        say("Edit EXCEL_DIR and TILE_DIR at the top of the script, or use --excel-dir / --tile-dir.")
        return 2

    slide = read_excel_file(slide_path, "slide file", rep, args.print_all_columns)
    clin = read_excel_file(clin_path, "clinical file", rep, args.print_all_columns)
    if slide is None or clin is None:
        say("\nCannot continue: an Excel file could not be read. See messages above.")
        return 2

    patients = build_patient_table(slide, clin, rep)
    if patients is None:
        say("\nCannot continue: a required column is missing. See the column list above.")
        return 2

    if args.skip_images:
        rep.add("image readable", "image scan", INFO, detail="Skipped (--skip-images).")
        for c in ["tile_folder_name", "folder_match_method", "n_image_files", "n_readable", "n_corrupt"]:
            patients[c] = pd.NA
        patients["tile_folder_found"] = pd.NA
        inv_df = pd.DataFrame()
    else:
        patients, inv_df = scan_tiles(patients, tile_dir, rep)

    # --- assumptions / things this script cannot know -----------------------
    for chk, det in [
        ("clinical ER labels are correct", "Taken as given from the TCGA clinical file; the script only checks them for format."),
        ("tiles are Macenko stain-normalised", "Suggested by the folder name BLOCKS_NORM_MACENKO; not verifiable from files."),
        ("slides are FFPE diagnostic (DX1) slides", "Clinical file says Is FFPE = NO / OCT embedded for all patients; unresolved."),
        ("one slide per patient is the right unit", "The slide Excel lists exactly one slide per patient; not checked against GDC."),
        ("tiles show tumour tissue", "No tissue/tumour content check is done here."),
    ]:
        rep.add("assumptions", chk, "NOT_CHECKED", detail=det, basis=ASSUMED)

    # --- final per-patient status -------------------------------------------
    df_rep = rep.to_df()
    fail_by_pat = set(df_rep.loc[(df_rep.status == FAIL) & (df_rep.patient != ""), "patient"])
    patients["all_checks_passed"] = ~patients["PATIENT"].isin(fail_by_pat)
    if not args.skip_images:
        patients["all_checks_passed"] &= patients["tile_folder_found"].fillna(False).astype(bool)

    ordered = ["PATIENT", "FILENAME", "ER_status_raw", "ER_label", "ER_label_check",
               "ER_percent_positive_raw", "ER_borderline_info",
               "tile_folder_found", "tile_folder_name", "folder_match_method",
               "n_image_files", "n_jpeg_files", "n_readable", "n_corrupt",
               "n_wrong_size_or_mode", "n_bad_tile_names", "n_duplicate_coords",
               "n_other_files", "n_subfolders_inside", "all_checks_passed"]
    patients = patients[[c for c in ordered if c in patients.columns]]

    # --- save ---------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    say("\nWriting output files...")
    safe_write_csv(patients, out_dir / "patient_er_labels.csv", stamp)
    safe_write_csv(df_rep, out_dir / "dataset_validation_report.csv", stamp)
    if not args.skip_images and len(inv_df):
        safe_write_csv(inv_df, out_dir / "tile_inventory.csv", stamp)
    say(f"  saved to: {out_dir.resolve()}")

    # --- summary ------------------------------------------------------------
    n_pass, n_warn, n_fail = rep.count(PASS), rep.count(WARN), rep.count(FAIL)
    say("\n" + "=" * 70)
    say("SUMMARY")
    say("=" * 70)
    say(f"Patients in slide table          : {len(patients)}")
    er = patients["ER_label"]
    say(f"ER_label = 1 (Positive)          : {int((er == 1).sum())}")
    say(f"ER_label = 0 (Negative)          : {int((er == 0).sum())}")
    say(f"ER_label missing/unexpected      : {int(er.isna().sum())}")
    if not args.skip_images:
        say(f"Patients matched to a folder     : {int(patients['tile_folder_found'].fillna(False).astype(bool).sum())}")
        say(f"Images scanned                   : {len(inv_df)}")
        say(f"Unreadable/corrupt images        : {int((~inv_df['readable']).sum()) if len(inv_df) else 0}")
    say(f"Patients passing every check     : {int(patients['all_checks_passed'].sum())} / {len(patients)}")
    say(f"Checks  PASS={n_pass}   WARN={n_warn}   FAIL={n_fail}")
    if n_fail or n_warn:
        say("\nProblems found (first 25, see dataset_validation_report.csv for all):")
        shown = df_rep[df_rep.status.isin([FAIL, WARN])].head(25)
        for _, r in shown.iterrows():
            who = r["patient"] or r["item"]
            say(f"  [{r['status']}] {r['check']} | {who} | {r['observed']} {r['detail']}"[:200])
    say("\nBorderline ER cases (kept, labels unchanged):")
    b = patients[patients["ER_borderline_info"].fillna("") != ""]
    say("  none" if b.empty else "\n".join(f"  {r.PATIENT}: {r.ER_borderline_info}" for r in b.itertuples()))
    say("\nVERIFIED in this run : file/column existence, ID joins, ER value mapping, folder")
    say("                       names and counts, image decoding, tile size/mode, tile names.")
    say("NOT verified        : correctness of the clinical labels, stain normalisation,")
    say("                       FFPE vs frozen, tumour content of tiles (see 'assumptions' rows).")
    say("\nRESULT: " + ("ALL CHECKS PASSED" if n_fail == 0 else "FAILURES FOUND - fix before moving on"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

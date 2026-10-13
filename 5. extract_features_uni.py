#!/usr/bin/env python3
"""
extract_features_uni.py   (Step 5 of the ER pipeline - UNI encoder)
===================================================================
Turns every 256x256 H&E tile into a 1024-number feature vector with a FROZEN UNI
(MahmoodLab/UNI, ViT-L/16 pathology foundation model) and saves one file per patient.

This is the UNI twin of extract_features.py (ResNet50). The two scripts are independent:
nothing in your ResNet50 files or folders is touched. UNI features go to features/uni/.

NO TRAINING and NO LABELS are used here. The encoder is not changed.

Reads (never modifies):
  tile_inventory.csv   made by validate_er_labels.py (step 1)
  the tile images      in your BLOCKS_NORM_MACENKO folder

Writes (new folder, resumable, never overwrites unless you say --overwrite):
  features/uni/<PATIENT>.npz         features (N x 1024), tile x/y, tile file names
  features/uni/feature_manifest.csv  one row per patient (tiles, dimension, status)
  features/uni/montage_<PATIENT>.png (only with --montage) 10x10 contact sheet

ONE-TIME SETUP
  python -m pip install timm huggingface_hub
  hf auth login          (paste a READ token from https://huggingface.co/settings/tokens;
                          use the Hugging Face account that was approved for MahmoodLab/UNI)

RECOMMENDED ORDER
  1) one-patient test:   python extract_features_uni.py --patients TCGA-A2-A04N
  2) all patients:       python extract_features_uni.py
     (patients that already have a valid file are skipped, so you can stop and resume)

PREPROCESSING: the tiles are already cut into 256x256 pieces and Macenko stain-normalised, so
nothing else is done to them except resizing to the 224x224 input UNI expects and applying
ImageNet mean/std colour scaling (the values the UNI authors use). No cropping, no filtering.

Licence reminder: UNI is CC-BY-NC-ND 4.0 (non-commercial research). Keep the UNI feature
files private; do not upload them to a public repository.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# =============================================================================
# SETTINGS (defaults match your Mac folders; override on the command line)
# =============================================================================
BASE = Path.home() / "Desktop" / "Capstone"
INVENTORY_CSV = BASE / "Implementation" / "Phase1" / "er_label_outputs" / "tile_inventory.csv"
TILE_DIR = BASE / "Dataset" / "BLOCKS_NORM_MACENKO"
FEATURES_ROOT = BASE / "Implementation" / "Phase1" / "features"

EXPECTED_TILES_PER_PATIENT = 100
BATCH_SIZE = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def say(msg: str = "") -> None:
    print(msg, flush=True)


# =============================================================================
# ENCODERS
# =============================================================================
class UNIEncoder:
    """Frozen UNI (Mahmood Lab, ViT-L/16 pretrained on 100k+ pathology slides). Output = 1024-d CLS embedding.

    Weights are downloaded from Hugging Face (gated model: you must be logged in AND have
    been approved on https://huggingface.co/MahmoodLab/UNI). Licence: CC-BY-NC-ND 4.0
    (non-commercial research only; do not redistribute the model or features derived from it).
    """
    name = "uni"
    HF_ID = "hf-hub:MahmoodLab/uni"

    def __init__(self, device_choice: str = "auto"):
        try:
            import torch
            import timm
            from torchvision import transforms
        except ImportError as e:
            say(f"ERROR: a required package is missing ({e}). Install with:\n"
                "    python -m pip install torch torchvision timm huggingface_hub")
            raise SystemExit(2)
        self.torch = torch
        if device_choice == "auto":
            if torch.cuda.is_available():
                device_choice = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device_choice = "mps"
            else:
                device_choice = "cpu"
        self.device = device_choice
        try:
            # init_values=1e-5 is REQUIRED by the UNI authors so the LayerScale weights load correctly;
            # dynamic_img_size=True lets the ViT accept 224x224 input without complaint.
            model = timm.create_model(self.HF_ID, pretrained=True, init_values=1e-5, dynamic_img_size=True)
        except Exception as e:  # noqa: BLE001
            say("ERROR: could not load UNI from Hugging Face.\n"
                f"       {type(e).__name__}: {str(e)[:300]}\n\n"
                "Usual causes, in order:\n"
                "  1) Not logged in. Run   hf auth login   (or  huggingface-cli login) and paste a token that has READ\n"
                "     permission (create it at https://huggingface.co/settings/tokens). Never share this token.\n"
                "  2) The token belongs to a different account than the one approved for UNI.\n"
                "  3) Access is still pending: open https://huggingface.co/MahmoodLab/UNI and check it says you have access.\n"
                "  4) No internet on first run (the weights, about 1.2 GB, are downloaded once and then cached).\n"
                "  5) timm is too old: python -m pip install -U timm huggingface_hub")
            raise SystemExit(2)
        self.dim = int(model.num_features)
        if self.dim != 1024:
            say(f"WARNING: UNI output dimension is {self.dim}, expected 1024 - continuing, but check you loaded the right model.")
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)             # frozen: no training, ever
        # Same handling as the ResNet50 path: the whole 256x256 tile is resized to 224x224 (no cropping),
        # ImageNet mean/std (these are the values the UNI authors use).
        self.tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.description = (f"UNI (MahmoodLab/UNI, ViT-L/16), dim={self.dim}, device={self.device}, "
                            f"torch={torch.__version__}, timm={timm.__version__}")

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        x = torch.stack([self.tf(im.convert("RGB")) for im in images]).to(self.device)
        with torch.inference_mode():
            f = self.model(x)
        return f.detach().cpu().numpy().astype(np.float32)


def build_encoder(device: str):
    return UNIEncoder(device)


# =============================================================================
# HELPERS
# =============================================================================
def open_tile(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


def encode_tiles(enc, paths: list[Path], batch_size: int) -> np.ndarray:
    feats = []
    for i in range(0, len(paths), batch_size):
        imgs = [open_tile(p) for p in paths[i:i + batch_size]]
        feats.append(enc.encode(imgs))
    return np.concatenate(feats, axis=0)


def existing_file_ok(path: Path, n_expected: int, dim: int) -> bool:
    """A previous output is accepted only if it loads and has the right shape."""
    try:
        with np.load(path, allow_pickle=False) as z:
            f = z["features"]
            return f.shape == (n_expected, dim) and np.isfinite(f).all() and len(z["file_names"]) == n_expected
    except Exception:  # noqa: BLE001
        return False


def save_patient(path: Path, feats, files, xs, ys, patient, enc_name) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")          # write to a temp file, then rename
    np.savez_compressed(tmp, features=feats.astype(np.float32), file_names=np.array(files),
                        x=np.array(xs, dtype=np.int64), y=np.array(ys, dtype=np.int64),
                        patient=np.array(patient), encoder=np.array(enc_name))
    tmp.replace(path)


def make_montage(paths: list[Path], out: Path, grid: int = 10, thumb: int = 64) -> None:
    sheet = Image.new("RGB", (grid * thumb, grid * thumb), (255, 255, 255))
    for i, p in enumerate(paths[:grid * grid]):
        im = open_tile(p).resize((thumb, thumb))
        sheet.paste(im, ((i % grid) * thumb, (i // grid) * thumb))
    sheet.save(out)


def rebuild_manifest(feat_dir: Path, enc_name: str) -> pd.DataFrame:
    rows = []
    for f in sorted(feat_dir.glob("TCGA-*.npz")):
        if ".tmp" in f.name:
            continue
        try:
            with np.load(f, allow_pickle=False) as z:
                feats = z["features"]
                rows.append(dict(PATIENT=f.stem, n_tiles=feats.shape[0], feature_dim=feats.shape[1],
                                 encoder=enc_name, all_finite=bool(np.isfinite(feats).all()), file=f.name))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(PATIENT=f.stem, n_tiles=-1, feature_dim=-1, encoder=enc_name,
                             all_finite=False, file=f"{f.name} (UNREADABLE: {type(e).__name__})"))
    return pd.DataFrame(rows, columns=["PATIENT", "n_tiles", "feature_dim", "encoder", "all_finite", "file"])


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Extract frozen UNI features for every tile (no training).")
    ap.add_argument("--inventory", type=Path, default=INVENTORY_CSV, help="tile_inventory.csv from step 1")
    ap.add_argument("--tile-dir", type=Path, default=TILE_DIR, help="folder with one sub-folder per patient")
    ap.add_argument("--out-root", type=Path, default=FEATURES_ROOT, help="features are written to <out-root>/<encoder>/")
    ap.add_argument("--patients", nargs="+", default=None, help="only these patient ids (e.g. TCGA-A2-A04N)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N patients (use 1 for the one-patient test)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--overwrite", action="store_true", help="redo patients that already have a valid feature file")
    ap.add_argument("--montage", action="store_true", help="save a 10x10 contact sheet of the first processed patient")
    ap.add_argument("--check-determinism", action="store_true", help="re-encode 8 tiles per patient and compare (always on for the first patient)")
    args = ap.parse_args()

    say("=" * 70)
    say("FEATURE EXTRACTION  (frozen encoder, no training, no labels)")
    say("=" * 70)
    say(f"Inventory : {args.inventory}")
    say(f"Tile dir  : {args.tile_dir}")
    if not args.inventory.is_file():
        say("ERROR: tile_inventory.csv not found. Run validate_er_labels.py first (or pass --inventory).")
        return 2
    if not args.tile_dir.is_dir():
        say("ERROR: tile folder not found (pass --tile-dir).")
        return 2

    inv = pd.read_csv(args.inventory, encoding="utf-8-sig", dtype={"PATIENT": str, "tile_folder_name": str, "file_name": str})
    need = {"PATIENT", "tile_folder_name", "file_name", "x", "y", "readable"}
    if not need <= set(inv.columns):
        say(f"ERROR: tile_inventory.csv is missing columns {sorted(need - set(inv.columns))}")
        return 2
    n_unreadable = int((~inv["readable"].astype(bool)).sum())
    if n_unreadable:
        say(f"WARNING: {n_unreadable} tile(s) are marked unreadable in the inventory; they are skipped and the patient is reported.")
    all_patients = sorted(inv["PATIENT"].unique())
    patients = all_patients
    if args.patients:
        unknown = [p for p in args.patients if p not in all_patients]
        if unknown:
            say(f"ERROR: unknown patient ids: {unknown}")
            return 2
        patients = sorted(args.patients)
    if args.limit:
        patients = patients[:args.limit]

    enc = build_encoder(args.device)
    feat_dir = args.out_root / enc.name
    feat_dir.mkdir(parents=True, exist_ok=True)
    say(f"Encoder   : {enc.name}  ({enc.description})")
    say(f"Output    : {feat_dir}")
    say(f"Patients  : {len(patients)} selected of {len(all_patients)}")
    say("")

    done, skipped, failed = [], [], []
    times = []
    first = True
    for i, pid in enumerate(patients, start=1):
        g = inv[(inv["PATIENT"] == pid) & (inv["readable"].astype(bool))].sort_values("file_name")
        out = feat_dir / f"{pid}.npz"
        n = len(g)
        if n == 0:
            failed.append((pid, "no readable tiles in inventory"))
            continue
        if out.exists() and not args.overwrite and existing_file_ok(out, n, enc.dim):
            skipped.append(pid)
            say(f"[{i}/{len(patients)}] {pid}: valid file exists, skipped")
            continue
        folder = g["tile_folder_name"].iloc[0]
        paths = [args.tile_dir / folder / f for f in g["file_name"]]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            failed.append((pid, f"{len(missing)} tile file(s) not found, e.g. {missing[0].name}"))
            say(f"[{i}/{len(patients)}] {pid}: FAILED - {failed[-1][1]}")
            continue
        t0 = time.time()
        try:
            feats = encode_tiles(enc, paths, args.batch_size)
        except Exception as e:  # noqa: BLE001
            failed.append((pid, f"{type(e).__name__}: {e}"))
            say(f"[{i}/{len(patients)}] {pid}: FAILED - {failed[-1][1]}")
            continue
        secs = time.time() - t0

        # ---- per-patient checks ------------------------------------------------
        problems = []
        if feats.shape != (n, enc.dim):
            problems.append(f"shape {feats.shape}, expected {(n, enc.dim)}")
        if not np.isfinite(feats).all():
            problems.append("NaN or infinite values in features")
        if n != EXPECTED_TILES_PER_PATIENT:
            say(f"    note: {pid} has {n} tiles (expected {EXPECTED_TILES_PER_PATIENT})")
        if args.check_determinism or first:
            k = min(8, n)
            again = encode_tiles(enc, paths[:k], args.batch_size)
            diff = float(np.abs(again - feats[:k]).max())
            tol = 1e-3 * max(1.0, float(np.abs(feats[:k]).max()))
            say(f"    determinism check: max difference on {k} re-encoded tiles = {diff:.2e}  ({'ok' if diff <= tol else 'TOO LARGE'})")
            if diff > tol:
                problems.append(f"re-encoding differs by {diff:.3g}")
        if float(feats.std(axis=0).max()) < 1e-5:
            problems.append("all tiles produce (nearly) identical features - check the images / encoder")
        if problems:
            failed.append((pid, "; ".join(problems)))
            say(f"[{i}/{len(patients)}] {pid}: FAILED CHECKS - {failed[-1][1]} (nothing saved)")
            continue

        save_patient(out, feats, g["file_name"].tolist(), g["x"].fillna(-1).astype(int).tolist(),
                     g["y"].fillna(-1).astype(int).tolist(), pid, enc.name)
        done.append(pid)
        times.append(secs)
        say(f"[{i}/{len(patients)}] {pid}: {n} tiles -> {feats.shape}  in {secs:.1f}s  "
            f"(feature mean {feats.mean():.3f}, std {feats.std():.3f})")
        if args.montage and first:
            mp = feat_dir / f"montage_{pid}.png"
            make_montage(paths, mp)
            say(f"    montage saved: {mp}")
        first = False

    # ---- manifest + summary --------------------------------------------------------
    man = rebuild_manifest(feat_dir, enc.name)
    man_path = feat_dir / "feature_manifest.csv"
    man.to_csv(man_path, index=False, encoding="utf-8-sig")

    say("\n" + "=" * 70)
    say("SUMMARY")
    say("=" * 70)
    say(f"Newly extracted : {len(done)}")
    say(f"Skipped (valid) : {len(skipped)}")
    say(f"Failed          : {len(failed)}")
    for pid, why in failed:
        say(f"   FAILED {pid}: {why}")
    if times:
        mean_t = float(np.mean(times))
        say(f"Time per patient: {mean_t:.1f}s on {enc.device}  ->  about {mean_t * len(all_patients) / 60:.1f} min for all {len(all_patients)} patients")
    say(f"Feature files on disk: {len(man)} of {len(all_patients)} patients  (manifest: {man_path})")
    if len(man):
        bad = man[(man.n_tiles != EXPECTED_TILES_PER_PATIENT) | (~man.all_finite)]
        say(f"Files with tile count != {EXPECTED_TILES_PER_PATIENT} or non-finite values: {len(bad)}")
    complete = len(man) == len(all_patients) and not failed
    if failed:
        say("\nRESULT: SOME PATIENTS FAILED - fix them before training")
        return 1
    if args.limit or args.patients:
        say("\nRESULT: TEST RUN OK. Next: look at the montage, then run without --limit for all patients.")
    else:
        nxt = "python train_basic_mil_uni.py"
        say("\nRESULT: " + (f"ALL PATIENTS EXTRACTED - next: {nxt}" if complete else "INCOMPLETE - rerun to resume"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
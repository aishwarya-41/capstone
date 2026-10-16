#!/usr/bin/env python3
"""
extract_features_conch.py   (run in the `conch` environment)
============================================================
CONCH counterpart of extract_features.py: turns every H&E tile into ONE 512-number vector with the frozen,
pretrained official CONCH vision encoder, and saves one file per patient in  features\\conch\\.

NO TRAINING and NO LABELS are used. The encoder is never changed.

Why a separate script (and not an option inside extract_features.py)?
  CONCH needs its own package, its own image preprocessing (resize to 448, CLIP-style mean/std) and its own
  environment (`conch`). Your working ResNet50 script and features\\resnet50 are not touched.
  Helpers that are identical (saving a patient file, rebuilding the manifest) are IMPORTED from extract_features.py.

Same file layout as the ResNet50 features, so train_basic_mil.py works with  --features-dir features\\conch :
  features\\conch\\<PATIENT>.npz   features (100 x 512), file_names, x, y, patient, encoder
  features\\conch\\feature_manifest.csv
  features\\conch\\extraction_info.json     model snapshot, checkpoint hash, preprocessing, versions
  features\\conch\\extraction_failures_<time>.csv   only if something failed
Tile order inside every file = tiles sorted by file name (identical to the ResNet50 files).

What CONCH is told to do (official README):
  create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=<pytorch_model.bin from MahmoodLab/CONCH>)
  preprocess(image)  ->  Resize 448 (bicubic) -> CenterCrop 448 -> RGB -> ToTensor -> Normalize(CLIP mean/std)
  model.encode_image(x, proj_contrast=False, normalize=False)   # raw embedding for MIL, 512-d
  (your 256x256 tiles are enlarged to 448x448 by the official transform; nothing is cut off because the
   tile is square)

SAFETY
  * resumable: a patient is skipped only if its file loads, has the right shape, finite values AND the same
    tile names/order as the inventory
  * refuses to continue into a folder that was made with a different model snapshot / preprocessing
    (unless --overwrite), so features of two settings can never be mixed
  * any unreadable tile / error is recorded with the patient and tile name; that patient is NOT saved

RECOMMENDED ORDER
  1) speed test, writes nothing:      python extract_features_conch.py --benchmark
  2) one-patient test (100 tiles):    python extract_features_conch.py --limit 1
  3) all patients (resumable):        python extract_features_conch.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")      # harmless Windows cache notice
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import numpy as np
import pandas as pd
from PIL import Image

import extract_features as ef          # reuse: save_patient, rebuild_manifest, open_tile, paths, constants

BASE = ef.BASE
INVENTORY_CSV = ef.INVENTORY_CSV
TILE_DIR = ef.TILE_DIR
FEATURES_ROOT = ef.FEATURES_ROOT
HF_REPO = "MahmoodLab/CONCH"
MODEL_CFG = "conch_ViT-B-16"
EXPECTED_DIM = 512                      # measured in conch_smoke_test.py (raw embedding after attention pooling + LayerNorm)
CALL = "encode_image(proj_contrast=False, normalize=False)"
say = ef.say


class ConchEncoder:
    name = "conch"

    def __init__(self, device: str = "cpu", threads: int | None = None):
        import torch
        import torchvision
        from huggingface_hub import hf_hub_download
        from conch.open_clip_custom import create_model_from_pretrained

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.device = device
        try:                                                    # use the cached copy; no network needed
            ckpt = Path(hf_hub_download(HF_REPO, filename="pytorch_model.bin", local_files_only=True))
        except Exception:                                       # noqa: BLE001  (first time on a new machine)
            ckpt = Path(hf_hub_download(HF_REPO, filename="pytorch_model.bin"))
        sha = hashlib.sha256()
        with open(ckpt, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                sha.update(chunk)
        self.model, self.preprocess = create_model_from_pretrained(MODEL_CFG, checkpoint_path=str(ckpt), device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)                             # frozen: no training, ever
        self.dim = EXPECTED_DIM
        self.info = dict(hf_repo=HF_REPO, hf_snapshot=ckpt.parent.name, checkpoint_sha256=sha.hexdigest(), model_cfg=MODEL_CFG,
                         call=CALL, expected_dim=EXPECTED_DIM, image_size=list(self.model.visual.image_size),
                         image_mean=list(self.model.visual.image_mean), image_std=list(self.model.visual.image_std),
                         preprocess=str(self.preprocess), device=device, torch=torch.__version__, torchvision=torchvision.__version__,
                         numpy=np.__version__, python=platform.python_version())
        self.description = (f"CONCH ViT-B/16 (MahmoodLab/CONCH, snapshot {ckpt.parent.name[:8]}), input {tuple(self.model.visual.image_size)}, "
                            f"device={device}, threads={torch.get_num_threads()}, torch={torch.__version__}")

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with torch.inference_mode():
            f = self.model.encode_image(x, proj_contrast=False, normalize=False)
        return f.detach().cpu().numpy().astype(np.float32)


def encode_images(enc: ConchEncoder, images: list[Image.Image], batch_size: int) -> np.ndarray:
    return np.concatenate([enc.encode(images[i:i + batch_size]) for i in range(0, len(images), batch_size)], axis=0)


def existing_ok(path: Path, pid: str, names: list[str], dim: int) -> bool:
    """A previous file is accepted only if it loads, has the right shape/values AND the same tiles in the same order."""
    try:
        with np.load(path, allow_pickle=False) as z:
            f = z["features"]
            return (f.shape == (len(names), dim) and bool(np.isfinite(f).all()) and [str(n) for n in z["file_names"]] == names
                    and str(z["patient"]) == pid and str(z["encoder"]) == "conch")
    except Exception:  # noqa: BLE001
        return False


def benchmark(enc: ConchEncoder, inv: pd.DataFrame, tile_dir: Path, n_tiles: int,
              batch_sizes: tuple[int, ...] = (8, 16, 32), passes: int = 1) -> None:
    """Several passes per batch size: a speed that gets worse from pass to pass means the CPU is throttling (heat / power)."""
    pid = sorted(inv.PATIENT.unique())[0]
    g = inv[inv.PATIENT == pid].sort_values("file_name").head(n_tiles)
    imgs = [ef.open_tile(tile_dir / r.tile_folder_name / r.file_name) for r in g.itertuples()]
    say(f"Benchmark on {len(imgs)} tiles of {pid} (nothing is saved). {enc.description}")
    enc.encode(imgs[:4])                                            # warm-up (first call is always slower)
    say("  warm-up done")
    for bs in batch_sizes:
        for ps in range(1, passes + 1):
            t0 = time.time()
            out = encode_images(enc, imgs, bs)
            s = (time.time() - t0) / len(imgs)
            say(f"  batch size {bs:>2}, pass {ps}/{passes}: {s:.2f} s/tile  -> 10,000 tiles ~ {s * 10000 / 3600:.1f} h   "
                f"(output {out.shape}, finite {bool(np.isfinite(out).all())})  [{dt.datetime.now():%H:%M:%S}]")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract frozen CONCH features for every tile (no training).")
    ap.add_argument("--inventory", type=Path, default=INVENTORY_CSV)
    ap.add_argument("--tile-dir", type=Path, default=TILE_DIR)
    ap.add_argument("--out-root", type=Path, default=FEATURES_ROOT, help="features are written to <out-root>/conch/")
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N patients (1 = the one-patient test)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch (default: PyTorch's choice)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--overwrite", action="store_true", help="redo patients with a valid file / ignore a settings mismatch")
    ap.add_argument("--benchmark", action="store_true", help="time the encoder on a few tiles and exit; writes nothing")
    ap.add_argument("--benchmark-tiles", type=int, default=48)
    ap.add_argument("--benchmark-batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--benchmark-passes", type=int, default=1, help="repeat each timing; a rising s/tile = throttling")
    args = ap.parse_args()

    say("=" * 70 + "\nCONCH FEATURE EXTRACTION  (frozen encoder, no training, no labels)\n" + "=" * 70)
    for p, what in ((args.inventory, "tile_inventory.csv"), (args.tile_dir, "tile folder")):
        if not p.exists():
            say(f"ERROR: {what} not found: {p}")
            return 2
    inv = pd.read_csv(args.inventory, encoding="utf-8-sig", dtype={"PATIENT": str, "tile_folder_name": str, "file_name": str})
    need = {"PATIENT", "tile_folder_name", "file_name", "x", "y", "readable"}
    if not need <= set(inv.columns):
        say(f"ERROR: tile_inventory.csv is missing columns {sorted(need - set(inv.columns))}")
        return 2
    n_unreadable = int((~inv["readable"].astype(bool)).sum())
    if n_unreadable:
        say(f"WARNING: {n_unreadable} tile(s) are marked unreadable in the inventory; they are skipped and the patient is reported.")
    all_patients = sorted(inv.PATIENT.unique())
    patients = sorted(args.patients) if args.patients else all_patients
    unknown = [p for p in patients if p not in all_patients]
    if unknown:
        say(f"ERROR: unknown patient ids: {unknown}")
        return 2
    if args.limit:
        patients = patients[:args.limit]

    enc = ConchEncoder(args.device, args.threads)
    say(f"Encoder   : {enc.description}")
    if args.benchmark:
        benchmark(enc, inv, args.tile_dir, args.benchmark_tiles, tuple(args.benchmark_batch_sizes), args.benchmark_passes)
        return 0

    feat_dir = args.out_root / enc.name
    feat_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    info_path = feat_dir / "extraction_info.json"
    if info_path.is_file():                                   # never mix settings inside one feature folder
        old = json.loads(info_path.read_text(encoding="utf-8"))
        keys = ("hf_snapshot", "checkpoint_sha256", "call", "image_size", "image_mean", "image_std", "expected_dim")
        diff = [k for k in keys if old.get(k) != enc.info.get(k)]
        if diff and not args.overwrite:
            say(f"ERROR: {feat_dir} was made with different settings ({diff}). Use a new --out-root, or --overwrite to redo everything.")
            return 2
    info_path.write_text(json.dumps(dict(enc.info, first_written=stamp, tile_order="sorted by file_name", batch_size=args.batch_size), indent=2), encoding="utf-8")
    say(f"Output    : {feat_dir}\nPatients  : {len(patients)} selected of {len(all_patients)}\n")

    done, skipped, failed, times = [], [], [], []
    first = True
    t_start = time.time()
    for i, pid in enumerate(patients, start=1):
        g = inv[(inv.PATIENT == pid) & (inv.readable.astype(bool))].sort_values("file_name")
        names, n = g.file_name.tolist(), len(g)
        out = feat_dir / f"{pid}.npz"
        if n == 0:
            failed.append((pid, "", "no readable tiles in inventory"))
            continue
        if out.exists() and not args.overwrite and existing_ok(out, pid, names, enc.dim):
            skipped.append(pid)
            say(f"[{i}/{len(patients)}] {pid}: valid file exists, skipped")
            continue
        folder = g.tile_folder_name.iloc[0]
        paths = [args.tile_dir / folder / f for f in names]
        t0 = time.time()
        imgs, bad = [], []
        for p in paths:                                           # open every tile first so a bad file is named exactly
            try:
                imgs.append(ef.open_tile(p))
            except Exception as e:  # noqa: BLE001
                bad.append((p.name, f"{type(e).__name__}: {e}"))
        if bad:
            for nm, why in bad:
                failed.append((pid, nm, why))
            say(f"[{i}/{len(patients)}] {pid}: FAILED - {len(bad)} unreadable tile(s), e.g. {bad[0][0]} ({bad[0][1]}); nothing saved")
            continue
        try:
            feats = encode_images(enc, imgs, args.batch_size)
        except Exception as e:  # noqa: BLE001
            failed.append((pid, "", f"{type(e).__name__}: {e}"))
            say(f"[{i}/{len(patients)}] {pid}: FAILED - {failed[-1][2]}")
            continue
        problems = []
        if feats.shape != (n, enc.dim):
            problems.append(f"shape {feats.shape}, expected {(n, enc.dim)}")
        if not np.isfinite(feats).all():
            problems.append("NaN or infinite values")
        if n != ef.EXPECTED_TILES_PER_PATIENT:
            say(f"    note: {pid} has {n} tiles (expected {ef.EXPECTED_TILES_PER_PATIENT})")
        if first:                                                 # determinism check on the first patient only
            k = min(8, n)
            again = encode_images(enc, imgs[:k], args.batch_size)
            diff = float(np.abs(again - feats[:k]).max())
            say(f"    determinism check: max difference on {k} re-encoded tiles = {diff:.2e}  ({'ok' if diff <= 1e-3 else 'TOO LARGE'})")
            if diff > 1e-3:
                problems.append(f"re-encoding differs by {diff:.3g}")
        if float(feats.std(axis=0).max()) < 1e-5:
            problems.append("all tiles produce (nearly) identical features")
        if problems:
            failed.append((pid, "", "; ".join(problems)))
            say(f"[{i}/{len(patients)}] {pid}: FAILED CHECKS - {problems} (nothing saved)")
            continue
        ef.save_patient(out, feats, names, g.x.fillna(-1).astype(int).tolist(), g.y.fillna(-1).astype(int).tolist(), pid, enc.name)
        secs = time.time() - t0
        done.append(pid)
        times.append(secs)
        left = (len(patients) - i) * float(np.mean(times))
        say(f"[{i}/{len(patients)}] {pid}: {n} tiles -> {feats.shape} in {secs:.0f}s ({secs / n:.2f} s/tile) | mean {feats.mean():.3f}, sd {feats.std():.3f}"
            f" | elapsed {(time.time() - t_start) / 60:.0f} min, about {left / 60:.0f} min left")
        first = False

    man = ef.rebuild_manifest(feat_dir, enc.name)
    man.to_csv(feat_dir / "feature_manifest.csv", index=False, encoding="utf-8-sig")
    if failed:
        pd.DataFrame(failed, columns=["PATIENT", "tile", "reason"]).to_csv(feat_dir / f"extraction_failures_{stamp}.csv", index=False)

    say("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    say(f"Newly extracted : {len(done)}\nSkipped (valid) : {len(skipped)}\nFailed          : {len({f[0] for f in failed})} patient(s)")
    for pid, tile, why in failed:
        say(f"   FAILED {pid} {tile}: {why}")
    if times:
        say(f"Time per patient: {np.mean(times):.0f}s -> about {np.mean(times) * len(all_patients) / 3600:.1f} h for all {len(all_patients)} patients")
    say(f"Feature files on disk: {len(man)} of {len(all_patients)} patients (manifest: {feat_dir / 'feature_manifest.csv'})")
    if len(man):
        bad_m = man[(man.n_tiles != ef.EXPECTED_TILES_PER_PATIENT) | (~man.all_finite) | (man.feature_dim != enc.dim)]
        say(f"Files with tile count != {ef.EXPECTED_TILES_PER_PATIENT}, dim != {enc.dim} or non-finite values: {len(bad_m)}")
    complete = len(man) == len(all_patients) and not failed
    if failed:
        say("\nRESULT: SOME PATIENTS FAILED - fix them before training")
        return 1
    if args.limit or args.patients:
        say("\nRESULT: TEST RUN OK. Next: run without --limit for all patients (it resumes where it stopped).")
    else:
        say("\nRESULT: " + ("ALL PATIENTS EXTRACTED" if complete else "INCOMPLETE - rerun to resume"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

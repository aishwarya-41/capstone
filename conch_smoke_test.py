#!/usr/bin/env python3
"""
conch_smoke_test.py   (run in the `conch` environment)
=======================================================
Checks, on a handful of tiles, that the official CONCH model really loads and gives
sensible tile embeddings BEFORE any full feature extraction is started.

It follows the official CONCH README exactly:
    create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=<pytorch_model.bin>)
    preprocess(image).unsqueeze(0)
    model.encode_image(x, proj_contrast=False, normalize=False)   # embeddings for MIL / linear probe

What it checks
  1. versions + CPU/GPU status
  2. checkpoint download (cached Hugging Face login is used; NO token is read, printed or stored here)
  3. model loads; preprocessing settings (size / mean / std) are printed from the model itself
  4. ONE real tile -> preprocess -> embedding: shape, finite, norm
  5. same tile twice gives the same vector (determinism), and one tile alone == the same tile inside a batch
  6. save to .npz and reload -> identical
  7. timing on a small batch -> a rough, measured estimate for all tiles

It writes ONLY to  smoke_tests\\conch\\  and never touches features\\ or any ResNet50 file.
Run:  python conch_smoke_test.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\LENOVO\OneDrive\Documents\capstone")
INVENTORY_CSV = BASE / "er_label_outputs" / "tile_inventory.csv"
OUT_DIR = BASE / "smoke_tests" / "conch"
HF_REPO = "MahmoodLab/CONCH"          # model id on Hugging Face
MODEL_CFG = "conch_ViT-B-16"           # name of the config inside the official repo
N_BENCH = 8                            # tiles used for the timing test (one tile from each of 8 different patients)


def say(msg: str = "") -> None:
    print(msg, flush=True)


def check(ok: bool, text: str) -> bool:
    say(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, default=INVENTORY_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    all_ok = True

    import torch
    import torchvision
    from PIL import Image
    from huggingface_hub import hf_hub_download

    say("=" * 70 + "\nCONCH SMOKE TEST\n" + "=" * 70)
    say(f"python {platform.python_version()} | torch {torch.__version__} | torchvision {torchvision.__version__} | "
        f"numpy {np.__version__} | CUDA available: {torch.cuda.is_available()} | CPU threads: {torch.get_num_threads()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"device used: {device}")

    # ---- 1. pick real tiles from the existing inventory (read-only) --------------------------
    inv = pd.read_csv(args.inventory, encoding="utf-8-sig", dtype={"PATIENT": str})
    inv = inv[inv.readable.astype(bool)]
    first_per_patient = inv.drop_duplicates("PATIENT").head(N_BENCH)
    paths = [Path(p) for p in first_per_patient.full_path]
    patients = first_per_patient.PATIENT.tolist()
    say(f"\nTiles for the test: {len(paths)} (first tile of the first {len(paths)} patients)")
    all_ok &= check(all(p.is_file() for p in paths), "all test tile files exist")

    # ---- 2. checkpoint: download once (cached afterwards); uses your saved HF login ---------------
    say("\nChecking / downloading the checkpoint (about 0.8 GB the first time)...")
    t0 = time.time()
    hf_hub_download(HF_REPO, filename="meta.yaml")
    ckpt = Path(hf_hub_download(HF_REPO, filename="pytorch_model.bin"))
    say(f"  checkpoint file : {ckpt}")
    revision = ckpt.parent.name          # the commit id of the Hugging Face snapshot
    sha = hashlib.sha256()
    with open(ckpt, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            sha.update(chunk)
    say(f"  size {ckpt.stat().st_size / 1e6:.1f} MB | HF snapshot {revision} | sha256 {sha.hexdigest()[:16]}... | {time.time() - t0:.0f}s")
    all_ok &= check(ckpt.stat().st_size > 5e8, "checkpoint file has a plausible size (> 500 MB)")

    # ---- 3. load model with the OFFICIAL function -------------------------------------------------
    from conch.open_clip_custom import create_model_from_pretrained
    t0 = time.time()
    model, preprocess = create_model_from_pretrained(MODEL_CFG, checkpoint_path=str(ckpt), device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    say(f"\nModel loaded in {time.time() - t0:.0f}s | parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    say(f"  vision input size : {model.visual.image_size}")
    say(f"  mean / std        : {model.visual.image_mean} / {model.visual.image_std}")
    say(f"  preprocess        : {preprocess}")
    all_ok &= check(not model.training, "model is in eval mode (dropout etc. off)")

    # ---- 4. one tile -> one embedding -------------------------------------------------------------
    say("\nOne-tile test")
    with Image.open(paths[0]) as im:
        im.load()
        img0 = im.convert("RGB")
    say(f"  tile: {paths[0].name}  | original size {img0.size}")
    x0 = preprocess(img0).unsqueeze(0).to(device)
    say(f"  preprocessed tensor: shape {tuple(x0.shape)}, dtype {x0.dtype}, min {x0.min():.2f}, max {x0.max():.2f}")
    with torch.inference_mode():
        e0 = model.encode_image(x0, proj_contrast=False, normalize=False)
    e0 = e0.detach().cpu().numpy().astype(np.float32)
    say(f"  embedding: shape {e0.shape}, L2 norm {np.linalg.norm(e0):.3f}, mean {e0.mean():.4f}, sd {e0.std():.4f}")
    all_ok &= check(e0.ndim == 2 and e0.shape[0] == 1, "embedding is (1, D)")
    all_ok &= check(bool(np.isfinite(e0).all()), "all embedding values are finite")
    all_ok &= check(float(e0.std()) > 1e-6, "embedding is not constant")
    say(f"  >> embedding dimension D = {e0.shape[1]} (the official README says raw, pre-projection embeddings; record this number)")

    # ---- 5. determinism + batch consistency ---------------------------------------------------------
    with torch.inference_mode():
        e0b = model.encode_image(x0, proj_contrast=False, normalize=False).cpu().numpy()
    say("\nConsistency")
    all_ok &= check(float(np.abs(e0 - e0b).max()) < 1e-5, f"same tile twice -> same vector (max abs diff {np.abs(e0 - e0b).max():.2e})")

    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im.load()
            imgs.append(im.convert("RGB"))
    xb = torch.stack([preprocess(i) for i in imgs]).to(device)
    t0 = time.time()
    with torch.inference_mode():
        eb = model.encode_image(xb, proj_contrast=False, normalize=False).cpu().numpy().astype(np.float32)
    dt_batch = time.time() - t0
    d = float(np.abs(eb[0] - e0[0]).max())
    all_ok &= check(d < 1e-3, f"tile 0 alone == tile 0 inside a batch of {len(imgs)} (max abs diff {d:.2e})")
    all_ok &= check(bool(np.isfinite(eb).all()) and eb.shape == (len(imgs), e0.shape[1]), f"batch output shape {eb.shape}, all finite")
    all_ok &= check(float(np.abs(eb[0] - eb[1]).max()) > 1e-3, "different tiles give different vectors")
    sec_per_tile = dt_batch / len(imgs)
    say(f"  timing: {len(imgs)} tiles in {dt_batch:.1f}s -> {sec_per_tile:.2f} s/tile (first batch, may include warm-up)")
    say(f"  rough ESTIMATE for 10,000 tiles: {sec_per_tile * 10000 / 3600:.1f} hours (to be confirmed on a longer benchmark)")

    # ---- 6. save + reload ---------------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_dir / f"conch_smoke_{stamp}.npz"        # timestamped: never overwrites an earlier smoke test
    np.savez_compressed(out, features=eb, file_names=np.array([p.name for p in paths]), patient=np.array(patients),
                        encoder=np.array("conch"), hf_snapshot=np.array(revision))
    with np.load(out, allow_pickle=False) as z:
        same = bool(np.array_equal(z["features"], eb)) and z["features"].dtype == np.float32
        names_ok = [str(n) for n in z["file_names"]] == [p.name for p in paths]
    say("\nSerialization")
    all_ok &= check(same and names_ok, f"saved and reloaded identically: {out.name}")

    meta = dict(timestamp=stamp, python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                numpy=np.__version__, device=device, hf_repo=HF_REPO, hf_snapshot=revision, checkpoint_sha256=sha.hexdigest(),
                model_cfg=MODEL_CFG, image_size=list(model.visual.image_size), image_mean=list(model.visual.image_mean),
                image_std=list(model.visual.image_std), preprocess=str(preprocess), call="encode_image(proj_contrast=False, normalize=False)",
                embedding_dim=int(e0.shape[1]), sec_per_tile_first_batch=sec_per_tile, tiles=[p.name for p in paths], all_checks_passed=bool(all_ok))
    (args.out_dir / f"conch_smoke_{stamp}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    say(f"  settings written to conch_smoke_{stamp}.json")

    say("\n" + "=" * 70)
    say("SMOKE TEST RESULT: " + ("PASSED" if all_ok else "FAILED - do not start full extraction; send me this output"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

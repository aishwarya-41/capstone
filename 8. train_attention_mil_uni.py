#!/usr/bin/env python3
"""
train_attention_mil_uni.py   (Step 7 of the ER pipeline - ATTENTION MIL on UNI features)
=========================================================================================
Attention-based Multiple-Instance Learning (gated attention, Ilse et al. 2018) for
ER-positive vs ER-negative prediction, on the 15 patient-level splits, using the frozen UNI
features from extract_features_uni.py.

THE ONE IDEA
  Basic MIL (train_basic_mil_uni.py) combined a patient's ~100 tile vectors with a FIXED rule:
  mean (every tile counts the same) or max (only the strongest value counts).
  Attention MIL LEARNS a weight for every tile (weights are >= 0 and add up to 1 per patient):

      tile vector -> small trainable layer (1024 -> 128, ReLU, dropout)
      attention score of each tile  a_i = w . ( tanh(V h_i) * sigmoid(U h_i) )     (the "gate")
      weights                       alpha_i = softmax over the patient's tiles
      patient vector                z = sum_i alpha_i * h_i
      prediction                    P(ER-positive) = sigmoid( linear(z) )

  Only the patient's ER label is used for training. Nobody tells the model which tiles matter.

THIS SCRIPT RUNS FIVE METHODS ON THE SAME 15 SPLITS (so the numbers are directly comparable)
  majority    always predicts ER-positive            (sanity floor)
  logreg      logistic regression on mean features    (simple baseline)
  mil_mean    basic MIL, mean pooling
  mil_max     basic MIL, max pooling
  mil_attn    ATTENTION MIL   <- new

  The first four are exactly the code and settings used by train_basic_mil_uni.py, with the same
  random seeds, so they should reproduce your earlier UNI numbers.
  Attention MIL is only interesting if it clearly beats LogReg and the basic MIL models.

LEAKAGE GUARDS (same as before)
  * features are scaled with TRAIN patients only; early stopping uses VALIDATION loss only;
  * TEST patients are used once, for the final numbers; threshold fixed at 0.5;
  * the script re-checks that train / val / test never overlap and that the split file is unchanged.

EXTRA OUTPUTS FOR ATTENTION (in <out-dir>)
  attention_weights_test.csv   for every test patient in every split: the weight of each tile
                               (with the tile file name and x/y position)
  attention_summary.csv        per patient: how concentrated the weights are
                               (eff_n_tiles = effective number of tiles used; 100 would mean
                                "all tiles equal", i.e. no better than mean pooling)
  attention_top_tiles.png      the most- and least-attended tiles of four example patients

  CAUTION: attention shows which tiles the MODEL leaned on, not proof of which tissue causes ER
  status. Treat the picture as a hypothesis to show a pathologist, not as a finding.

SETUP: put this file in the SAME folder as train_basic_mil_uni.py (it re-uses that file's code).

RUN
  python train_attention_mil_uni.py --selftest          # 1) check the code works (seconds, no data)
  python train_attention_mil_uni.py --max-splits 1      # 2) quick real test on 1 split
  python train_attention_mil_uni.py                     # 3) full run, all 15 splits
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import train_basic_mil_uni as B          # re-use the loaders, metrics, plots and baselines
except ImportError:
    print("ERROR: train_basic_mil_uni.py must be in the SAME folder as this script.")
    raise SystemExit(2)

say, PASS, WARN, FAIL, INFO = B.say, B.PASS, B.WARN, B.FAIL, B.INFO
ALL_METHODS = ["majority", "logreg", "mil_mean", "mil_max", "mil_attn"]
B.METHOD_LABELS["mil_attn"] = "Attention MIL (gated)"

PHASE1 = B.BASE
DEFAULT_FEATURES = PHASE1 / "features" / "uni"
DEFAULT_OUT = PHASE1 / "mil_results" / "attention_mil_uni"
DEFAULT_INVENTORY = PHASE1 / "er_label_outputs" / "tile_inventory.csv"
DEFAULT_TILE_DIR = PHASE1.parents[1] / "Dataset" / "BLOCKS_NORM_MACENKO"


# =============================================================================
# MODEL
# =============================================================================
def build_attention_model(in_dim: int, hidden: int, attn_dim: int, dropout: float):
    import torch
    import torch.nn as nn

    class GatedAttentionMIL(nn.Module):
        """tile features -> trainable layer -> LEARNED tile weights -> weighted sum -> linear -> logit."""

        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.att_v = nn.Linear(hidden, attn_dim)      # "content" branch (tanh)
            self.att_u = nn.Linear(hidden, attn_dim)      # "gate" branch (sigmoid)
            self.att_w = nn.Linear(attn_dim, 1)           # one score per tile
            self.classifier = nn.Linear(hidden, 1)

        def forward(self, x, mask, return_attention: bool = False):   # x: B x N x D, mask: B x N (True = real tile)
            h = self.embed(x)                                           # B x N x H
            score = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)   # B x N
            score = score.masked_fill(~mask, float("-inf"))             # padded (non-existent) tiles get no weight
            alpha = torch.softmax(score, dim=1)                         # B x N, sums to 1 over the real tiles
            z = (alpha.unsqueeze(-1) * h).sum(dim=1)                    # B x H
            logit = self.classifier(z).squeeze(-1)                      # B
            return (logit, alpha) if return_attention else logit

    return GatedAttentionMIL()


def run_attention(sd: dict, args, seed: int) -> dict:
    """Train one attention-MIL model on one split. Returns val/test probabilities + test attention weights."""
    import torch
    import torch.nn as nn

    device = torch.device(args.device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    bags = sd["bags"]
    mu, sg = B.fit_scaler(bags, sd["train_ids"])         # scaling fitted on TRAIN patients only

    def tensors(ids):
        x, m = B.pad_bags([(bags[p] - mu) / sg for p in ids])
        return torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)

    Xtr, Mtr = tensors(sd["train_ids"])
    Xva, Mva = tensors(sd["val_ids"])
    Xte, Mte = tensors(sd["test_ids"])
    ytr = torch.tensor(sd["y_train"], dtype=torch.float32, device=device)
    yva = torch.tensor(sd["y_val"], dtype=torch.float32, device=device)

    n_pos = float(ytr.sum().item())
    n_neg = float(len(ytr) - n_pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1.0)], device=device))

    model = build_attention_model(Xtr.shape[2], args.hidden, args.attn_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(seed)

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr), generator=gen)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size].to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx], Mtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xva, Mva), yva).item())          # VALIDATION loss decides when to stop
        if vloss < best_loss - 1e-4:
            best_loss, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(Xva, Mva)).cpu().numpy()
        logit_te, alpha_te = model(Xte, Mte, return_attention=True)
        p_test = torch.sigmoid(logit_te).cpu().numpy()
        alpha_te = alpha_te.cpu().numpy()
    attn = {pid: alpha_te[i, :len(bags[pid])].astype(np.float64) for i, pid in enumerate(sd["test_ids"])}
    eff = float(np.mean([1.0 / np.sum(a ** 2) for a in attn.values()]))
    return dict(p_val=p_val, p_test=p_test, attn=attn,
                info=f"best_epoch={best_epoch}, val_loss={best_loss:.3f}, eff_tiles={eff:.0f}")


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def run_experiment_attn(bags, labels, splits, methods, args, tile_meta=None, verbose=True):
    """One row per (method, split, test patient); plus the attention weights of the attention model."""
    rows, attn_rows = [], []
    split_ids = list(dict.fromkeys(splits["split_id"]))
    t_start = time.time()
    for k, sid in enumerate(split_ids):
        g = splits[splits.split_id == sid]
        sd = B.split_data(bags, labels, g)
        rep, fold = int(g["repeat"].iloc[0]), int(g["fold"].iloc[0])
        for m in methods:
            seed = args.seed + 1000 * k + B.hash_method(m)     # same seeds as train_basic_mil_uni.py
            if m == "majority":
                res = B.run_majority(sd, args, seed)
            elif m == "logreg":
                res = B.run_logreg(sd, args, seed)
            elif m in ("mil_mean", "mil_max"):
                res = B.run_mil(sd, m.split("_")[1], args, seed)
            elif m == "mil_attn":
                res = run_attention(sd, args, seed)
            else:
                raise SystemExit(f"unknown method {m}")
            thr = B.choose_threshold(args.threshold_mode, sd["y_val"], res["p_val"])
            mt = B.compute_metrics(sd["y_test"], res["p_test"], thr)
            if verbose:
                auc = "n/a" if np.isnan(mt["AUC"]) else f"{mt['AUC']:.3f}"
                say(f"  [{k + 1}/{len(split_ids)}] {sid}  {m:<9} test AUC={auc}  balAcc={mt['balanced_accuracy']:.3f}  ({res['info']})  [{time.time() - t_start:.0f}s]")
            for pid, yy, pp in zip(sd["test_ids"], sd["y_test"], res["p_test"]):
                rows.append(dict(method=m, split_id=sid, repeat=rep, fold=fold, PATIENT=pid, ER_label=int(yy),
                                 prob_ERpos=float(pp), threshold=thr, pred=int(pp >= thr)))
                if m == "mil_attn":
                    a = res["attn"][pid]
                    names, xs, ys = (tile_meta[pid] if tile_meta and pid in tile_meta else ([""] * len(a), [-1] * len(a), [-1] * len(a)))
                    for ti in range(len(a)):
                        attn_rows.append((sid, rep, pid, int(yy), float(pp), ti, names[ti], int(xs[ti]), int(ys[ti]), float(a[ti])))
    attn_df = pd.DataFrame(attn_rows, columns=["split_id", "repeat", "PATIENT", "ER_label", "prob_ERpos",
                                               "tile_index", "file_name", "x", "y", "attention"])
    return pd.DataFrame(rows), attn_df


def summarize_attention(attn_df: pd.DataFrame) -> pd.DataFrame:
    """Per patient and split: how concentrated are the weights?"""
    out = []
    for (sid, pid), g in attn_df.groupby(["split_id", "PATIENT"], sort=False):
        a = np.sort(g.attention.values)[::-1]
        out.append(dict(split_id=sid, repeat=int(g.repeat.iloc[0]), PATIENT=pid, ER_label=int(g.ER_label.iloc[0]),
                        prob_ERpos=float(g.prob_ERpos.iloc[0]), n_tiles=len(a), max_weight=float(a[0]),
                        top10_weight_share=float(a[:10].sum()), eff_n_tiles=float(1.0 / np.sum(a ** 2)),
                        uniform_weight=1.0 / len(a)))
    return pd.DataFrame(out)


def load_tile_meta(features_dir: Path, patients: list[str]) -> dict:
    meta = {}
    for pid in patients:
        with np.load(features_dir / f"{pid}.npz", allow_pickle=False) as z:
            n = z["features"].shape[0]
            if {"file_names", "x", "y"} <= set(z.files):
                meta[pid] = (list(z["file_names"]), z["x"].tolist(), z["y"].tolist())
            else:                                           # files made by other tools: weights are still saved, without tile names
                meta[pid] = ([""] * n, [-1] * n, [-1] * n)
    return meta


# =============================================================================
# FIGURE: most / least attended tiles of four example patients
# =============================================================================
def make_tile_figure(attn_df, preds, inventory: Path, tile_dir: Path, out_dir: Path, stamp: str, k: int = 8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    inv = pd.read_csv(inventory, encoding="utf-8-sig", dtype={"PATIENT": str, "tile_folder_name": str, "file_name": str})
    folder = inv.drop_duplicates("PATIENT").set_index("PATIENT").tile_folder_name
    p = preds[(preds.method == "mil_attn") & (preds.repeat == 1)]
    neg, pos = p[p.ER_label == 0], p[p.ER_label == 1]
    picks = []

    def pick(df, ascending, label):
        if len(df):
            picks.append((df.sort_values("prob_ERpos", ascending=ascending).iloc[0], label))

    pick(neg[neg.pred == 0], True, "ER-NEGATIVE, correctly called")
    pick(pos[pos.pred == 1], False, "ER-POSITIVE, correctly called")
    pick(neg[neg.pred == 1], False, "ER-NEGATIVE, wrongly called ER+")
    pick(pos[pos.pred == 0], True, "ER-POSITIVE, wrongly called ER-")
    if not picks:
        return None
    n = len(picks)
    fig, axes = plt.subplots(3 * n, k, figsize=(1.35 * k, 1.75 * 2 * n + 0.35 * n + 0.5),
                             gridspec_kw=dict(height_ratios=[0.28, 1, 1] * n))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for r, (row, label) in enumerate(picks):
        g = attn_df[(attn_df.PATIENT == row.PATIENT) & (attn_df.repeat == 1)].sort_values("attention", ascending=False)
        axes[3 * r, 0].text(0, 0.2, f"{row.PATIENT}: {label}    P(ER+) = {row.prob_ERpos:.2f}", fontsize=9, fontweight="bold",
                            transform=axes[3 * r, 0].transAxes, va="bottom", ha="left", clip_on=False)
        for part, (sub, nm) in enumerate(((g.head(k), "MOST attended"), (g.tail(k).iloc[::-1], "LEAST attended"))):
            for c in range(k):
                ax = axes[3 * r + 1 + part, c]
                if c < len(sub):
                    t = sub.iloc[c]
                    try:
                        ax.imshow(Image.open(tile_dir / folder[row.PATIENT] / t.file_name).convert("RGB"))
                    except Exception:  # noqa: BLE001
                        ax.text(0.5, 0.5, "tile\nnot found", ha="center", va="center", fontsize=6)
                    ax.set_title(f"{t.attention * 100:.1f}%", fontsize=7, pad=2)
            axes[3 * r + 1 + part, 0].text(-0.1, 0.5, nm, transform=axes[3 * r + 1 + part, 0].transAxes, rotation=90,
                                           ha="right", va="center", fontsize=7)
    fig.suptitle("Attention MIL: share of the patient's total attention given to each tile (uniform would be 1.0% per tile)",
                 fontsize=9, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.955, bottom=0.01, hspace=0.35, wspace=0.06)
    path = out_dir / "attention_top_tiles.png"
    B._backup(path, stamp)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# =============================================================================
# SELF-TEST (synthetic data, no files needed)
# =============================================================================
def selftest(args) -> int:
    from sklearn.model_selection import StratifiedKFold, train_test_split
    say("=" * 70)
    say("SELF-TEST on synthetic data (no files are read or written)")
    say("=" * 70)
    rng = np.random.default_rng(123)
    n = 100
    ids = [f"SYN-{i:03d}" for i in range(n)]
    y = np.array([1] * 76 + [0] * 24)
    rng.shuffle(y)
    labels = dict(zip(ids, y.tolist()))
    rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=1)
    for f, (pool, test) in enumerate(skf.split(ids, y), start=1):
        tr, va = train_test_split(pool, test_size=0.15, stratify=y[pool], random_state=f)
        for role, idx in (("train", tr), ("val", va), ("test", test)):
            rows += [(f"r1_f{f}", 1, f, ids[i], int(y[i]), role) for i in idx]
    splits = pd.DataFrame(rows, columns=["split_id", "repeat", "fold", "PATIENT", "ER_label", "role"])
    st = copy.copy(args)
    st.epochs, st.patience, st.hidden, st.attn_dim, st.batch_size = 40, 8, 32, 16, 16
    methods = ["mil_mean", "mil_attn"]
    ok = True
    for scenario in ("SIGNAL", "NOISE"):
        bags, planted = {}, {}
        for pid, yy in labels.items():
            a = rng.normal(size=(100, 64)).astype(np.float32)
            if scenario == "SIGNAL" and yy == 0:                  # only 5 of 100 tiles of an ER-negative patient carry the pattern
                k = rng.choice(100, 5, replace=False)
                a[k, :5] += 3.0
                planted[pid] = k
            bags[pid] = a
        say(f"\n--- scenario {scenario}: " + ("a pattern in just 5% of the tiles of ER-negative patients -> attention should find those tiles"
                                              if scenario == "SIGNAL" else "pure random noise -> AUC must be about 0.5 (otherwise something leaks)"))
        preds, attn = run_experiment_attn(bags, labels, splits, methods, st, verbose=False)
        res = {m: float(np.mean([roc_auc_score(g.ER_label, g.prob_ERpos) for _, g in d.groupby("split_id")])) for m, d in preds.groupby("method")}
        for m, v in res.items():
            say(f"    {m:<9} mean test AUC over 5 splits = {v:.3f}")
        sums = attn.groupby(["split_id", "PATIENT"]).attention.sum()
        good = bool(np.allclose(sums.values, 1.0, atol=1e-4))
        ok &= good
        say(f"    [{'PASS' if good else 'FAIL'}] attention weights of every patient add up to 1")
        if scenario == "SIGNAL":
            good = res["mil_attn"] >= 0.75
            ok &= good
            say(f"    [{'PASS' if good else 'FAIL'}] attention MIL finds the planted pattern (AUC >= 0.75)")
            ratios = []
            for pid, k in planted.items():
                a = attn[attn.PATIENT == pid]
                if len(a):
                    a = a.groupby("tile_index").attention.mean()
                    ratios.append(float(a.reindex(k).mean() / (1.0 / 100)))
            r = float(np.mean(ratios)) if ratios else float("nan")
            good = r >= 1.2
            ok &= good
            say(f"    [{'PASS' if good else 'FAIL'}] planted tiles get more attention than average: {r:.1f}x the uniform weight (needs >= 1.2x)")
        else:
            good = 0.15 <= res["mil_attn"] <= 0.70
            ok &= good
            say(f"    [{'PASS' if good else 'FAIL'}] attention MIL is not above chance on noise (0.15 <= AUC <= 0.70)")
    say("\nSELF-TEST RESULT: " + ("PASSED - the attention code works" if ok else "FAILED - do not trust results; send me this output"))
    return 0 if ok else 1


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Attention MIL on UNI features (+ the same baselines) on the 15 patient-level splits.")
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--splits-csv", type=Path, default=B.SPLITS_CSV)
    ap.add_argument("--labels-csv", type=Path, default=B.LABELS_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY, help="tile_inventory.csv (only for the example-tiles figure)")
    ap.add_argument("--tile-dir", type=Path, default=DEFAULT_TILE_DIR, help="BLOCKS_NORM_MACENKO folder (only for the example-tiles figure)")
    ap.add_argument("--no-tile-figure", action="store_true", help="skip attention_top_tiles.png")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--max-splits", type=int, default=None, help="quick test: only the first N splits (pooled metrics are skipped)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--attn-dim", type=int, default=64, help="size of the attention scoring layers")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threshold-mode", choices=["fixed", "val_youden"], default="fixed")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true", help="run the synthetic-data self-test and exit")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        say("ERROR: PyTorch is not installed. Install it with:  python -m pip install torch torchvision matplotlib")
        return 2
    if args.selftest:
        return selftest(args)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = B.Report()
    say("=" * 70)
    say("ATTENTION MIL FOR ER PREDICTION  (frozen UNI features, patient-level CV)")
    say("=" * 70)
    say(f"Features : {args.features_dir}\nSplits   : {args.splits_csv}\nLabels   : {args.labels_csv}\nOutput   : {args.out_dir}")
    loaded = B.load_inputs(args, rep)
    if loaded is None:
        say("\nInput checks FAILED:")
        for r in rep.rows:
            if r["status"] == FAIL:
                say(f"  [FAIL] {r['check']}  observed={r['observed']}  {r['detail']}")
        return 2
    splits, labels, bags, patients = loaded
    for r in rep.rows:
        say(f"  [{r['status']}] {r['check']}  {r['observed']}")
    n_pos = sum(labels[p] for p in patients)
    say(f"Patients : {len(patients)}  (ER+ {n_pos}, ER- {len(patients) - n_pos}); feature dim {next(iter(bags.values())).shape[1]}")
    tile_meta = load_tile_meta(args.features_dir, patients)

    if args.max_splits:
        keep = list(dict.fromkeys(splits.split_id))[:args.max_splits]
        splits = splits[splits.split_id.isin(keep)]
        say(f"QUICK TEST: only splits {keep}")
    for k, v in dict(encoder="uni", methods=args.methods, hidden=args.hidden, attn_dim=args.attn_dim, dropout=args.dropout, lr=args.lr,
                     weight_decay=args.weight_decay, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                     threshold_mode=args.threshold_mode, device=args.device, seed=args.seed).items():
        rep.add(f"setting: {k}", INFO, observed=str(v))

    say("\nTraining / evaluating (test set is only used for the numbers shown):")
    t0 = time.time()
    preds, attn_df = run_experiment_attn(bags, labels, splits, args.methods, args, tile_meta)
    say(f"\nFinished in {(time.time() - t0) / 60:.1f} min")

    per_split, per_repeat, ens, ens_pred = B.aggregate(preds, patients, args.seed)
    summary = B.build_summary(per_split, per_repeat, ens)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    say("\nWriting output files...")
    B.write_csv(summary, args.out_dir / "summary_metrics.csv", stamp)
    B.write_csv(per_split, args.out_dir / "per_split_metrics.csv", stamp)
    B.write_csv(preds, args.out_dir / "predictions_test.csv", stamp)
    plots = B.make_plots(per_split, ens_pred, ens, args.out_dir, stamp)
    asum = pd.DataFrame()
    if len(attn_df):
        B.write_csv(attn_df, args.out_dir / "attention_weights_test.csv", stamp)
        asum = summarize_attention(attn_df)
        B.write_csv(asum, args.out_dir / "attention_summary.csv", stamp)
        if not args.no_tile_figure:
            try:
                if args.inventory.is_file() and args.tile_dir.is_dir():
                    p = make_tile_figure(attn_df, preds, args.inventory, args.tile_dir, args.out_dir, stamp)
                    if p:
                        plots.append(p)
                else:
                    say("  (example-tiles figure skipped: tile_inventory.csv or the tile folder was not found; pass --inventory / --tile-dir)")
            except Exception as e:  # noqa: BLE001
                say(f"  (example-tiles figure skipped: {type(e).__name__}: {e})")
    complete = bool(len(ens))
    rep.add("every patient tested exactly once per repeat (pooled metrics possible)", PASS if complete else WARN,
            detail="" if complete else "Not all folds were run (quick test), so pooled/ensemble metrics were skipped.")
    B.write_csv(rep.df(), args.out_dir / "mil_run_report.csv", stamp)
    say(f"  saved to: {args.out_dir.resolve()}")

    # ---- console summary ---------------------------------------------------------
    say("\n" + "=" * 70)
    say("RESULTS  (positive class = ER-positive; every number is on patients NOT used for training)")
    say("=" * 70)
    say(f"{'method':<26}{'AUC (mean±SD over splits)':<28}{'balAcc':<8}{'F1 ER+':<8}{'F1 ER-':<8}")
    for _, r in summary.iterrows():
        say(f"{r['label']:<26}{r.AUC_mean_over_splits:.3f} ± {r.AUC_sd_over_splits:.3f}{'':<12}{r.balAcc_mean_over_splits:<8.3f}{r.F1_ERpos_mean_over_splits:<8.3f}{r.F1_ERneg_mean_over_splits:<8.3f}")
    if complete:
        say("\nOne out-of-fold prediction per patient (probabilities averaged over the repeats):")
        say(f"{'method':<26}{'AUC [95% CI]':<26}{'F1 ER+':<8}{'F1 ER-':<8}{'sens':<7}{'spec':<7}  confusion (TN FP / FN TP)")
        for _, r in summary.iterrows():
            say(f"{r['label']:<26}{r.ens_AUC:.3f} [{r.ens_AUC_CI95_low:.2f}-{r.ens_AUC_CI95_high:.2f}]{'':<5}{r.ens_F1_ERpos:<8.3f}{r.ens_F1_ERneg:<8.3f}"
                f"{r.ens_sensitivity_ERpos:<7.2f}{r.ens_specificity_ERneg:<7.2f}  {int(r.ens_TN)} {int(r.ens_FP)} / {int(r.ens_FN)} {int(r.ens_TP)}")
    if len(asum):
        say("\nHOW CONCENTRATED IS THE ATTENTION?  (a patient has ~100 tiles; uniform weights would use all of them)")
        say(f"  effective number of tiles used, median over patients/splits:  {asum.eff_n_tiles.median():.0f}"
            f"   (ER-positive {asum[asum.ER_label == 1].eff_n_tiles.median():.0f}, ER-negative {asum[asum.ER_label == 0].eff_n_tiles.median():.0f})")
        say(f"  share of total attention on the 10 top tiles, median:         {asum.top10_weight_share.median() * 100:.0f}%   (uniform would be 10%)")
        if asum.eff_n_tiles.median() > 0.9 * asum.n_tiles.median():
            say("  -> the weights are almost uniform: this attention model behaves like mean pooling, so the pictures carry little information.")
    say("\nHOW TO READ THIS")
    say("  * Compare 'Attention MIL' with 'LogReg on mean features' and with the basic MIL rows IN THE SAME TABLE.")
    say("  * The 15 splits reuse the same 100 patients: differences of a few hundredths of AUC are within noise.")
    say("  * Attention weights show what the model used, not proof of what causes ER status. Show the tile figure to a pathologist.")
    say(f"\nFiles: {', '.join(p.name for p in plots)}, summary_metrics.csv, per_split_metrics.csv, predictions_test.csv, "
        "attention_weights_test.csv, attention_summary.csv, mil_run_report.csv")
    say("\nRESULT: DONE" if not rep.n_fail() else "\nRESULT: CHECKS FAILED")
    return 0 if not rep.n_fail() else 1


if __name__ == "__main__":
    sys.exit(main())

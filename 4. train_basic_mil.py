#!/usr/bin/env python3
"""
train_basic_mil.py   (Step 4 of the ER pipeline)
================================================
BASIC Multiple-Instance Learning (MIL) for ER-positive vs ER-negative prediction,
evaluated on the 15 patient-level splits made by make_patient_splits.py.

WHAT "BASIC MIL" MEANS HERE
  * One PATIENT = one "bag". The ~100 tiles of the patient are the "instances".
  * Only the patient has a label (ER). Tiles have no label of their own.
  * Each tile's frozen ResNet50 feature (2048 numbers) goes through a small trainable
    layer (2048 -> 128, ReLU, dropout). The tile vectors are then combined with a
    FIXED rule - MEAN or MAX over tiles - and a linear layer gives P(ER-positive).
  * "Fixed rule" is the point: every tile counts the same (mean) or only the strongest
    tile counts (max). The NEXT step (attention MIL) replaces this fixed rule by a
    learned weight per tile.

WHAT IS RUN (same 15 splits for every method, so the numbers are comparable)
  majority   always predicts the more common class (ER-positive). Sanity floor.
  logreg     logistic regression on each patient's AVERAGED tile features. Simple baseline
             that any MIL model has to beat to be worth its complexity.
  mil_mean   basic MIL with mean pooling
  mil_max    basic MIL with max pooling

LEAKAGE GUARDS (built in)
  * Only patients in the split's TRAIN set are used to fit the feature scaling.
  * Early stopping uses the VALIDATION set only.
  * The TEST set is used once, at the end of each split, only to compute metrics.
  * The decision threshold is 0.5 (fixed) or chosen on VALIDATION patients - never test.
  * The script re-checks that train / val / test patients never overlap.

OUTPUTS (new folder; older files are renamed with a timestamp, never overwritten)
  summary_metrics.csv       one row per method: AUC, F1, balanced accuracy, confusion matrix ...
  per_split_metrics.csv     test metrics for each of the 15 splits and method
  predictions_test.csv      every out-of-fold test prediction (patient, probability, ...)
  roc_and_auc.png           ROC curves and the spread of AUC over the 15 splits
  confusion_matrices.png    confusion matrix per method
  mil_run_report.csv        checks (PASS / FAIL) and settings used

FIRST RUN THIS (takes seconds, needs no data files):
    python train_basic_mil.py --selftest
THEN A QUICK REAL TEST (1 split):
    python train_basic_mil.py --max-splits 1
THEN THE FULL RUN:
    python train_basic_mil.py
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             roc_auc_score, roc_curve)

# =============================================================================
# SETTINGS (defaults match your Mac folders; override on the command line)
# =============================================================================
BASE = Path.home() / "Desktop" / "Capstone" / "Implementation" / "Phase1"
FEATURES_DIR = BASE / "features" / "resnet50"
SPLITS_CSV = BASE / "cv_splits" / "patient_cv_splits.csv"
LABELS_CSV = BASE / "er_label_outputs" / "patient_er_labels.csv"
OUT_DIR = BASE / "mil_results" / "basic_mil_resnet50"

EXPECTED_SPLIT_FINGERPRINT = "44265b9eab79c5bf"   # printed by make_patient_splits.py on your machine
PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
METHOD_LABELS = {"majority": "Majority class", "logreg": "LogReg on mean features",
                 "mil_mean": "Basic MIL (mean pool)", "mil_max": "Basic MIL (max pool)"}


def say(msg: str = "") -> None:
    print(msg, flush=True)


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, check, status, observed="", expected="", detail=""):
        self.rows.append(dict(check=check, status=status, expected=expected, observed=observed, detail=detail))

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["check", "status", "expected", "observed", "detail"])

    def n_fail(self) -> int:
        return sum(r["status"] == FAIL for r in self.rows)


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """y: 1 = ER-positive, 0 = ER-negative. p: predicted probability of ER-positive."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    sens = tp / (tp + fn) if (tp + fn) else np.nan       # recall of ER-positive
    spec = tn / (tn + fp) if (tn + fp) else np.nan       # recall of ER-negative
    return dict(AUC=auc,
                accuracy=(tp + tn) / len(y),
                balanced_accuracy=balanced_accuracy_score(y, pred) if len(np.unique(y)) == 2 else np.nan,
                sensitivity_ERpos=sens, specificity_ERneg=spec,
                F1_ERpos=f1_score(y, pred, pos_label=1, zero_division=0),
                F1_ERneg=f1_score(y, pred, pos_label=0, zero_division=0),
                macro_F1=f1_score(y, pred, average="macro", zero_division=0),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), n=len(y), threshold=float(thr))


def choose_threshold(mode: str, y_val: np.ndarray, p_val: np.ndarray) -> float:
    if mode == "val_youden" and len(np.unique(y_val)) == 2:
        fpr, tpr, thr = roc_curve(y_val, p_val)
        j = tpr - fpr
        t = thr[int(np.argmax(j))]
        return float(np.clip(t, 1e-6, 1 - 1e-6)) if np.isfinite(t) else 0.5
    return 0.5


# =============================================================================
# MODELS
# =============================================================================
def build_mil_model(in_dim: int, hidden: int, dropout: float, pooling: str):
    import torch.nn as nn

    class BasicMIL(nn.Module):
        """tile features -> small trainable layer -> FIXED pooling (mean/max) -> linear -> logit."""

        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.classifier = nn.Linear(hidden, 1)
            self.pooling = pooling

        def forward(self, x, mask):                       # x: B x N x D, mask: B x N (True = real tile)
            h = self.embed(x)                             # B x N x H
            m = mask.unsqueeze(-1)                        # B x N x 1
            if self.pooling == "mean":
                z = (h * m.float()).sum(dim=1) / m.float().sum(dim=1).clamp(min=1.0)
            else:                                         # max
                z = h.masked_fill(~m, float("-inf")).max(dim=1).values
            return self.classifier(z).squeeze(-1)         # B logits

    return BasicMIL()


def pad_bags(arrs: list[np.ndarray]):
    """Stack bags of (possibly) different tile counts into one array + mask."""
    nmax = max(a.shape[0] for a in arrs)
    d = arrs[0].shape[1]
    x = np.zeros((len(arrs), nmax, d), dtype=np.float32)
    m = np.zeros((len(arrs), nmax), dtype=bool)
    for i, a in enumerate(arrs):
        x[i, :len(a)] = a
        m[i, :len(a)] = True
    return x, m


def fit_scaler(bags: dict, train_ids: list[str]):
    """Feature scaling fitted on TRAIN patients' tiles only (no information from val/test)."""
    cat = np.concatenate([bags[p] for p in train_ids], axis=0)
    mu = cat.mean(axis=0)
    sd = cat.std(axis=0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def run_mil(sd: dict, pooling: str, args, seed: int) -> dict:
    """Train one basic-MIL model on one split. Returns val/test probabilities."""
    import torch
    import torch.nn as nn

    device = torch.device(args.device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    bags = sd["bags"]
    mu, sg = fit_scaler(bags, sd["train_ids"])

    def tensors(ids):
        x, m = pad_bags([(bags[p] - mu) / sg for p in ids])
        return torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)

    Xtr, Mtr = tensors(sd["train_ids"])
    Xva, Mva = tensors(sd["val_ids"])
    Xte, Mte = tensors(sd["test_ids"])
    ytr = torch.tensor(sd["y_train"], dtype=torch.float32, device=device)
    yva = torch.tensor(sd["y_val"], dtype=torch.float32, device=device)

    n_pos = float(ytr.sum().item())
    n_neg = float(len(ytr) - n_pos)
    # class balance: ER-positive is the majority, so its loss term is down-weighted (n_neg / n_pos < 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1.0)], device=device))

    model = build_mil_model(Xtr.shape[2], args.hidden, args.dropout, pooling).to(device)
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
            vloss = float(loss_fn(model(Xva, Mva), yva).item())      # VALIDATION loss decides when to stop
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
        p_test = torch.sigmoid(model(Xte, Mte)).cpu().numpy()
    return dict(p_val=p_val, p_test=p_test, info=f"best_epoch={best_epoch}, val_loss={best_loss:.3f}")


def run_logreg(sd: dict, args, seed: int) -> dict:
    """Logistic regression on each patient's mean tile feature (features scaled on TRAIN only)."""
    bags = sd["bags"]

    def vec(ids):
        return np.stack([bags[p].mean(axis=0) for p in ids])

    Xtr, Xva, Xte = vec(sd["train_ids"]), vec(sd["val_ids"]), vec(sd["test_ids"])
    mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sg, (Xva - mu) / sg, (Xte - mu) / sg
    best = None
    for C in (0.0005, 0.005, 0.05, 0.5):                      # strength picked on VALIDATION log-loss
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
        clf.fit(Xtr, sd["y_train"])
        pv = clf.predict_proba(Xva)[:, 1]
        ll = log_loss(sd["y_val"], np.clip(pv, 1e-6, 1 - 1e-6), labels=[0, 1])
        if best is None or ll < best[0]:
            best = (ll, C, clf)
    _, C, clf = best
    return dict(p_val=clf.predict_proba(Xva)[:, 1], p_test=clf.predict_proba(Xte)[:, 1], info=f"C={C}")


def run_majority(sd: dict, args, seed: int) -> dict:
    prev = float(np.mean(sd["y_train"]))
    top = 1.0 if prev >= 0.5 else 0.0        # constant "always the common class" -> AUC is exactly 0.5 by construction
    return dict(p_val=np.full(len(sd["val_ids"]), top), p_test=np.full(len(sd["test_ids"]), top), info=f"always predicts {'ER-positive' if top == 1.0 else 'ER-negative'}")


# =============================================================================
# EXPERIMENT RUNNER (used by the real run AND by --selftest)
# =============================================================================
def split_data(bags: dict, labels: dict, g: pd.DataFrame) -> dict:
    ids = {r: g.loc[g.role == r, "PATIENT"].tolist() for r in ("train", "val", "test")}
    # leakage guard: roles must be disjoint
    assert not (set(ids["train"]) & set(ids["val"])), "train/val overlap"
    assert not (set(ids["train"]) & set(ids["test"])), "train/test overlap"
    assert not (set(ids["val"]) & set(ids["test"])), "val/test overlap"
    out = dict(bags=bags)
    for r in ("train", "val", "test"):
        out[f"{r}_ids"] = ids[r]
        out[f"y_{r}"] = np.array([labels[p] for p in ids[r]], dtype=int)
    return out


def run_experiment(bags: dict, labels: dict, splits: pd.DataFrame, methods: list[str], args, verbose=True) -> pd.DataFrame:
    """Returns one row per (method, split, test patient) with the predicted probability."""
    rows = []
    split_ids = list(dict.fromkeys(splits["split_id"]))
    t_start = time.time()
    for k, sid in enumerate(split_ids):
        g = splits[splits.split_id == sid]
        sd = split_data(bags, labels, g)
        rep, fold = int(g["repeat"].iloc[0]), int(g["fold"].iloc[0])
        for m in methods:
            seed = args.seed + 1000 * k + hash_method(m)
            if m == "majority":
                res = run_majority(sd, args, seed)
            elif m == "logreg":
                res = run_logreg(sd, args, seed)
            elif m in ("mil_mean", "mil_max"):
                res = run_mil(sd, m.split("_")[1], args, seed)
            else:
                raise SystemExit(f"unknown method {m}")
            thr = choose_threshold(args.threshold_mode, sd["y_val"], res["p_val"])
            mt = compute_metrics(sd["y_test"], res["p_test"], thr)
            if verbose:
                auc = "n/a" if np.isnan(mt["AUC"]) else f"{mt['AUC']:.3f}"
                say(f"  [{k + 1}/{len(split_ids)}] {sid}  {m:<9} test AUC={auc}  balAcc={mt['balanced_accuracy']:.3f}  ({res['info']})  [{time.time() - t_start:.0f}s]")
            for pid, yy, pp in zip(sd["test_ids"], sd["y_test"], res["p_test"]):
                rows.append(dict(method=m, split_id=sid, repeat=rep, fold=fold, PATIENT=pid, ER_label=int(yy),
                                 prob_ERpos=float(pp), threshold=thr, pred=int(pp >= thr)))
    return pd.DataFrame(rows)


def hash_method(m: str) -> int:
    return int(hashlib.md5(m.encode()).hexdigest()[:4], 16) % 97


# =============================================================================
# AGGREGATION
# =============================================================================
def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (np.nan, np.nan)


def aggregate(preds: pd.DataFrame, all_patients: list[str], seed: int):
    """Per-split metrics, per-repeat pooled metrics, and one ensemble prediction per patient."""
    per_split, per_repeat, ens_rows, ens_pred = [], [], [], {}
    for m, d in preds.groupby("method", sort=False):
        for sid, g in d.groupby("split_id", sort=False):
            per_split.append(dict(method=m, split_id=sid, repeat=int(g["repeat"].iloc[0]), **compute_metrics(g.ER_label, g.prob_ERpos, g.threshold.iloc[0])))
        complete = [r for r, g in d.groupby("repeat") if set(g.PATIENT) == set(all_patients) and g.PATIENT.is_unique]
        for r in complete:
            g = d[d.repeat == r]
            # AUC from the probabilities; the other metrics use the predictions that were actually made
            mt = {**compute_metrics(g.ER_label, g.prob_ERpos, 0.5), **_metrics_from_pred(g.ER_label.values, g.prob_ERpos.values, g.pred.values)}
            per_repeat.append(dict(method=m, repeat=r, **mt))
        if complete:
            g = d[d.repeat.isin(complete)].groupby("PATIENT").agg(ER_label=("ER_label", "first"), prob=("prob_ERpos", "mean"),
                                                                  thr=("threshold", "mean")).reset_index()
            mt = compute_metrics(g.ER_label.values, g.prob.values, float(g.thr.mean()))
            lo, hi = bootstrap_auc_ci(g.ER_label.values, g.prob.values, seed=seed)
            ens_rows.append(dict(method=m, n_repeats=len(complete), AUC_CI95_low=lo, AUC_CI95_high=hi, **mt))
            ens_pred[m] = g
    return pd.DataFrame(per_split), pd.DataFrame(per_repeat), pd.DataFrame(ens_rows), ens_pred


def _metrics_from_pred(y, p, pred) -> dict:
    """Same as compute_metrics but with the predictions that were actually made."""
    y = np.asarray(y).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=(tp + tn) / len(y), balanced_accuracy=balanced_accuracy_score(y, pred),
                sensitivity_ERpos=tp / (tp + fn) if (tp + fn) else np.nan, specificity_ERneg=tn / (tn + fp) if (tn + fp) else np.nan,
                F1_ERpos=f1_score(y, pred, pos_label=1, zero_division=0), F1_ERneg=f1_score(y, pred, pos_label=0, zero_division=0),
                macro_F1=f1_score(y, pred, average="macro", zero_division=0), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def build_summary(per_split, per_repeat, ens) -> pd.DataFrame:
    rows = []
    for m in per_split.method.unique():
        s = per_split[per_split.method == m]
        row = dict(method=m, label=METHOD_LABELS.get(m, m), n_splits=len(s),
                   AUC_mean_over_splits=s.AUC.mean(), AUC_sd_over_splits=s.AUC.std(),
                   F1_ERpos_mean_over_splits=s.F1_ERpos.mean(), F1_ERneg_mean_over_splits=s.F1_ERneg.mean(),
                   balAcc_mean_over_splits=s.balanced_accuracy.mean())
        if len(per_repeat):
            r = per_repeat[per_repeat.method == m]
            row.update(AUC_pooled_per_repeat_mean=r.AUC.mean(), AUC_pooled_per_repeat_sd=r.AUC.std())
        if len(ens) and m in set(ens.method):
            e = ens[ens.method == m].iloc[0]
            row.update({f"ens_{k}": e[k] for k in ["AUC", "AUC_CI95_low", "AUC_CI95_high", "F1_ERpos", "F1_ERneg", "macro_F1",
                                                   "balanced_accuracy", "sensitivity_ERpos", "specificity_ERneg", "TN", "FP", "FN", "TP", "threshold"]})
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# PLOTS
# =============================================================================
def make_plots(per_split, ens_pred, ens, out_dir: Path, stamp: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(per_split.method.unique())
    colors = {"majority": "#8a8a8a", "logreg": "#1f77b4", "mil_mean": "#d95f02", "mil_max": "#1b9e77"}
    saved = []
    if ens_pred:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for m, g in ens_pred.items():
            fpr, tpr, _ = roc_curve(g.ER_label, g.prob)
            e = ens[ens.method == m].iloc[0]
            ax[0].plot(fpr, tpr, color=colors.get(m), lw=2, label=f"{METHOD_LABELS.get(m, m)}  AUC={e.AUC:.2f} [{e.AUC_CI95_low:.2f}-{e.AUC_CI95_high:.2f}]")
        ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
        ax[0].set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
                  title="ROC: one out-of-fold prediction per patient\n(averaged over the repeats)")
        ax[0].legend(loc="lower right", fontsize=8)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].text(0.5, 0.5, "ROC needs the complete 15-split run", ha="center")
    data = [per_split[per_split.method == m].AUC.dropna().values for m in methods]
    ax[1].boxplot(data, showmeans=True)
    ax[1].set_xticklabels([METHOD_LABELS.get(m, m).replace(" (", "\n(") for m in methods], fontsize=8)
    for i, d in enumerate(data, start=1):
        ax[1].scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(d)), d, s=14, alpha=0.6, color="k")
    ax[1].axhline(0.5, color="k", ls="--", lw=1)
    ax[1].set(ylabel="Test AUC of one split", title="AUC in each split (every dot = 1 split of 20 test patients)")
    plt.tight_layout()
    p = out_dir / "roc_and_auc.png"
    _backup(p, stamp)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    if ens_pred:
        fig, axes = plt.subplots(1, len(ens_pred), figsize=(3.6 * len(ens_pred), 3.8))
        axes = np.atleast_1d(axes)
        for ax_, (m, g) in zip(axes, ens_pred.items()):
            thr = float(ens[ens.method == m].iloc[0].threshold)
            cm = confusion_matrix(g.ER_label, (g.prob >= thr).astype(int), labels=[0, 1])
            ax_.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax_.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax_.set_xticks([0, 1], ["pred ER-", "pred ER+"])
            ax_.set_yticks([0, 1], ["true ER-", "true ER+"])
            ax_.set_title(METHOD_LABELS.get(m, m), fontsize=9)
        plt.tight_layout()
        p = out_dir / "confusion_matrices.png"
        _backup(p, stamp)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)
    return saved


def _backup(path: Path, stamp: str) -> None:
    if path.exists():
        path.rename(path.with_name(f"{path.stem}_previous_{stamp}{path.suffix}"))


def write_csv(df: pd.DataFrame, path: Path, stamp: str) -> None:
    _backup(path, stamp)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =============================================================================
# DATA LOADING + CHECKS
# =============================================================================
def file_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]


def load_inputs(args, rep: Report):
    for p in (args.splits_csv, args.labels_csv):
        if not p.is_file():
            say(f"ERROR: file not found: {p}")
            return None
    if not args.features_dir.is_dir():
        say(f"ERROR: feature folder not found: {args.features_dir}\n       Run extract_features.py first.")
        return None
    splits = pd.read_csv(args.splits_csv, encoding="utf-8-sig", dtype={"PATIENT": str})
    lab = pd.read_csv(args.labels_csv, encoding="utf-8-sig", dtype={"PATIENT": str})
    fp = file_hash(splits)
    rep.add("split file fingerprint", PASS if fp == EXPECTED_SPLIT_FINGERPRINT else WARN, observed=fp, expected=EXPECTED_SPLIT_FINGERPRINT,
            detail="" if fp == EXPECTED_SPLIT_FINGERPRINT else "Different from the fingerprint printed by make_patient_splits.py on your machine - make sure these are the intended splits.")
    patients = sorted(splits.PATIENT.unique())
    labels = dict(zip(lab.PATIENT, lab.ER_label.astype(int)))
    bad = [p for p in patients if p not in labels]
    rep.add("every split patient has an ER label", PASS if not bad else FAIL, observed=len(bad), expected=0)
    mism = splits.drop_duplicates("PATIENT").assign(L=lambda d: d.PATIENT.map(labels)).query("L != ER_label")
    rep.add("labels in splits file equal validated label table", PASS if mism.empty else FAIL, observed=len(mism), expected=0)
    # role integrity
    bad_split = []
    for sid, g in splits.groupby("split_id"):
        roles = {r: set(x.PATIENT) for r, x in g.groupby("role")}
        if len(g) != len(patients) or any(roles[a] & roles[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            bad_split.append(sid)
    rep.add("train / val / test never overlap and cover all patients in every split", PASS if not bad_split else FAIL, observed=len(bad_split), expected=0)
    # features
    bags, miss, dims, nonfinite = {}, [], set(), []
    for pid in patients:
        f = args.features_dir / f"{pid}.npz"
        if not f.is_file():
            miss.append(pid)
            continue
        with np.load(f, allow_pickle=False) as z:
            a = z["features"].astype(np.float32)
        if not np.isfinite(a).all():
            nonfinite.append(pid)
        dims.add(a.shape[1])
        bags[pid] = a
    rep.add("feature file exists for every patient", PASS if not miss else FAIL, observed=f"{len(miss)} missing", expected=0,
            detail=", ".join(miss[:8]))
    rep.add("all feature values finite", PASS if not nonfinite else FAIL, observed=len(nonfinite), expected=0)
    rep.add("all patients have the same feature dimension", PASS if len(dims) <= 1 else FAIL, observed=sorted(dims))
    if bags:
        ns = [len(v) for v in bags.values()]
        rep.add("tiles per patient", INFO, observed=f"min={min(ns)}, max={max(ns)}, total={sum(ns)}")
    if rep.n_fail():
        return None
    return splits, labels, bags, patients


# =============================================================================
# SELF-TEST (synthetic data, no files needed): checks the whole code path
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
    # quick splits: 1 repeat of stratified 5-fold, 15% validation (same recipe as make_patient_splits.py)
    rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=1)
    for f, (pool, test) in enumerate(skf.split(ids, y), start=1):
        tr, va = train_test_split(pool, test_size=0.15, stratify=y[pool], random_state=f)
        for role, idx in (("train", tr), ("val", va), ("test", test)):
            rows += [(f"r1_f{f}", 1, f, ids[i], int(y[i]), role) for i in idx]
    splits = pd.DataFrame(rows, columns=["split_id", "repeat", "fold", "PATIENT", "ER_label", "role"])

    st = copy.copy(args)
    st.epochs, st.patience, st.hidden, st.batch_size = 40, 8, 32, 16
    methods = ["majority", "logreg", "mil_mean", "mil_max"]
    ok = True
    for scenario in ("SIGNAL", "NOISE"):
        bags = {}
        for pid, yy in labels.items():
            a = rng.normal(size=(100, 64)).astype(np.float32)
            if scenario == "SIGNAL" and yy == 0:                    # 25% of an ER-negative patient's tiles carry a pattern
                k = rng.choice(100, 25, replace=False)
                a[k, :5] += 2.0
            bags[pid] = a
        say(f"\n--- scenario {scenario}: " + ("a planted pattern exists in ER-negative patients -> models should find it"
                                              if scenario == "SIGNAL" else "pure random noise -> AUC must be about 0.5 (otherwise something leaks)"))
        preds = run_experiment(bags, labels, splits, methods, st, verbose=False)
        res = {}
        for m, d in preds.groupby("method"):
            res[m] = float(np.mean([roc_auc_score(g.ER_label, g.prob_ERpos) for _, g in d.groupby("split_id")]))
            say(f"    {m:<9} mean test AUC over 5 splits = {res[m]:.3f}")
        if scenario == "SIGNAL":
            for m in ("logreg", "mil_mean", "mil_max"):
                good = res[m] >= 0.80
                ok &= good
                say(f"    [{'PASS' if good else 'FAIL'}] {m} finds the planted pattern (AUC >= 0.80)")
        else:
            for m in ("logreg", "mil_mean", "mil_max"):
                good = 0.15 <= res[m] <= 0.70      # leakage would push AUC UP; CV on pure noise is often slightly below 0.5
                ok &= good
                say(f"    [{'PASS' if good else 'FAIL'}] {m} is not above chance on noise (0.15 <= AUC <= 0.70)")
        good = abs(res["majority"] - 0.5) < 1e-9
        ok &= good
        say(f"    [{'PASS' if good else 'FAIL'}] majority baseline AUC = 0.5 exactly")
    say("\nSELF-TEST RESULT: " + ("PASSED - the training code works" if ok else "FAILED - do not trust results; send me this output"))
    return 0 if ok else 1


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Basic MIL (mean / max pooling) for ER prediction on the 15 patient-level splits.")
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--splits-csv", type=Path, default=SPLITS_CSV)
    ap.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--methods", nargs="+", default=["majority", "logreg", "mil_mean", "mil_max"],
                    choices=["majority", "logreg", "mil_mean", "mil_max"])
    ap.add_argument("--max-splits", type=int, default=None, help="quick test: only the first N splits (pooled metrics are skipped)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15, help="stop if validation loss has not improved for this many epochs")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threshold-mode", choices=["fixed", "val_youden"], default="fixed",
                    help="fixed = 0.5 (the loss is class-balanced); val_youden = best sensitivity+specificity on VALIDATION patients")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true", help="run the synthetic-data self-test and exit")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        if any(m.startswith("mil") for m in args.methods) or args.selftest:
            say("ERROR: PyTorch is not installed. Install it with:  python -m pip install torch torchvision matplotlib")
            return 2

    if args.selftest:
        return selftest(args)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = Report()
    say("=" * 70)
    say("BASIC MIL FOR ER PREDICTION  (frozen ResNet50 features, patient-level CV)")
    say("=" * 70)
    say(f"Features : {args.features_dir}\nSplits   : {args.splits_csv}\nLabels   : {args.labels_csv}\nOutput   : {args.out_dir}")
    loaded = load_inputs(args, rep)
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

    if args.max_splits:
        keep = list(dict.fromkeys(splits.split_id))[:args.max_splits]
        splits = splits[splits.split_id.isin(keep)]
        say(f"QUICK TEST: only splits {keep}")
    for k, v in dict(methods=args.methods, hidden=args.hidden, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
                     epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, threshold_mode=args.threshold_mode,
                     device=args.device, seed=args.seed).items():
        rep.add(f"setting: {k}", INFO, observed=str(v))

    say("\nTraining / evaluating (test set is only used for the numbers shown):")
    t0 = time.time()
    preds = run_experiment(bags, labels, splits, args.methods, args)
    say(f"\nFinished in {(time.time() - t0) / 60:.1f} min")

    per_split, per_repeat, ens, ens_pred = aggregate(preds, patients, args.seed)
    summary = build_summary(per_split, per_repeat, ens)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    say("\nWriting output files...")
    write_csv(summary, args.out_dir / "summary_metrics.csv", stamp)
    write_csv(per_split, args.out_dir / "per_split_metrics.csv", stamp)
    write_csv(preds, args.out_dir / "predictions_test.csv", stamp)
    plots = make_plots(per_split, ens_pred, ens, args.out_dir, stamp)
    complete = bool(len(ens))
    rep.add("every patient tested exactly once per repeat (pooled metrics possible)", PASS if complete else WARN,
            detail="" if complete else "Not all folds were run (quick test), so pooled/ensemble metrics were skipped.")
    write_csv(rep.df(), args.out_dir / "mil_run_report.csv", stamp)
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
    say("\nHOW TO READ THIS")
    say("  * Chance AUC is 0.5. 'Majority class' must show 0.5 and F1 ER- = 0; if not, something is wrong.")
    say("  * Accuracy is misleading here: always predicting ER-positive is already 76% accurate.")
    say("  * Look at F1 for ER-NEGATIVE, balanced accuracy and the confusion matrix: they show whether the minority class is found.")
    say("  * The 15 splits reuse the same 100 patients, so the ± and the CI are optimistic about certainty.")
    say("  * A MIL model is only interesting if it clearly beats 'LogReg on mean features'.")
    say(f"\nFiles: {', '.join(p.name for p in plots)}, summary_metrics.csv, per_split_metrics.csv, predictions_test.csv, mil_run_report.csv")
    say("\nRESULT: DONE" if not rep.n_fail() else "\nRESULT: CHECKS FAILED")
    return 0 if not rep.n_fail() else 1


if __name__ == "__main__":
    sys.exit(main())

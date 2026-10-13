#!/usr/bin/env python3
"""
compare_resnet50_vs_uni.py   (Step 6b of the ER pipeline)
==================================================
Puts two finished basic-MIL runs side by side (default: ResNet50 vs UNI) using the
SAME 15 patient-level splits, so the comparison is fair and PAIRED.

NO TRAINING happens here. It only reads what train_basic_mil.py (ResNet50) and train_basic_mil_uni.py (UNI) already wrote:
  mil_results/basic_mil_<A>/predictions_test.csv
  mil_results/basic_mil_<B>/predictions_test.csv

Writes (new folder mil_results/encoder_comparison_<A>_vs_<B>/):
  encoder_comparison.csv   per method: AUC for each encoder, difference, paired bootstrap CI,
                           in how many of the 15 splits encoder B was better
  encoder_comparison.png   left: ROC curves (solid = B, dashed = A)
                           right: per-split AUC difference (B minus A), one dot per split

Usage:
  python compare_resnet50_vs_uni.py

HOW TO READ IT
  * "AUC difference" > 0 means encoder B (UNI) scored higher than encoder A.
  * The 95% interval comes from re-sampling the 100 patients (paired: both encoders are scored
    on the same resampled patients). If it contains 0, the data do not show a clear difference.
  * The 15 splits reuse the same 100 patients, so intervals are somewhat too narrow.
    Treat "UNI better in 12/15 splits" as a hint, not a proof.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

BASE = Path.home() / "Desktop" / "Capstone" / "Implementation" / "Phase1"
MIL_ROOT = BASE / "mil_results"
LABELS = {"majority": "Majority class", "logreg": "LogReg on mean features",
          "mil_mean": "Basic MIL (mean pool)", "mil_max": "Basic MIL (max pool)"}
COLORS = {"majority": "#8a8a8a", "logreg": "#1f77b4", "mil_mean": "#d95f02", "mil_max": "#1b9e77"}


def say(msg: str = "") -> None:
    print(msg, flush=True)


def load_preds(folder: Path) -> pd.DataFrame | None:
    f = folder / "predictions_test.csv"
    if not f.is_file():
        say(f"ERROR: {f} not found. Run the matching training script first (train_basic_mil.py for ResNet50, train_basic_mil_uni.py for UNI).")
        return None
    return pd.read_csv(f, encoding="utf-8-sig", dtype={"PATIENT": str})


def ensemble(d: pd.DataFrame, all_patients: set[str]) -> pd.DataFrame | None:
    """One probability per patient: mean over the repeats (only repeats that tested every patient once)."""
    complete = [r for r, g in d.groupby("repeat") if set(g.PATIENT) == all_patients and g.PATIENT.is_unique]
    if not complete:
        return None
    g = d[d.repeat.isin(complete)].groupby("PATIENT").agg(ER_label=("ER_label", "first"), prob=("prob_ERpos", "mean")).reset_index()
    return g.sort_values("PATIENT").reset_index(drop=True)


def paired_bootstrap(y, pa, pb, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]))
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) if diffs else (np.nan, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired comparison of two basic-MIL runs on the same splits (no training).")
    ap.add_argument("--a", default="resnet50", help="first encoder / baseline (folder basic_mil_<a>)")
    ap.add_argument("--b", default="uni", help="second encoder (folder basic_mil_<b>)")
    ap.add_argument("--mil-root", type=Path, default=MIL_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    A = load_preds(args.mil_root / f"basic_mil_{args.a}")
    B = load_preds(args.mil_root / f"basic_mil_{args.b}")
    if A is None or B is None:
        return 2
    out_dir = args.out_dir or args.mil_root / f"encoder_comparison_{args.a}_vs_{args.b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- the two runs must be about exactly the same patients and splits --------------------
    problems = []
    if set(A.split_id) != set(B.split_id):
        problems.append("the two runs do not contain the same splits")
    if set(A.PATIENT) != set(B.PATIENT):
        problems.append("the two runs do not contain the same patients")
    ya = A.drop_duplicates("PATIENT").set_index("PATIENT").ER_label
    yb = B.drop_duplicates("PATIENT").set_index("PATIENT").ER_label
    if not ya.sort_index().equals(yb.sort_index()):
        problems.append("ER labels differ between the two runs")
    if problems:
        say("ERROR: runs are not comparable: " + "; ".join(problems))
        return 2
    all_patients = set(A.PATIENT)
    methods = [m for m in LABELS if m in set(A.method) and m in set(B.method)]
    say(f"Comparing  A = {args.a}   vs   B = {args.b}   ({len(all_patients)} patients, {A.split_id.nunique()} splits)")

    rows, ens_curves = [], {}
    for m in methods:
        da, db = A[A.method == m], B[B.method == m]
        # per-split paired AUC
        sa = pd.Series({sid: roc_auc_score(g.ER_label, g.prob_ERpos) for sid, g in da.groupby("split_id")})
        sb = pd.Series({sid: roc_auc_score(g.ER_label, g.prob_ERpos) for sid, g in db.groupby("split_id")})
        diff = (sb - sa.reindex(sb.index)).dropna()
        row = dict(method=m, label=LABELS[m], n_splits=len(diff),
                   **{f"AUC_mean_over_splits_{args.a}": sa.mean(), f"AUC_mean_over_splits_{args.b}": sb.mean()},
                   split_AUC_diff_mean=diff.mean(), split_AUC_diff_sd=diff.std(),
                   **{f"splits_where_{args.b}_better": int((diff > 0).sum()), "splits_tied": int((diff == 0).sum())})
        ea, eb = ensemble(da, all_patients), ensemble(db, all_patients)
        if ea is not None and eb is not None:
            y = ea.ER_label.values
            auc_a, auc_b = roc_auc_score(y, ea.prob), roc_auc_score(y, eb.prob)
            lo, hi = paired_bootstrap(y, ea.prob.values, eb.prob.values)
            row.update({f"ens_AUC_{args.a}": auc_a, f"ens_AUC_{args.b}": auc_b, "ens_AUC_diff": auc_b - auc_a,
                        "ens_AUC_diff_CI95_low": lo, "ens_AUC_diff_CI95_high": hi})
            ens_curves[m] = (ea, eb)
        rows.append(row)
        row["_diff_series"] = diff
    table = pd.DataFrame(rows)

    # ---- console summary -----------------------------------------------------------------
    say("\n" + "=" * 78)
    say(f"RESULTS  (difference = {args.b} minus {args.a}; every number is on patients not used for training)")
    say("=" * 78)
    for _, r in table.iterrows():
        say(f"\n{r.label}")
        say(f"  mean AUC over splits : {args.a} {r[f'AUC_mean_over_splits_{args.a}']:.3f}   {args.b} {r[f'AUC_mean_over_splits_{args.b}']:.3f}"
            f"   difference {r.split_AUC_diff_mean:+.3f} (SD {r.split_AUC_diff_sd:.3f})")
        say(f"  {args.b} better in    : {int(r[f'splits_where_{args.b}_better'])} of {int(r.n_splits)} splits")
        if f"ens_AUC_{args.a}" in r and not pd.isna(r.get(f"ens_AUC_{args.a}")):
            sig = "interval excludes 0" if (r.ens_AUC_diff_CI95_low > 0 or r.ens_AUC_diff_CI95_high < 0) else "interval includes 0 -> no clear difference"
            say(f"  ensemble AUC         : {args.a} {r[f'ens_AUC_{args.a}']:.3f}   {args.b} {r[f'ens_AUC_{args.b}']:.3f}"
                f"   difference {r.ens_AUC_diff:+.3f}  95% CI [{r.ens_AUC_diff_CI95_low:+.3f}, {r.ens_AUC_diff_CI95_high:+.3f}]  ({sig})")
    say("\nREMINDER: the 15 splits reuse the same 100 patients, so these intervals are optimistic. "
        "Do not pick the best-looking method on the test numbers and then report it as if it were untouched.")

    # ---- files ---------------------------------------------------------------------------
    csv_path = out_dir / "encoder_comparison.csv"
    table.drop(columns=["_diff_series"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for m, (ea, eb) in ens_curves.items():
            if m == "majority":
                continue
            for e, ls, nm in ((ea, "--", args.a), (eb, "-", args.b)):
                fpr, tpr, _ = roc_curve(e.ER_label, e.prob)
                ax[0].plot(fpr, tpr, ls=ls, lw=2, color=COLORS[m], label=f"{LABELS[m]} - {nm}  AUC={roc_auc_score(e.ER_label, e.prob):.2f}")
        ax[0].plot([0, 1], [0, 1], "k:", lw=1, label="chance")
        ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
                  title=f"ROC (one prediction per patient)\nsolid = {args.b}, dashed = {args.a}")
        ax[0].legend(fontsize=7, loc="lower right")
        use = [m for m in methods if m != "majority"]
        data = [table.loc[table.method == m, "_diff_series"].iloc[0].values for m in use]
        ax[1].boxplot(data, showmeans=True)
        ax[1].set_xticks(range(1, len(use) + 1))
        ax[1].set_xticklabels([LABELS[m].replace(" (", "\n(").replace(" on ", "\non ") for m in use], fontsize=8)
        rng = np.random.default_rng(0)
        for i, d in enumerate(data, start=1):
            ax[1].scatter(np.full(len(d), i) + rng.uniform(-0.08, 0.08, len(d)), d, s=16, alpha=0.6, color="k")
        ax[1].axhline(0, color="k", ls="--", lw=1)
        ax[1].set(ylabel=f"AUC({args.b}) - AUC({args.a}) in the same split",
                  title="Paired difference per split (dot = 1 split)\nabove 0 = " + f"{args.b} better")
        plt.tight_layout()
        png = out_dir / "encoder_comparison.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        say(f"\nSaved: {csv_path}\nSaved: {png}")
    except ImportError:
        say(f"\nSaved: {csv_path}   (install matplotlib to also get the figure)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
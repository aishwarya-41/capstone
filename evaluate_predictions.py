#!/usr/bin/env python3
"""
evaluate_predictions.py   (extra evaluation for ANY encoder / MIL run; read-only on your results)
=================================================================================================
Reads a predictions_test.csv written by train_basic_mil.py (ResNet50, CONCH, UNI ... same format) and
adds the metrics that script does not report. Using this ONE script for every encoder means the
numbers are computed identically, so encoder comparisons cannot differ because of metric code.

It never changes the existing results: it only WRITES NEW FILES (if a file of the same name exists it
is renamed with a timestamp first).

TWO VIEWS OF THE SAME PREDICTIONS  (each patient is tested once per repeat, in 3 repeats)
  single_model   For each repeat, the 100 patients' predictions come from one set of 5 fold-models.
                 Metrics are computed per repeat, then averaged over the repeats (+ SD over repeats).
                 This is the honest "what does one trained pipeline achieve" number.
  ensemble       Each patient's probability is averaged over the repeats (what summary_metrics.csv calls
                 'ens_'). Averaging models usually raises AUC a little, so do not compare this view with
                 single-model numbers from other papers.

UNCERTAINTY  (why it is done this way)
  The 300 prediction rows are NOT 300 independent samples: the same 100 patients appear in every repeat.
  So the bootstrap resamples PATIENTS (with replacement), and each drawn patient brings ALL of their
  repeat predictions along. For the single_model view the metric is recomputed in every repeat on the
  resampled patients and then averaged over repeats, exactly like the point estimate.
  The same resampled patients are used for every method (paired), so differences between methods get a
  proper CI. Percentile 95% intervals.
  What this interval DOES capture : which patients happened to be in the cohort.
  What it does NOT capture        : training-set variability / the dependence between cross-validation
                                    folds. With 24 ER-negative patients treat all intervals as optimistic.

METRICS
  AUC, PR_AUC_ERpos (average precision, ER-positive = positive class), PR_AUC_ERneg (ER-negative =
  positive class, scores 1-p), sensitivity (ER+ recall), specificity (ER- recall), balanced_accuracy.
  PR-AUC is only meaningful next to its chance level = class prevalence (0.76 for ER+, 0.24 for ER-).
  Sensitivity / specificity / balanced accuracy use the thresholds that were actually used in the run
  (stored in the predictions file).

OUTPUT FILES (in the same folder as the predictions unless --out-dir is given)
  metrics_extended.csv            one row per method x view x metric: estimate, SD over repeats, 95% CI
  metrics_per_repeat.csv          one row per method x repeat (all metrics + confusion counts)
  metrics_paired_comparisons.csv  method A minus method B: difference with paired patient-bootstrap CI
  metrics_extended_info.json      input file hash, settings, package versions, cross-check results

RUN (any environment that has numpy, pandas, scikit-learn):
    python evaluate_predictions.py
    python evaluate_predictions.py --pred-csv mil_results\\basic_mil_conch\\predictions_test.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

BASE = Path(r"C:\Users\LENOVO\OneDrive\Documents\capstone")
PRED_CSV = BASE / "mil_results" / "basic_mil_resnet50" / "predictions_test.csv"
LABELS_CSV = BASE / "er_label_outputs" / "patient_er_labels.csv"
REQUIRED = ["method", "split_id", "repeat", "fold", "PATIENT", "ER_label", "prob_ERpos", "threshold", "pred"]
METRICS = ["AUC", "PR_AUC_ERpos", "PR_AUC_ERneg", "sensitivity", "specificity", "balanced_accuracy"]
VIEWS = ["single_model", "ensemble"]


def say(msg: str = "") -> None:
    print(msg, flush=True)


# =============================================================================
# METRICS
# =============================================================================
def metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict:
    """y: 1 = ER+, 0 = ER-. p: P(ER+). pred: the 0/1 decision that was actually made."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(AUC=roc_auc_score(y, p),
                PR_AUC_ERpos=average_precision_score(y, p),
                PR_AUC_ERneg=average_precision_score(1 - y, 1 - p),
                sensitivity=sens, specificity=spec, balanced_accuracy=(sens + spec) / 2,
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def backup(path: Path, stamp: str) -> None:
    if path.exists():
        path.rename(path.with_name(f"{path.stem}_previous_{stamp}{path.suffix}"))


def write_csv(df: pd.DataFrame, path: Path, stamp: str) -> None:
    backup(path, stamp)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =============================================================================
# LOAD + VALIDATE
# =============================================================================
def load_and_check(args):
    """Returns (arrays per method, list of patients, repeats, problems). Stops on any structural problem."""
    problems: list[str] = []
    df = pd.read_csv(args.pred_csv, encoding="utf-8-sig", dtype={"PATIENT": str})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise SystemExit(f"ERROR: predictions file lacks columns {miss}. Found: {list(df.columns)}")
    if not np.isfinite(df.prob_ERpos).all() or not df.prob_ERpos.between(0, 1).all():
        problems.append("prob_ERpos has NaN/inf or values outside [0, 1]")
    if not set(df.ER_label.unique()) <= {0, 1}:
        problems.append("ER_label has values other than 0/1")
    if df.groupby("PATIENT").ER_label.nunique().max() != 1:
        problems.append("a patient has different ER_label values in different rows")
    patients = sorted(df.PATIENT.unique())
    repeats = sorted(int(r) for r in df["repeat"].unique())
    if args.labels_csv.is_file():
        lab = pd.read_csv(args.labels_csv, encoding="utf-8-sig", dtype={"PATIENT": str})
        lm = dict(zip(lab.PATIENT, lab.ER_label.astype(int)))
        if set(patients) != set(lm):
            problems.append(f"patients in predictions ({len(patients)}) differ from the label table ({len(lm)})")
        elif any(lm[p] != l for p, l in df.drop_duplicates("PATIENT")[["PATIENT", "ER_label"]].itertuples(index=False)):
            problems.append("ER_label in predictions disagrees with patient_er_labels.csv")
    else:
        say(f"  (label table not found at {args.labels_csv}; label cross-check skipped)")
    y = df.drop_duplicates("PATIENT").set_index("PATIENT").loc[patients, "ER_label"].to_numpy(int)

    data = {}
    for m, d in df.groupby("method", sort=False):
        P = np.full((len(repeats), len(patients)), np.nan)
        T, D = P.copy(), P.copy()
        for ri, r in enumerate(repeats):
            g = d[d["repeat"] == r]
            if g.PATIENT.duplicated().any() or set(g.PATIENT) != set(patients):
                problems.append(f"{m}: repeat {r} does not contain every patient exactly once "
                                f"(is this a --max-splits quick test? pooled metrics need the full run)")
                continue
            g = g.set_index("PATIENT").loc[patients]
            P[ri], T[ri], D[ri] = g.prob_ERpos.to_numpy(), g.threshold.to_numpy(), g.pred.to_numpy()
        data[m] = dict(P=P, T=T, D=D)
    if problems:
        raise SystemExit("ERROR: input checks failed:\n  - " + "\n  - ".join(problems))
    return df, data, patients, repeats, y


# =============================================================================
# CORE: one metric set per (view) on a chosen set of patient indices
# =============================================================================
def view_metrics(d: dict, y: np.ndarray, idx: np.ndarray, view: str) -> list[dict]:
    """single_model -> list with one metric dict per repeat; ensemble -> list with one dict."""
    yy = y[idx]
    if view == "single_model":
        return [metrics(yy, d["P"][r, idx], d["D"][r, idx].astype(int)) for r in range(d["P"].shape[0])]
    p = d["P"][:, idx].mean(axis=0)
    thr = d["T"][:, idx].mean(axis=0)
    return [metrics(yy, p, (p >= thr).astype(int))]


def point_estimates(d, y, view):
    ms = view_metrics(d, y, np.arange(len(y)), view)
    est = {k: float(np.mean([m[k] for m in ms])) for k in METRICS}
    sd = {k: float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else np.nan for k in METRICS}
    return est, sd, ms


def bootstrap(data, y, n_boot, seed):
    """boot[method][view][metric] -> array of n_boot values. Same resampled patients for every method."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {m: {v: {k: [] for k in METRICS} for v in VIEWS} for m in data}
    skipped, t0 = 0, time.time()
    for b in range(n_boot):
        idx = rng.integers(0, n, n)                      # resample PATIENTS; all their repeats come along
        if len(np.unique(y[idx])) < 2:
            skipped += 1
            continue
        for m, d in data.items():
            for v in VIEWS:
                ms = view_metrics(d, y, idx, v)
                for k in METRICS:
                    boot[m][v][k].append(np.mean([x[k] for x in ms]))
        if (b + 1) % 250 == 0:
            say(f"    bootstrap {b + 1}/{n_boot}  [{time.time() - t0:.0f}s]")
    for m in boot:
        for v in VIEWS:
            for k in METRICS:
                boot[m][v][k] = np.asarray(boot[m][v][k])
    return boot, skipped


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Extra metrics + patient-level bootstrap CIs for a predictions_test.csv.")
    ap.add_argument("--pred-csv", type=Path, default=PRED_CSV)
    ap.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: the folder containing the predictions file")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.pred_csv.is_file():
        raise SystemExit(f"ERROR: file not found: {args.pred_csv}")
    out_dir = args.out_dir or args.pred_csv.parent
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    say("=" * 70 + "\nEXTENDED EVALUATION (read-only on existing results)\n" + "=" * 70)
    say(f"Predictions: {args.pred_csv}\nOutput folder: {out_dir}")
    df, data, patients, repeats, y = load_and_check(args)
    n_pos = int(y.sum())
    prev = n_pos / len(y)
    say(f"Methods: {list(data)} | patients: {len(patients)} (ER+ {n_pos}, ER- {len(y) - n_pos}) | repeats: {repeats}")
    say(f"Chance level of PR-AUC: ER+ = {prev:.3f}, ER- = {1 - prev:.3f}")

    # ---- cross-check against the numbers train_basic_mil.py already saved --------------------------
    checks = []
    summ_path = args.pred_csv.parent / "summary_metrics.csv"
    if summ_path.is_file():
        sm = pd.read_csv(summ_path)
        for m in data:
            row = sm[sm.method == m]
            if row.empty:
                continue
            row = row.iloc[0]
            ens, _, _ = point_estimates(data[m], y, "ensemble")
            sing, _, _ = point_estimates(data[m], y, "single_model")
            for name, mine, saved in (("ensemble AUC vs ens_AUC", ens["AUC"], row.get("ens_AUC")),
                                      ("single-model AUC vs AUC_pooled_per_repeat_mean", sing["AUC"], row.get("AUC_pooled_per_repeat_mean"))):
                ok = bool(np.isclose(mine, saved, atol=1e-9))
                checks.append(dict(method=m, check=name, mine=float(mine), saved=float(saved), ok=ok))
                say(f"  [{'PASS' if ok else 'FAIL'}] {m}: {name}: {mine:.4f} vs saved {saved:.4f}")
    else:
        say("  (no summary_metrics.csv next to the predictions; cross-check skipped)")
    if any(not c["ok"] for c in checks):
        raise SystemExit("ERROR: my recomputed AUCs do not match summary_metrics.csv - not writing files. Send me this output.")

    # ---- point estimates + per-repeat table ----------------------------------------------------------
    say(f"\nBootstrapping {args.n_boot} patient resamples (all repeats of a patient stay together)...")
    boot, skipped = bootstrap(data, y, args.n_boot, args.seed)
    n_used = args.n_boot - skipped
    rows, per_repeat = [], []
    for m, d in data.items():
        for v in VIEWS:
            est, sd, ms = point_estimates(d, y, v)
            if v == "single_model":
                for r, mt in zip(repeats, ms):
                    per_repeat.append(dict(method=m, repeat=r, **mt))
            for k in METRICS:
                lo, hi = np.percentile(boot[m][v][k], [2.5, 97.5])
                rows.append(dict(method=m, view=v, metric=k, estimate=est[k], sd_over_repeats=sd[k] if v == "single_model" else np.nan,
                                 ci95_low=lo, ci95_high=hi, n_boot_used=n_used,
                                 chance_level=(prev if k == "PR_AUC_ERpos" else 1 - prev if k == "PR_AUC_ERneg" else 0.5 if k in ("AUC", "balanced_accuracy") else np.nan)))
    ext = pd.DataFrame(rows)

    # ---- paired comparisons --------------------------------------------------------------------------
    pair_rows = []
    cand = [m for m in data if m != "majority"]
    for a, b in itertools.combinations(cand, 2):
        for v in VIEWS:
            ea, _, _ = point_estimates(data[a], y, v)
            eb, _, _ = point_estimates(data[b], y, v)
            for k in ("AUC", "PR_AUC_ERpos", "PR_AUC_ERneg", "balanced_accuracy"):
                diff = boot[a][v][k] - boot[b][v][k]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                pair_rows.append(dict(method_A=a, method_B=b, view=v, metric=k, estimate_A=ea[k], estimate_B=eb[k],
                                      difference_A_minus_B=ea[k] - eb[k], ci95_low=lo, ci95_high=hi,
                                      ci_excludes_zero=bool(lo > 0 or hi < 0), share_of_resamples_A_better=float((diff > 0).mean())))
    paired = pd.DataFrame(pair_rows)

    # ---- write ---------------------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(ext, out_dir / "metrics_extended.csv", stamp)
    write_csv(pd.DataFrame(per_repeat), out_dir / "metrics_per_repeat.csv", stamp)
    write_csv(paired, out_dir / "metrics_paired_comparisons.csv", stamp)
    info = dict(timestamp=stamp, predictions_file=str(args.pred_csv), predictions_sha256=hashlib.sha256(args.pred_csv.read_bytes()).hexdigest(),
                n_patients=len(patients), n_repeats=len(repeats), methods=list(data), n_boot=args.n_boot, n_boot_used=n_used, seed=args.seed,
                bootstrap_unit="patient (all repeat predictions of a drawn patient kept together; same draws for all methods)",
                versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
                cross_checks_vs_summary_metrics=checks)
    backup(out_dir / "metrics_extended_info.json", stamp)
    (out_dir / "metrics_extended_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # ---- console summary -----------------------------------------------------------------------------
    for v in VIEWS:
        say("\n" + "=" * 70)
        say(("SINGLE MODEL per repeat, averaged over the repeats [95% patient-bootstrap CI]; SD over repeats is in metrics_extended.csv" if v == "single_model"
             else "ENSEMBLE: probabilities averaged over the repeats (usually a bit higher; not comparable to single-model numbers)"))
        say("=" * 70)
        say(f"{'method':<10}" + "".join(f"{k:<26}" for k in ("AUC", "PR_AUC_ERneg", "PR_AUC_ERpos")) + "sens / spec / balAcc")
        for m in data:
            e = ext[(ext.method == m) & (ext.view == v)].set_index("metric")
            cells = []
            for k in ("AUC", "PR_AUC_ERneg", "PR_AUC_ERpos"):
                r = e.loc[k]
                cells.append(f"{r.estimate:.3f} [{r.ci95_low:.2f}-{r.ci95_high:.2f}]".ljust(26))
            say(f"{m:<10}" + "".join(cells) + f"{e.loc['sensitivity'].estimate:.2f} / {e.loc['specificity'].estimate:.2f} / {e.loc['balanced_accuracy'].estimate:.2f}")
    if len(paired):
        say("\nPAIRED DIFFERENCES in AUC, single-model view (A minus B, patient-bootstrap 95% CI):")
        for r in paired[(paired.metric == "AUC") & (paired.view == "single_model")].itertuples():
            say(f"  {r.method_A} - {r.method_B}: {r.difference_A_minus_B:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]"
                f"  {'CI excludes 0' if r.ci_excludes_zero else 'CI includes 0 (no clear difference)'}")
    say("\nHOW TO READ THIS")
    say("  * [a-b] = 95% bootstrap interval over PATIENTS. It ignores training randomness and the overlap between CV folds, so it is optimistic.")
    say("  * PR-AUC only means something relative to its chance level (ER+ 0.76, ER- 0.24 here).")
    say("  * A difference whose CI includes 0 is NOT evidence that one method is better.")
    say(f"\nWrote metrics_extended.csv, metrics_per_repeat.csv, metrics_paired_comparisons.csv, metrics_extended_info.json to {out_dir}")
    say("RESULT: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

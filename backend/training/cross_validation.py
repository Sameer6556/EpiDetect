"""
Main training and cross-validation pipeline.
It tests the models against basic baselines to prove our features actually work,
and then does 10-fold CV to get the final accuracy.
"""

import os
import time
import json
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    classification_report, roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import (
    N_FOLDS, THRESHOLD, FEATURE_SELECT_K, SEEDS,
    RESULTS_DIR, MODEL_DIR, SEIZURE_WEIGHT,
)
from models.ensemble import build_ensemble, multi_seed_predict


# ── Baseline comparison ──────────────────────────────────

def _run_baseline(name, model, X, y, cv):
    """Run a single baseline model through CV and return all 7 metrics."""
    accs, precs, recs, f1s, aucs, specs, kappas = [], [], [], [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        model.fit(X_tr, y[train_idx])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_te)[:, 1]
        else:
            probs = model.decision_function(X_te)

        preds = (probs >= THRESHOLD).astype(int) if hasattr(model, "predict_proba") \
            else model.predict(X_te)

        accs.append(accuracy_score(y[test_idx], preds) * 100)
        precs.append(precision_score(y[test_idx], preds, zero_division=0) * 100)
        recs.append(recall_score(y[test_idx], preds, zero_division=0) * 100)
        f1s.append(f1_score(y[test_idx], preds, zero_division=0) * 100)

        try:
            aucs.append(roc_auc_score(y[test_idx], probs) * 100)
        except ValueError:
            aucs.append(0.0)

        cm = confusion_matrix(y[test_idx], preds)
        specs.append(cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-12) * 100)
        kappas.append(cohen_kappa_score(y[test_idx], preds) * 100)

    row = {
        "name": name,
        "Acc": np.mean(accs), "Prec": np.mean(precs), "Rec": np.mean(recs),
        "F1": np.mean(f1s), "AUC": np.mean(aucs),
        "Spec": np.mean(specs), "Kappa": np.mean(kappas),
    }

    print(f"  {name:38s}: Acc={row['Acc']:.2f}%  Prec={row['Prec']:.2f}%  "
          f"Rec={row['Rec']:.2f}%  F1={row['F1']:.2f}%  AUC={row['AUC']:.2f}%  "
          f"Spec={row['Spec']:.2f}%  Kappa={row['Kappa']:.2f}%")

    return row


def run_baselines(X_raw, X_features, y):
    """
    Compare three standard classifiers on raw signals vs engineered features.
    This clearly shows the value of the feature engineering step.
    """
    cw = {0: 1, 1: int(SEIZURE_WEIGHT)}
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    baselines = []

    # Before feature engineering — raw 4097-sample signals
    print("\n  -- Raw signals (before feature engineering) --")
    baselines.append(_run_baseline(
        "Raw - Logistic Regression",
        LogisticRegression(max_iter=1000, class_weight=cw, random_state=42),
        X_raw, y, cv))
    baselines.append(_run_baseline(
        "Raw - Random Forest",
        RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42, n_jobs=-1),
        X_raw, y, cv))
    baselines.append(_run_baseline(
        "Raw - SVM (RBF)",
        SVC(kernel="rbf", class_weight=cw, probability=True, random_state=42),
        X_raw, y, cv))

    # After feature engineering — 720 extracted features
    print("\n  -- Engineered features (after feature engineering) --")
    baselines.append(_run_baseline(
        "Feat - Logistic Regression",
        LogisticRegression(max_iter=1000, class_weight=cw, random_state=42),
        X_features, y, cv))
    baselines.append(_run_baseline(
        "Feat - Random Forest",
        RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42, n_jobs=-1),
        X_features, y, cv))
    baselines.append(_run_baseline(
        "Feat - SVM (RBF)",
        SVC(kernel="rbf", class_weight=cw, probability=True, random_state=42),
        X_features, y, cv))

    return baselines


# ── Bootstrap CI ─────────────────────────────────────────

def _bootstrap_ci(y_true, y_prob, n_iter=1000, alpha=0.05):
    """Compute 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    boot = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}

    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        yp = (y_prob[idx] >= THRESHOLD).astype(int)

        if len(np.unique(yt)) < 2:
            continue

        boot["Accuracy"].append(accuracy_score(yt, yp) * 100)
        boot["Precision"].append(precision_score(yt, yp, zero_division=0) * 100)
        boot["Recall"].append(recall_score(yt, yp, zero_division=0) * 100)
        boot["F1"].append(f1_score(yt, yp, zero_division=0) * 100)
        try:
            boot["AUC"].append(roc_auc_score(yt, y_prob[idx]) * 100)
        except ValueError:
            pass

    ci = {}
    lo = (alpha / 2) * 100
    hi = (1 - alpha / 2) * 100
    for k, vals in boot.items():
        if vals:
            ci[k] = (np.percentile(vals, lo), np.percentile(vals, hi))

    return ci


# ── Main CV pipeline ─────────────────────────────────────

def run_cross_validation(X_features, y, X_raw=None):
    """
    Full evaluation pipeline:
      1. Baselines (if X_raw provided)
      2. 10-fold CV with multi-seed ensemble
      3. Metrics summary, confusion matrix, classification report
      4. Bootstrap CI
      5. ROC/PR curves
      6. Comparison table
      7. Save model artifacts
    """
    baselines = []
    if X_raw is not None:
        print(f"\n{'=' * 60}")
        print("  Step 4: Baseline Comparisons")
        print("=" * 60)
        baselines = run_baselines(X_raw, X_features, y)

    # ── Cross-validation ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 5: Training & Evaluation")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_metrics = []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    best_fold_f1 = -1
    best_fold_data = None

    t_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"  {N_FOLDS}-Fold Cross-Validation")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y)):

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_features[train_idx])
        X_test = scaler.transform(X_features[test_idx])

        # Feature selection — union of MI and F-classif
        mi_sel = SelectKBest(mutual_info_classif, k=FEATURE_SELECT_K)
        mi_sel.fit(X_train, y[train_idx])
        mi_top = set(np.argsort(mi_sel.scores_)[-FEATURE_SELECT_K:])

        fc_sel = SelectKBest(f_classif, k=FEATURE_SELECT_K)
        fc_sel.fit(X_train, y[train_idx])
        fc_top = set(np.argsort(fc_sel.scores_)[-FEATURE_SELECT_K:])

        selected = sorted(mi_top | fc_top)
        X_train_sel = X_train[:, selected]
        X_test_sel = X_test[:, selected]

        # Multi-seed prediction
        probs = multi_seed_predict(X_train_sel, y[train_idx], X_test_sel)
        preds = (probs >= THRESHOLD).astype(int)

        # Metrics for this fold
        acc  = accuracy_score(y[test_idx], preds) * 100
        prec = precision_score(y[test_idx], preds, zero_division=0) * 100
        rec  = recall_score(y[test_idx], preds, zero_division=0) * 100
        f1   = f1_score(y[test_idx], preds, zero_division=0) * 100

        try:
            auc = roc_auc_score(y[test_idx], probs) * 100
        except ValueError:
            auc = 0.0

        cm = confusion_matrix(y[test_idx], preds)
        spec = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-12) * 100
        kappa = cohen_kappa_score(y[test_idx], preds) * 100

        fold_result = {
            "Accuracy": acc, "Precision": prec, "Recall": rec,
            "F1_Score": f1, "AUC_ROC": auc, "Specificity": spec, "Kappa": kappa,
        }
        fold_metrics.append(fold_result)

        all_y_true.extend(y[test_idx])
        all_y_pred.extend(preds)
        all_y_prob.extend(probs)

        # Track best fold for model saving
        if f1 > best_fold_f1:
            best_fold_f1 = f1
            best_fold_data = {
                "scaler": scaler,
                "selected": selected,
                "X_train": X_train_sel,
                "X_test": X_test_sel,
                "y_train": y[train_idx],
                "y_test": y[test_idx],
            }

        elapsed = time.time() - t_start
        print(f"  Fold {fold + 1:2d}: "
              f"Acc={acc:.2f}%  Prec={prec:.2f}%  Rec={rec:.2f}%  "
              f"F1={f1:.2f}%  AUC={auc:.2f}%  Thr={THRESHOLD}  "
              f"Feat={len(selected)}  [{elapsed:.0f}s]")

    total_time = time.time() - t_start

    # ── Overall summary ──────────────────────────────────
    results = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        results[f"{key}_mean"] = np.mean(vals)
        results[f"{key}_std"] = np.std(vals)

    overall_cm = confusion_matrix(np.array(all_y_true), np.array(all_y_pred))
    total_errors = int(overall_cm[0, 1] + overall_cm[1, 0])

    print(f"\n{'=' * 60}")
    print("  Step 6: Results & Visualization")
    print("=" * 60)

    print(f"\n{'_' * 60}")
    print(f"  OVERALL ({N_FOLDS}-Fold CV):")
    for key in ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC", "Specificity", "Kappa"]:
        print(f"    {key:13s}: {results[f'{key}_mean']:.2f} +/- {results[f'{key}_std']:.2f}%")
    print(f"  Time: {total_time:.1f}s")

    print(f"\n  Confusion Matrix:")
    print(f"                    Pred Non-Epi  Pred Epi")
    print(f"    Actual Non-Epi    {overall_cm[0, 0]:5d}        {overall_cm[0, 1]:5d}")
    print(f"    Actual Epi        {overall_cm[1, 0]:5d}        {overall_cm[1, 1]:5d}")
    print(f"  Errors: {total_errors}/{len(all_y_true)}")

    # Classification report
    print(f"\n  Per-Class Classification Report:")
    print(classification_report(
        np.array(all_y_true), np.array(all_y_pred),
        target_names=["Non-Epileptic", "Epileptic"], digits=4))

    # Bootstrap CI
    y_true_arr = np.array(all_y_true)
    y_prob_arr = np.array(all_y_prob)

    print("  95% Bootstrap Confidence Intervals (1000 iterations):")
    ci = _bootstrap_ci(y_true_arr, y_prob_arr)
    for metric, (lo, hi) in ci.items():
        print(f"    {metric:10s}: [{lo:.2f}%, {hi:.2f}%]")

    # ── Comparison table ─────────────────────────────────
    if baselines:
        print(f"\n  Comparison with Baselines (all metrics):")
        print(f"  {'Model':40s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  "
              f"{'F1':>6s}  {'AUC':>6s}  {'Spec':>6s}  {'Kappa':>6s}")
        print(f"  {'_' * 90}")

        print(f"\n  Raw signals (before feature engineering):")
        for b in baselines[:3]:
            print(f"  {b['name']:40s}  {b['Acc']:5.2f}%  {b['Prec']:5.2f}%  "
                  f"{b['Rec']:5.2f}%  {b['F1']:5.2f}%  {b['AUC']:5.2f}%  "
                  f"{b['Spec']:5.2f}%  {b['Kappa']:5.2f}%")

        print(f"\n  Engineered features (720 features):")
        for b in baselines[3:]:
            print(f"  {b['name']:40s}  {b['Acc']:5.2f}%  {b['Prec']:5.2f}%  "
                  f"{b['Rec']:5.2f}%  {b['F1']:5.2f}%  {b['AUC']:5.2f}%  "
                  f"{b['Spec']:5.2f}%  {b['Kappa']:5.2f}%")

        print(f"\n  {'_' * 90}")
        print(f"  {'EpiDetect v2 (Ensemble)':40s}  {results['Accuracy_mean']:5.2f}%  "
              f"{results['Precision_mean']:5.2f}%  {results['Recall_mean']:5.2f}%  "
              f"{results['F1_Score_mean']:5.2f}%  {results['AUC_ROC_mean']:5.2f}%  "
              f"{results['Specificity_mean']:5.2f}%  {results['Kappa_mean']:5.2f}%")

    # ── Plots ────────────────────────────────────────────
    _plot_roc_pr(y_true_arr, y_prob_arr)
    _plot_confusion_matrix(overall_cm)
    _plot_metrics_bar(results)

    # ── Save results JSON ────────────────────────────────
    summary = {
        "results": {k: float(v) for k, v in results.items()},
        "confusion_matrix": overall_cm.tolist(),
        "errors": total_errors,
        "total_signals": len(all_y_true),
        "time_seconds": round(total_time, 1),
        "baselines_raw": [b for b in baselines[:3]] if baselines else [],
        "baselines_feat": [b for b in baselines[3:]] if baselines else [],
    }
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # ── Save model ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 8: Saving Model Artifacts")
    print("=" * 60)

    print("  Training final multi-seed model on best fold...")
    final_ensembles = []
    for seed in SEEDS:
        ens = build_ensemble(seed)
        ens.fit(best_fold_data["X_train"], best_fold_data["y_train"])
        final_ensembles.append(ens)
    print(f"  Trained {len(SEEDS)} ensemble models (seeds: {SEEDS})")

    joblib.dump(final_ensembles, os.path.join(MODEL_DIR, "ensemble.pkl"))
    joblib.dump(best_fold_data["scaler"], os.path.join(MODEL_DIR, "scaler.pkl"))
    np.save(os.path.join(MODEL_DIR, "selected_features.npy"),
            np.array(best_fold_data["selected"]))
    print(f"  Saved to {MODEL_DIR}")

    # Attach XAI data for downstream use
    summary["_xai_data"] = {
        "ensemble": final_ensembles[0],
        "X_train": best_fold_data["X_train"],
        "X_test": best_fold_data["X_test"],
        "y_test": best_fold_data["y_test"],
        "selected_features": best_fold_data["selected"],
    }

    return summary


# ── Plotting helpers ─────────────────────────────────────

def _plot_roc_pr(y_true, y_prob):
    """ROC and Precision-Recall curves side by side."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec_curve, prec_curve, color="#7C3AED", lw=2, label=f"PR (AP = {avg_prec:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: roc_pr_curves.png")


def _plot_confusion_matrix(cm):
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Epileptic", "Epileptic"],
                yticklabels=["Non-Epileptic", "Epileptic"],
                annot_kws={"size": 14}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (10-Fold CV)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: confusion_matrix.png")


def _plot_metrics_bar(results):
    """Bar chart of all 7 metrics with error bars."""
    metric_keys = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC", "Specificity", "Kappa"]
    means = [results[f"{k}_mean"] for k in metric_keys]
    stds = [results[f"{k}_std"] for k in metric_keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(metric_keys, means, yerr=stds, color="#2563EB", alpha=0.8,
                  capsize=5, edgecolor="white", linewidth=0.5)
    ax.set_ylim(90, 101)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Model Performance Metrics (10-Fold CV)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: metrics_bar.png")

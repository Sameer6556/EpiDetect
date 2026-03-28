"""
The Explainable AI (XAI) part of the project.
This calculates feature importance and SHAP values so we can understand what the model is looking at.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import RESULTS_DIR


def compute_feature_importance(ensemble, X_train, feature_names=None):
    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    all_importances = []
    for name, estimator in ensemble.named_estimators_.items():
        if hasattr(estimator, "feature_importances_"):
            imp = estimator.feature_importances_
            imp = imp / (imp.sum() + 1e-12)
            all_importances.append(imp)

    if not all_importances:
        return None

    avg_importance = np.mean(all_importances, axis=0)
    top_k = min(20, n_features)
    top_indices = np.argsort(avg_importance)[::-1][:top_k]

    return {
        "importance": avg_importance,
        "feature_names": feature_names,
        "top_indices": top_indices,
    }


def compute_permutation_importance(ensemble, X_test, y_test, feature_names=None, n_repeats=15):
    """
    Permutation importance on the FULL ENSEMBLE using neg_log_loss.
    Accuracy is too robust to show drops when shuffling a single feature in an 8-model voting ensemble.
    Log-loss measures the exact probability confidence, providing a smooth and accurate importance distribution.
    """
    from sklearn.inspection import permutation_importance as sklearn_perm

    n_features = X_test.shape[1]
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    try:
        # Compute directly on the full ensemble
        result = sklearn_perm(ensemble, X_test, y_test,
                              n_repeats=n_repeats, random_state=42,
                              n_jobs=1, scoring="neg_log_loss")
    except Exception as e:
        print("Error computing permutation importance:", e)
        return None

    # The result importances are already the mean accuracy drops
    avg_mean = result.importances_mean
    avg_raw = result.importances

    top_k = min(20, n_features)
    top_indices = np.argsort(avg_mean)[::-1][:top_k]

    return {
        "importances_raw": avg_raw,
        "importance_mean": avg_mean,
        "feature_names": feature_names,
        "top_indices": top_indices,
    }


def compute_multi_tree_shap(ensemble, X_test, feature_names=None):
    try:
        import shap
    except ImportError:
        return None

    n_features = X_test.shape[1]
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    tree_models = ["rf", "et", "gb"]
    all_shap = []

    for name in tree_models:
        estimator = ensemble.named_estimators_.get(name)
        if estimator is None: continue
        try:
            explainer = shap.TreeExplainer(estimator)
            sv = explainer.shap_values(X_test)
            if isinstance(sv, list): sv = sv[1]
            if sv.ndim == 3: sv = sv[:, :, 1]
            all_shap.append(sv)
        except:
            continue

    if not all_shap:
        return None

    return np.mean(all_shap, axis=0)


def plot_feature_importance(importance_data, save_path=None):
    if importance_data is None: return
    top_idx = importance_data["top_indices"]
    names = importance_data["feature_names"]
    imp = importance_data["importance"]

    top_names = [names[i] for i in top_idx][::-1]
    top_values = [imp[i] for i in top_idx][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_names)))
    ax.barh(range(len(top_names)), top_values, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Averaged Importance Score", fontsize=10)
    ax.set_title("Top 20 Features — Tree-Based Importance", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(save_path or os.path.join(RESULTS_DIR, "feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_permutation_importance(perm_data, save_path=None):
    """
    Renders Permutation Importance as a horizontal bar chart with value labels.
    """
    if perm_data is None: return
    top_idx = perm_data["top_indices"]
    names = perm_data["feature_names"]
    avg = perm_data["importance_mean"]

    top_names = [names[i] for i in top_idx][::-1]
    top_values = [avg[i] for i in top_idx][::-1]

    fig, ax = plt.subplots(figsize=(18, 10))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(top_names)))
    bars = ax.barh(range(len(top_names)), top_values,
                   color=colors, edgecolor="none", height=0.65)

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10.5)
    ax.set_xlabel("Log-Loss Degradation When Shuffled", fontsize=11)
    ax.set_title("Top 20 Features — Permutation Importance", fontsize=13,
                 fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    x_max = max(top_values) if top_values else 1e-4
    ax.set_xlim(left=0, right=x_max * 1.18)

    for bar, val in zip(bars, top_values):
        ax.text(val + x_max * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8.5, color="#333333")

    plt.savefig(save_path or os.path.join(RESULTS_DIR, "permutation_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: permutation_importance.png")


def plot_shap_summary(shap_values, X_test, feature_names=None, save_path=None):
    if shap_values is None: return
    try:
        import shap
        import matplotlib.ticker as ticker
    except ImportError:
        return
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

    # Wide figure so long feature names and both ±SHAP sides have room
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        max_display=15,
        plot_size=(22, 10),
        show=False,
    )

    ax = plt.gca()

    # Force symmetric x-axis: show negative AND positive SHAP values
    xabs = max(abs(float(shap_values.min())), abs(float(shap_values.max())))
    pad = xabs * 0.18
    ax.set_xlim(-(xabs + pad), xabs + pad)

    # Symmetric tick marks so both sides are labelled
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, symmetric=True))

    # Dashed reference line at 0
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.45, zorder=0)

    ax.set_title("SHAP Summary Plot (Beeswarm)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=11)

    plt.savefig(
        save_path or os.path.join(RESULTS_DIR, "shap_summary.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


def plot_shap_bar(shap_values, feature_names=None, save_path=None):
    if shap_values is None: return
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_k = min(15, len(mean_abs))
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_idx][::-1]
    top_values = [mean_abs[i] for i in top_idx][::-1]

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.barh(range(len(top_names)), top_values, color='#03A9F4', alpha=0.85, height=0.65)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10.5)
    ax.set_xlabel("mean(|SHAP value|) (average impact on model output)", fontsize=11)
    ax.set_title("Top 15 Features — SHAP Bar Plot (Mean Absolute Impact)", fontsize=13,
                 fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.savefig(save_path or os.path.join(RESULTS_DIR, "shap_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_bar(results, save_path=None):
    keys = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC", "Specificity", "Kappa"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Specificity", "Kappa"]
    vals = [results[f"{k}_mean"] for k in keys]
    colors = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350', '#AB47BC', '#26C6DA', '#FFEE58']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor='none', width=0.6)

    ax.set_ylim(80, 101.5)
    ax.set_yticks(np.arange(80, 101, 2))
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.grid(axis='y', alpha=0.2, linestyle='--')

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.2, f'{v:.2f}%',
                ha='center', fontweight='bold', fontsize=9)

    ax.set_title('Classification Metrics of EpiDetect Ensemble (10-Fold CV)',
                 fontsize=12, fontweight='bold', pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path or os.path.join(RESULTS_DIR, "metrics_bar.png"), dpi=200)
    plt.close()

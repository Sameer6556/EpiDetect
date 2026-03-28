"""
main.py — Main script for training the EpiDetect ensemble model.
This script runs the entire pipeline: loading the dataset, preprocessing,
extracting features, running cross-validation, and saving the final model.
"""

import warnings
import numpy as np

# hiding annoying warnings so the terminal output stays clean
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
np.random.seed(42)

from config.settings import DATA_DIR
from data.loader import load_bonn_dataset
from preprocessing.preprocess import preprocess_signals
from features.extract import extract_features_batch, get_feature_names
from training.cross_validation import run_cross_validation
from xai.explainer import (
    compute_feature_importance,
    compute_permutation_importance,
    compute_multi_tree_shap,
    plot_feature_importance,
    plot_permutation_importance,
    plot_shap_summary,
    plot_shap_bar,
)


def main():
    # ── Step 1 ────────────────────────────────────────────
    print("=" * 60)
    print("  Step 1: Loading Bonn Dataset")
    print("=" * 60)

    X_raw, y = load_bonn_dataset(DATA_DIR)

    if len(y) == 0:
        print("  No data found. Check DATA_DIR in config/settings.py")
        return

    # ── Step 2 ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 2: Preprocessing")
    print("=" * 60)

    X_clean = preprocess_signals(X_raw)
    print(f"  Done: {X_clean.shape}")

    # ── Step 3 ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 3: Feature Extraction (720 features)")
    print("=" * 60)

    X_features = extract_features_batch(X_clean)
    feature_names = get_feature_names()
    print(f"  Feature names count: {len(feature_names)}")

    # Pass features to cross-validation (this handles training and saving)
    results = run_cross_validation(X_features, y, X_raw=X_clean)

    # ── Step 7: XAI ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 7: XAI Explanations")
    print("=" * 60)

    xai_data = results.get("_xai_data")
    if xai_data:
        ensemble = xai_data["ensemble"]
        X_train = xai_data["X_train"]
        X_test = xai_data["X_test"]
        y_test = xai_data["y_test"]
        sel_idx = xai_data["selected_features"]
        sel_names = [feature_names[i] for i in sel_idx]

        # Calculate standard feature importance
        print("\n  Computing tree-based feature importance...")
        imp_data = compute_feature_importance(ensemble, X_train, feature_names=sel_names)
        plot_feature_importance(imp_data)

        # Calculate permutation importance across the whole ensemble
        print("  Computing permutation importance for full ensemble...")
        perm_data = compute_permutation_importance(
            ensemble, X_test, y_test, feature_names=sel_names)
        plot_permutation_importance(perm_data)

        # Get SHAP values for the tree-based models
        print("\n  Computing multi-tree SHAP values...")
        print(f"  Computing SHAP across: rf, et, gb")
        shap_values = compute_multi_tree_shap(ensemble, X_test, feature_names=sel_names)
        if shap_values is not None:
            plot_shap_summary(shap_values, X_test, feature_names=sel_names)
            plot_shap_bar(shap_values, feature_names=sel_names)
    else:
        print("  Skipped — no trained model data available.")

    # ── Final summary ────────────────────────────────────
    r = results["results"]
    print(f"\n{'=' * 60}")
    print("  EpiDetect v2 Pipeline Complete!")
    print("=" * 60)
    print(f"  Accuracy:    {r['Accuracy_mean']:.2f} +/- {r['Accuracy_std']:.2f}%")
    print(f"  Precision:   {r['Precision_mean']:.2f} +/- {r['Precision_std']:.2f}%")
    print(f"  Recall:      {r['Recall_mean']:.2f} +/- {r['Recall_std']:.2f}%")
    print(f"  F1 Score:    {r['F1_Score_mean']:.2f} +/- {r['F1_Score_std']:.2f}%")
    print(f"  AUC-ROC:     {r['AUC_ROC_mean']:.2f} +/- {r['AUC_ROC_std']:.2f}%")
    print(f"  Errors:      {results['errors']}/{results['total_signals']}")
    print(f"  Features:    720")
    print(f"  Time:        {results['time_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

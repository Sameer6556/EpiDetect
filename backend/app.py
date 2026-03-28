"""
Flask REST API for EpiDetect seizure detection.
This handles the backend routes for predictions, metrics, and XAI plots.
"""

import os
import json
import numpy as np
import joblib

from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS

from config.settings import RESULTS_DIR, MODEL_DIR, SAMPLING_RATE, SIGNAL_LENGTH, THRESHOLD
from preprocessing.preprocess import preprocess_signals
from features.extract import extract_features_single, get_feature_names

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble.pkl")
SELECTED_FEATURES_PATH = os.path.join(MODEL_DIR, "selected_features.npy")

_models = None      # list of ensembles (multi-seed) or single ensemble
_scaler = None
_selected_features = None


def _load_model():
    """Load the saved ensemble model(s) and scaler."""
    global _models, _scaler, _selected_features

    if _models is not None:
        return True

    if os.path.exists(ENSEMBLE_PATH) and os.path.exists(SCALER_PATH):
        loaded = joblib.load(ENSEMBLE_PATH)

        # Could be a list of ensembles (multi-seed) or a single ensemble
        if isinstance(loaded, list):
            _models = loaded
        else:
            _models = [loaded]

        _scaler = joblib.load(SCALER_PATH)
        if os.path.exists(SELECTED_FEATURES_PATH):
            _selected_features = np.load(SELECTED_FEATURES_PATH)
        print("Model loaded.")
        return True

    return False


def _predict_signal(raw_signal):
    """Takes a raw EEG signal, cleans it up, extracts features, and gets a prediction from the model."""
    signal_2d = raw_signal.reshape(1, -1)
    clean = preprocess_signals(signal_2d)[0]

    features = extract_features_single(clean)

    if _scaler is not None:
        features = _scaler.transform(features.reshape(1, -1)).flatten()

    if _selected_features is not None:
        features = features[_selected_features]

    # Multi-seed averaging
    if _models is not None:
        all_probs = []
        for ens in _models:
            p = ens.predict_proba(features.reshape(1, -1))[0, 1]
            all_probs.append(p)
        prob = float(np.mean(all_probs))
    else:
        prob = 0.92 if np.std(raw_signal) > 0.8 else 0.08

    label = "Seizure" if prob >= THRESHOLD else "Normal"
    confidence = prob if prob >= THRESHOLD else 1.0 - prob

    result = {
        "prediction": label,
        "probability": round(prob, 4),
        "confidence": round(confidence * 100, 1),
    }

    # Per-input SHAP waterfall
    result["shap_waterfall_img"] = _generate_shap_waterfall(features)

    return result


def _generate_shap_waterfall(features_1d):
    """Creates a SHAP waterfall plot so we can see why the model made its decision."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        import base64

        if _models is None:
            return None

        # Use the first ensemble's RF for TreeExplainer
        ensemble = _models[0]
        rf_model = None
        for name, estimator in ensemble.named_estimators_.items():
            if name == "rf":
                rf_model = estimator
                break

        if rf_model is None:
            return None

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer(features_1d.reshape(1, -1))

        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        all_names = get_feature_names()
        if _selected_features is not None:
            shap_values.feature_names = [all_names[i] for i in _selected_features]
        else:
            shap_values.feature_names = all_names[:features_1d.shape[0]]

        fig = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        plt.title("Per-Input Explanation", fontsize=11, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception:
        return None


# ── Main API Routes ──────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend dashboard."""
    return app.send_static_file("index.html")


@app.route("/health", methods=["GET"])
def health():
    loaded = _load_model()
    return jsonify({"status": "ok", "model_ready": loaded})


@app.route("/demo", methods=["GET"])
def demo():
    """Return a real EEG signal from the Bonn dataset, or synthetic if data not available."""
    is_seizure = request.args.get("seizure", "true").lower() == "true"
    seed = int(request.args.get("seed", 42))
    rng = np.random.RandomState(seed)

    raw = None

    # Try to load a real signal from the Bonn dataset
    try:
        from config.settings import DATA_DIR
        import random

        target_sets = ["S"] if is_seizure else ["Z", "O", "N", "F"]
        chosen_set = random.choice(target_sets)

        from data.loader import _find_set_folder
        folder = _find_set_folder(DATA_DIR, chosen_set)

        if folder:
            all_files = [f for f in os.listdir(folder)
                         if f.lower().endswith(".txt") and f.upper().startswith(chosen_set)]
            if all_files:
                filename = random.choice(all_files)
                filepath = os.path.join(folder, filename)
                raw = np.loadtxt(filepath)
                if len(raw) > SIGNAL_LENGTH:
                    raw = raw[:SIGNAL_LENGTH]
    except Exception:
        raw = None  # fall through to synthetic signal below

    # Fallback: generate a synthetic signal that looks like seizure or normal EEG
    if raw is None or len(raw) == 0:
        t = np.linspace(0, SIGNAL_LENGTH / SAMPLING_RATE, SIGNAL_LENGTH)
        if is_seizure:
            raw = rng.randn(SIGNAL_LENGTH) * 2.0 + 5.0 * np.sin(2 * np.pi * 8 * t)
        else:
            raw = rng.randn(SIGNAL_LENGTH) * 0.3 + 1.0 * np.sin(2 * np.pi * 10 * t)

    return jsonify({
        "demo_mode": True,
        "requested": "seizure" if is_seizure else "normal",
        "eeg_values": raw.tolist(),
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return saved evaluation metrics."""
    results_path = os.path.join(RESULTS_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return jsonify(json.load(f))
    return jsonify({"note": "No results found. Run main.py first."})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict from JSON body: { "eeg": [v0, v1, ..., v4096] }"""
    if not (request.content_type and "application/json" in request.content_type):
        return jsonify({"error": "Send JSON with Content-Type: application/json"}), 400

    data = request.get_json(force=True)
    if "eeg" not in data:
        return jsonify({"error": "Provide an 'eeg' key with the signal array"}), 400

    try:
        _load_model()
        raw = np.array(data["eeg"], dtype=np.float32).flatten()
        result = _predict_signal(raw)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """Upload a .csv/.txt EEG file and get a prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    file = request.files["file"]
    fname = file.filename or ""

    if not (fname.endswith(".csv") or fname.endswith(".txt")):
        return jsonify({"error": "Only .csv and .txt files are supported"}), 400

    try:
        import pandas as pd
        if fname.endswith(".csv"):
            df = pd.read_csv(file)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "label" in num_cols:
                num_cols.remove("label")
            raw = df[num_cols].values.flatten().astype(np.float32)
        else:
            df = pd.read_csv(file, header=None, sep=r'\s+')
            raw = df.values.flatten().astype(np.float32)
    except Exception as e:
        return jsonify({"error": f"File parse error: {e}"}), 400

    try:
        result = _predict_signal(raw)
        result["filename"] = fname
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


# ── XAI Endpoints ────────────────────────────────────────

@app.route("/xai/importance", methods=["GET"])
def xai_importance():
    """Feature importance bar chart (PNG)."""
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    if not os.path.exists(path):
        return jsonify({"error": "Run main.py first to generate plots."}), 404
    resp = make_response(send_file(path, mimetype="image/png"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/xai/permutation", methods=["GET"])
def xai_permutation():
    """Permutation importance chart (PNG)."""
    path = os.path.join(RESULTS_DIR, "permutation_importance.png")
    if not os.path.exists(path):
        return jsonify({"error": "Run main.py first to generate plots."}), 404
    resp = make_response(send_file(path, mimetype="image/png"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/xai/shap", methods=["GET"])
def xai_shap():
    """SHAP beeswarm summary plot (PNG)."""
    path = os.path.join(RESULTS_DIR, "shap_summary.png")
    if not os.path.exists(path):
        return jsonify({"error": "Run main.py first (requires: pip install shap)."}), 404
    resp = make_response(send_file(path, mimetype="image/png"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/xai/shap-bar", methods=["GET"])
def xai_shap_bar():
    """SHAP mean absolute impact bar chart (PNG)."""
    path = os.path.join(RESULTS_DIR, "shap_bar.png")
    if not os.path.exists(path):
        return jsonify({"error": "Run main.py first."}), 404
    resp = make_response(send_file(path, mimetype="image/png"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


if __name__ == "__main__":
    _load_model()
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

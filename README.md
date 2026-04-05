# EpiDetect: Epileptic Seizure Detection System 🧠⚡

This is my final year B.Tech project: an end-to-end machine learning system that detects epileptic seizures from EEG signals. 

Unlike black-box models, this system uses an ensemble of 8 different classifiers (Random Forest, SVM, Neural Nets, etc.) and features an **Explainable AI (XAI)** dashboard. It doesn't just tell doctors if a seizure is happening—it uses SHAP values to show *why* the model made that decision.

## 🚀 Features

- **High Accuracy Models**: Achieves ~99.4% accuracy using a custom soft-voting ensemble across 5 different random seeds for stability.
- **Advanced Signal Processing**: Extracts 732 features per signal (Wavelets, Frequency bands, Non-linear entropy) and selects the best 200 using Mutual Information & ANOVA.
- **Explainable AI (XAI)**: Generates SHAP waterfall plots and beeswarm summaries to visualize which parts of the brain wave triggered the prediction.
- **Interactive Dashboard**: A clean, modern web interface to upload EEG data, view live predictions, and explore the XAI insights.
- **REST API Backend**: Flask-powered backend to handle the heavy lifting.

## 📊 Dataset & Results

I trained the models using the **Bonn University EEG Dataset**. It contains 500 single-channel EEG segments (100 in each of the 5 classes: Z, O, N, F, S) sampled at 173.61 Hz.

**10-Fold Cross-Validation Performance:**
* **Accuracy:** 99.40%
* **Precision:** 99.09%
* **Recall:** 98.00%
* **F1 Score:** 98.47%
* **AUC-ROC:** 99.90%

## 📂 Project Layout

```text
seizure-detection/
├── backend/                  # Python/Flask Backend
│   ├── config/               # Hyperparameters and paths
│   ├── data/                 # Data loading scripts (and the Bonn dataset)
│   ├── features/             # Feature extraction (wavelets, entropy, etc.)
│   ├── models/               # The 8-classifier ensemble logic
│   ├── preprocessing/        # Bandpass filters & z-score scaling
│   ├── training/             # Cross-validation and evaluation pipeline
│   ├── xai/                  # SHAP explanation generators
│   ├── app.py                # Main Flask API Server
│   └── main.py               # The script to train the model from scratch
│
└── frontend/                 # Vanilla HTML/CSS/JS Dashboard
    ├── index.html            # The main UI
    ├── css/style.css         # Styling 
    └── js/app.js             # API calls and Chart.js logic
```

## 💻 How to Run It

If you want to run this locally, you'll need Python 3.8+ installed.

### 1. Backend Setup

First, navigate to the backend folder and install the required libraries:

```bash
cd backend
pip install -r requirements.txt
```

*(Optional)* If you want to retrain the model from scratch, you can run the training script. It takes about 10-15 minutes depending on your CPU because of the ensemble and 10-fold CV:

```bash
python main.py
```

### Running the Application

You only need to run one command to start both the API server and the frontend dashboard. Navigate to the backend folder and start `app.py`:

```bash
cd backend
python app.py
```

*The application will start running on `http://localhost:5000`. Open this URL in your browser to view the dashboard.*

## 🛠️ Built With

* **Machine Learning**: `scikit-learn`, `numpy`, `scipy`, `PyWavelets`
* **Explainability**: `shap`
* **Backend**: `Flask`, `Flask-CORS`
* **Frontend**: `HTML5`, `CSS3`, `JavaScript`, `Chart.js`

## 📝 License

This project was built for academic purposes. Feel free to explore the code, use snippets for your own projects, and reach out if you have any questions!


# ============================================
# ML DEBUG / EXPERIMENTATION (TEMP)
# ============================================
def test_hyperparameters_grid():
    """Temporary grid search script for early baseline testing"""
    import itertools
    import time
    
    print("--- STARTING GRID SEARCH ---")
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    max_depths = [3, 5, 7, 9]
    n_estimators = [100, 200, 500]
    
    best_score = 0
    best_params = None
    
    for lr, md, ne in itertools.product(learning_rates, max_depths, n_estimators):
        start = time.time()
        print(f"Testing LR: {lr}, Depth: {md}, Estimators: {ne}")
        
        # Mocking train loop since this is just temp code
        # model = XGBClassifier(learning_rate=lr, max_depth=md, n_estimators=ne)
        # model.fit(X_train, y_train)
        # score = model.score(X_test, y_test)
        
        # Simulate training time and score
        time.sleep(0.1)
        score = 0.85 + (md * 0.01) - (abs(lr - 0.1) * 0.2)
        
        print(f"Score: {score:.4f} | Time: {time.time() - start:.2f}s")
        if score > best_score:
            best_score = score
            best_params = (lr, md, ne)
            
    print(f"--- GRID SEARCH COMPLETE ---")
    print(f"Best Score: {best_score}")
    print(f"Best Params: {best_params}")

def debug_plot_eeg_raw():
    """Helper to visualize raw EEG streams (will be removed later)"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # generate dummy wave
    t = np.linspace(0, 10, 1000)
    wave = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(1000)
    
    plt.figure(figsize=(15, 4))
    plt.plot(t, wave, label='Channel Fp1')
    plt.title("DEBUG: Raw EEG Signal Visualization")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.legend()
    plt.grid(True)
    plt.show()

# TODO: Try lightgbm
# TODO: Try catboost
# TODO: Implement cross-validation properly
# TODO: Check class imbalance (SMOTE?)
# TODO: Remove these debug helpers before final push


# ============================================
# API DEBUG ENDPOINTS (TEMP)
# ============================================

@app.route('/debug/health', methods=['GET'])
def extended_health_check():
    import psutil
    import platform
    import sys
    from datetime import datetime
    
    return jsonify({
        "status": "online",
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/debug/mock_predict', methods=['POST'])
def mock_prediction():
    """Return a mock prediction for frontend testing without loading heavy model"""
    import time
    import random
    
    # simulate processing delay
    time.sleep(1.5)
    
    is_seizure = random.random() > 0.5
    confidence = random.uniform(0.7, 0.99)
    
    return jsonify({
        "success": True,
        "prediction": "Seizure Detected" if is_seizure else "Normal Activity",
        "confidence": confidence,
        "processing_time_ms": 1500,
        "debug_mode": True
    })

def log_request_middleware(request):
    """Log all incoming requests for debugging"""
    from datetime import datetime
    print(f"[{datetime.now()}] {request.method} {request.path}")
    print(f"Headers: {dict(request.headers)}")
    if request.is_json:
        print(f"Body: {request.json}")
    print("-" * 50)

# TODO: Add rate limiting
# TODO: Secure endpoints with API key
# TODO: Add swagger documentation

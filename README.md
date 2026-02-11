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

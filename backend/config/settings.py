"""
Settings and constants.
I put all the paths, hyperparameters, and dataset info in here 
so I don't have to hardcode them everywhere else.
"""

import os

# Bonn dataset specific properties
SAMPLING_RATE = 173.61          # Hz — sampling frequency of the Bonn recordings
SIGNAL_LENGTH = 4097            # samples per recording (~23.6 seconds)

# Feature engineering parameters
DWT_LEVEL = 4                   # Discrete Wavelet Transform decomposition depth
N_SEGMENTS = 16                 # number of sub-segments each signal is split into

# Model training configuration
N_FOLDS = 10                    # stratified k-fold cross-validation
SEEDS = [42, 123, 456, 789, 2024]
THRESHOLD = 0.5                 # classification decision boundary (don't change)
SEIZURE_WEIGHT = 4.0            # class weight for the minority (seizure) class

# Selecting the best features
FEATURE_SELECT_K = 200          # top-k features per method (MI + F-classif union)

# Dictionary for mapping the folders to 0 or 1 labels
# Binary mapping: 0 = non-epileptic, 1 = epileptic seizure
BINARY_LABELS = {
    "Z": 0,   # Set A — healthy volunteer, eyes open
    "O": 0,   # Set B — healthy volunteer, eyes closed
    "N": 0,   # Set C — epileptic patient, seizure-free zone
    "F": 0,   # Set D — epileptic patient, epileptogenic zone
    "S": 1,   # Set E — epileptic patient, during seizure
}

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_DIR = os.path.join(BASE_DIR, "data", "bonn")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "saved_model")

# Create output directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

"""
This creates our custom 8-model ensemble.
It combines different types of ML models (like Trees, SVMs, and Neural Nets)
and trains them multiple times with different seeds so the final prediction is super stable.
"""

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from config.settings import SEEDS, SEIZURE_WEIGHT


def build_ensemble(seed):
    """
    Build a soft-voting ensemble of 8 classifiers for a given seed.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    VotingClassifier
    """
    cw = {0: 1.0, 1: SEIZURE_WEIGHT}   # upweight the minority class

    estimators = [
        ("hgb", HistGradientBoostingClassifier(
            max_iter=400, max_depth=7, learning_rate=0.05,
            min_samples_leaf=8, l2_regularization=1.0,
            class_weight=cw, random_state=seed,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_split=3,
            min_samples_leaf=1, class_weight=cw,
            random_state=seed, n_jobs=-1,
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=500, max_depth=14, min_samples_split=2,
            min_samples_leaf=1, class_weight=cw,
            random_state=seed, n_jobs=-1,
        )),
        ("svm", SVC(
            kernel="rbf", C=20.0, gamma="scale", probability=True,
            class_weight=cw, random_state=seed,
        )),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            solver="adam", alpha=0.0005, learning_rate="adaptive",
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=seed,
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=seed,
        )),
        ("ada", AdaBoostClassifier(
            n_estimators=200, learning_rate=0.05, random_state=seed,
        )),
        ("knn", KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1,
        )),
    ]

    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def multi_seed_predict(X_train, y_train, X_test):
    """
    Train the ensemble with 5 different seeds and average predictions.

    This reduces the variance from random initialization across
    models like MLP, RF, etc. — making the final prediction
    more stable and reliable.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
    y_train : np.ndarray, shape (n_train,)
    X_test  : np.ndarray, shape (n_test, n_features)

    Returns
    -------
    np.ndarray, shape (n_test,)
        Averaged probability of the sample being an epileptic seizure.
    """
    all_probs = []

    for seed in SEEDS:
        ensemble = build_ensemble(seed)
        ensemble.fit(X_train, y_train)
        probs = ensemble.predict_proba(X_test)[:, 1]
        all_probs.append(probs)

    return np.mean(all_probs, axis=0)

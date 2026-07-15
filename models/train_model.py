"""
train_model.py
---------------
Trains a machine-learning classifier that predicts a country's Human
Development Index (HDI) tier -- Low / Medium / High / Very High -- from
four standard UNDP indicators:

    1. Life expectancy at birth
    2. Mean years of schooling
    3. Expected years of schooling
    4. GNI per capita (PPP $)

Pipeline
========
1. Load the dataset produced by data/generate_dataset.py
2. Exploratory Data Analysis -> saved as PNG charts in /plots
3. Train/test split + StandardScaler
4. Train and compare a few scikit-learn models (Logistic Regression,
   Random Forest, Gradient Boosting) via cross-validation
5. Pick the best model, fit on the full training set, evaluate on the
   held-out test set (accuracy, classification report, confusion matrix)
6. Persist the fitted scaler + model + label encoder to /models as .pkl
   files that the Flask app loads at request time.
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "hdi_dataset.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    "life_expectancy",
    "mean_years_schooling",
    "expected_years_schooling",
    "gni_per_capita",
]
TARGET = "hdi_tier"
TIER_ORDER = ["Low", "Medium", "High", "Very High"]

sns.set_theme(style="whitegrid", palette="viridis")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def run_eda(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, order=TIER_ORDER, hue=TARGET,
                  palette="crest", legend=False)
    plt.title("HDI Tier Distribution")
    plt.xlabel("HDI Tier")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # 2. Feature correlation heatmap
    plt.figure(figsize=(6, 5))
    corr = df[FEATURES + ["hdi_score"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # 3. Pairwise feature distributions by tier
    g = sns.pairplot(df, vars=FEATURES, hue=TARGET, hue_order=TIER_ORDER,
                      palette="viridis", plot_kws={"alpha": 0.4, "s": 15})
    g.fig.suptitle("Indicator Relationships by HDI Tier", y=1.02)
    g.savefig(os.path.join(PLOTS_DIR, "pairplot.png"), dpi=150)
    plt.close("all")

    # 4. Boxplots per feature
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.flatten(), FEATURES):
        sns.boxplot(data=df, x=TARGET, y=feat, order=TIER_ORDER, ax=ax,
                    hue=TARGET, palette="crest", legend=False)
        ax.set_title(feat.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_boxplots.png"), dpi=150)
    plt.close()

    print(f"EDA plots saved to {PLOTS_DIR}")


def train():
    df = load_data()
    run_eda(df)

    X = df[FEATURES].values
    y_raw = df[TARGET].values

    encoder = LabelEncoder()
    encoder.fit(TIER_ORDER)  # fix class order explicitly
    y = encoder.transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    cv_results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                                  scoring="accuracy", n_jobs=-1)
        cv_results[name] = scores.mean()
        print(f"{name}: CV accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_results, key=cv_results.get)
    best_model = candidates[best_name]
    print(f"\nBest model: {best_name}")

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=TIER_ORDER, output_dict=True
    )
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=TIER_ORDER))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=TIER_ORDER, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {best_name} (acc={test_acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance (if supported)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(6, 4))
        sns.barplot(x=importances[order], y=np.array(FEATURES)[order],
                    hue=np.array(FEATURES)[order], palette="crest", legend=False)
        plt.title(f"Feature Importance - {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
        plt.close()

    # Persist artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "hdi_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "tier_order": TIER_ORDER,
        "test_accuracy": round(test_acc, 4),
        "cv_scores": {k: round(v, 4) for k, v in cv_results.items()},
        "classification_report": report,
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model, scaler, encoder, and metadata to {MODELS_DIR}")


if __name__ == "__main__":
    train()

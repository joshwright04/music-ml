from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os

from evaluate_baseline_models import evaluate_model, plot_model_comparison

# Global var for the GTZAN features csv
FEATURES_CSV = "./data/Data/features_30_sec.csv"

def main() -> None:
    """
    Loads the GTZAN features_30_sec.csv file and splits the data 
    into training and testing (and stratifies)

    The models used here are:
        Random Forest Classifier
        Logistic Regression
        Support Vector Machine

    The models trained here are later reused to evaluate
    vae generated songs. Each trained model is saved with the joblib library
    """
    df = pd.read_csv(FEATURES_CSV)

    # Split dataset into features (X) and genre label (y)
    X = df.drop(columns=["label", "filename"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Initialize baseline models 
    # StandardScaler normalizes feature values before classification.
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42)),
        ]),
    }

    results = {}
    
    # Create folder for saved baseline models
    os.makedirs("saved_baseline_models", exist_ok=True)

    for name, model in models.items():
        # Fit each model
        model.fit(X_train, y_train)

        # Predict on the test set
        preds = model.predict(X_test)

        # evaluate the model accuracy
        results[name] = evaluate_model(name, model, y_test, preds)

        safe_name = name.lower().replace(" ", "_")

        # Save trained model with joblib in the saved_baseline_models directory
        joblib.dump(
            model,
            f"saved_baseline_models/{safe_name}.pkl"
        )

        print(f"Saved {name} model.")

    # Plot comparison graph across all 3 models
    plot_model_comparison(results)

if __name__ == "__main__":
    main()
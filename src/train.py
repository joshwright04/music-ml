from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


FEATURES_CSV = "results/features.csv"


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)

    # remove non-feature columns
    X = df.drop(columns=["label", "filename"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # scaler + model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()
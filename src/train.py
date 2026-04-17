from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


FEATURES_CSV = "./data/Data/features_30_sec.csv"


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

    for name, model in models.items():
        results = {}
        print("=" * 60)
        print(f"{name}")
        print("=" * 60)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results[name] = acc
        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=report_df.index[:-3],  # drop avg rows
            y=report_df["recall"][:-3]
        )
        plt.title(f"{name} - Accuracy per Genre")
        plt.xlabel("Genre")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("Accuracy:", accuracy_score(y_test, preds))
        print("\nClassification report:")
        print(classification_report(y_test, preds))
        print("\nConfusion matrix:")
        cm = confusion_matrix(y_test, preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
        )
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
        print("\n")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title("Model Comparison (Accuracy)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
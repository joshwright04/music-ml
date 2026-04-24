from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Directory where baseline model evaluation plots are saved
output_dir = Path("original_basic_plots")

def evaluate_model(name, model, y_test, preds) -> float:
    """
    Evaluate a baseline classification model.

    This function:
    - Prints accuracy
    - Prints a full classification report
    - Saves a genre recall bar chart png to 'original_baseline_plots/'
    - Saves a confusion matrix heatmap to 'original_baseline_plots/'

    Returns the model in question's accuracy score as a float.
    """
    # Create the directory if it doesn't already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # clean model name for filenames
    safe_name = name.lower().replace(" ", "_")

    # Clean terminal output
    print("=" * 60)
    print(name)
    print("=" * 60)

    # Generate accuracy score
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, preds))
    print("\nConfusion matrix:")

    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Genre recall/recognition plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=report_df.index[:-3],
        y=report_df["recall"][:-3]
    )
    plt.title(f"{name} - Genre Recognition Rate")
    plt.xlabel("Genre")
    plt.ylabel("Recall")
    plt.xticks(rotation=45)
    plt.tight_layout()

    recall_path = output_dir / f"{safe_name}_genre_recall.png"
    plt.savefig(recall_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion matrix plot
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

    cm_path = output_dir / f"{safe_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plots:")
    print(f" - {recall_path}")
    print(f" - {cm_path}")
    print("\n")

    # Return the model's accuracy (for model comparison plot later on)
    return acc


def plot_model_comparison(results: dict[str, float]) -> None:
    """
    Plot a single png comparing the accuracy of each model
    """
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title("Model Comparison (Accuracy)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    path = output_dir / "model_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")

    print(f"Saved plots:")
    print(f" - {path}")
    print("\n")
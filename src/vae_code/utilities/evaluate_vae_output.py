import os
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_vae_output_on_baseline_models(
    baseline_models: list,
    generated_features_path: str,
    target_genre: str,
    output_dir: str,
) -> None:
    if not os.path.exists(generated_features_path):
        print("Generated features CSV not found.")
        return

    df = pd.read_csv(generated_features_path)

    if df.empty:
        print("Generated features CSV is empty.")
        return

    X_generated = df.drop(columns=["label", "filename"])

    results = {}

    print(f"\nTarget genre: {target_genre}\n")

    for model in baseline_models:
        model_name = model.named_steps["clf"].__class__.__name__
        predictions = model.predict(X_generated)

        correct = sum(pred == target_genre for pred in predictions)
        accuracy = correct / len(predictions)

        results[model_name] = accuracy * 100

        print(f"{model_name}: {correct}/{len(predictions)} correct ({accuracy:.2%})")

        for filename, pred in zip(df["filename"], predictions):
            print(f"  {filename} -> {pred}")

        print()

    save_accuracy_graph(results, output_dir)

def save_accuracy_graph(
    results: dict,
    output_dir: str
) -> None:
    model_names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, accuracies)

    plt.ylabel("Accuracy (%)")
    plt.title("Baseline Model Accuracy on VAE Generated Songs")
    plt.ylim(0, 100)

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "baseline_model_accuracy.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved graph to {output_path}")
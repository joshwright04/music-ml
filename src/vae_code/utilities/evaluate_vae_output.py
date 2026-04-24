import os
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_vae_output_on_baseline_models(
    baseline_models: list,
    generated_features_path: str,
    target_genre: str,
    output_dir: str,
) -> None:
    """
    Evaluate generated vae songs using the previously trained
    baseline classification models.

    Each model attempts to classify the generated songs and accuracy is recorded.
    """
    # Error handling
    if not os.path.exists(generated_features_path):
        print("Generated features CSV not found.")
        return

    # Load the generated feature dataset
    df = pd.read_csv(generated_features_path)

    # More error handling
    if df.empty:
        print("Generated features CSV is empty.")
        return

    # Drop unused features
    X_generated = df.drop(columns=["label", "filename"])

    results = {}

    print(f"\nTarget genre: {target_genre}\n")

    # For each baseline model, predict on each row (song) in the df
    for model in baseline_models:
        model_name = model.named_steps["clf"].__class__.__name__
        predictions = model.predict(X_generated)

        # Print accuracy to the terminal
        correct = sum(pred == target_genre for pred in predictions)
        accuracy = correct / len(predictions)

        results[model_name] = accuracy * 100

        print(f"{model_name}: {correct}/{len(predictions)} correct ({accuracy:.2%})")

        for filename, pred in zip(df["filename"], predictions):
            print(f"  {filename} -> {pred}\n")

    # Plot the results in the same directory as the audio files themselves
    save_accuracy_graph(results, output_dir)

def save_accuracy_graph(
    results: dict,
    output_dir: str
) -> None:
    """
    Create and save a bar chart showing how accurately
    each baseline model classified the generated songs.
    Graph is saved to the same directory as the csv and .wav files
    """
    model_names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, accuracies)

    plt.ylabel("Accuracy (%)")
    plt.title("Baseline Model Accuracy on VAE Generated Songs")

    # Might not be needed but accuracy should always be between 0 and 100
    plt.ylim(0, 100)

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "baseline_model_accuracy.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved graph to {output_path}")
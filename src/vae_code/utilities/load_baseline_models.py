import joblib

def load_baseline_models() -> tuple:
    models = [
        joblib.load(
            "saved_baseline_models/random_forest.pkl"
        ),
        joblib.load(
            "saved_baseline_models/logistic_regression.pkl"
        ),
        joblib.load(
            "saved_baseline_models/svm.pkl"
        ),
    ]

    print("Baseline models loaded successfully.")

    return models
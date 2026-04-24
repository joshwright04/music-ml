import joblib

def load_baseline_models() -> tuple:
    """
    Load the existing baseline models to evaluate vae output on. 
    Currently, the three models we expect to find are:
    Random forest classifier
    Logistic regression classifier
    Support Vector Machine
    """
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
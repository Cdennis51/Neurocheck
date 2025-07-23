import pandas as pd
import mlflow.xgboost
from preprocess import filter_to_eeg_channels

def preprocess_input(file_path):
    # Load and filter raw CSV
    df = pd.read_csv(file_path)
    X = filter_to_eeg_channels(df)

    # Reduce time-series to single row (mean across time)
    X_input = X.mean().to_frame().T
    return X_input

def load_model():
    return mlflow.xgboost.load_model("runs:/<your_run_id>/model")  # Replace with your actual run ID

def predict(file_path):
    X_input = preprocess_input(file_path)
    model = load_model()
    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)
    return {
        "prediction": int(y_pred[0]),
        "confidence": float(max(y_proba[0]))
    }

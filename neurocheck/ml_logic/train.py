import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import filter_to_eeg_channels
from neurocheck.ml_logic.model import initialize_xgb_model, train_model
from registry import save_results, save_model

# Config
RAW_DATA_PATH = "/Users/majamielke/code/Cdennis51/Neurocheck/raw_data/MEFAR_preprocessed/MEFAR_MID.csv"

# Load raw data
df = pd.read_csv(RAW_DATA_PATH)
print("✅ Raw data loaded")

# Target & Features
y = df["class"]
X = filter_to_eeg_channels(df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = initialize_xgb_model()
train_model(X_train, y_train, model)
print("✅ Model trained")

# TODO Save metrics & model
"""save_results(params=model.get_params(), metrics={
    "train_accuracy": model.score(X_train, y_train),
    "test_accuracy": model.score(X_test, y_test)
})"""

save_model(model)
print("✅ Model and results saved")

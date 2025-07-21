import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from neurocheck.ml_logic.registry import save_model
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def initialize_xgb_model(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=1.0,
    random_state=42
):
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state
    )
    return model

def train_model(X_train, y_train, model):
    """
    Trains the model on the given data and saves it as a pickle file.

    Parameters:
    - X_train: training features
    - y_train: training labels
    - model: sklearn or XGBoost model instance
    - model_path: file path to save the trained model
    """
    model = model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data.
    Prints classification report, accuracy, and confusion matrix.
    Returns predicted labels.
    """
    y_pred = model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return y_pred

def predict(frontend_data_preprocessed: pd.DataFrame, model) -> dict:
    """
    This function takes the preprocessed user data and predicts mental fatigue using our model.
    It returns a prediction and scoring metrics.
    """
    # Type check
    if not isinstance(frontend_data_preprocessed, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    #test shape of frontend_data_preprocessed
    assert frontend_data_preprocessed.shape[0] == 1, "Preprocessed DataFrame must have exactly one row"

    # TODO: maybe add feature check?

    # predict
    y_pred = model.predict(frontend_data_preprocessed)
    y_proba = model.predict_proba(frontend_data_preprocessed) if hasattr(model, 'predict_proba') else None

    result = {
        "prediction": int(y_pred[0])
    }

    if y_proba is not None:
        result["confidence"] = float(max(y_proba[0]))

    return result

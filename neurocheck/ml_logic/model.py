import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from neurocheck.ml_logic.registry import save_model

def initialize_model(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42, class_weight='balanced'):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight=class_weight
        )
    return model

def train_model(X_train, y_train, model):
    """
    This function takes the preprocessed features and trains a model.
    It then saves the model as a pickle file.
    """
    model = model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    This function takes the trained model and evaluates it on X_test.
    It returns the classification outcome and gives a classification report.
    """
    y_pred = model.predict(X_test)

    print(classification_report(y_test,y_pred))

    return y_pred

def predict(frontend_data_preprocessed):
    """
    This function takes the preprocessed user data and predicts mental fatigue using our model.
    It returns a prediction and scoring metrics.
    """
    #load model from mlflow?

    #test shape of frontend_data_preprocessed
    try:
        assert(frontend_data_preprocessed.shape == )
    except:
        print(f"The prediciton data doesn't have the correct shape. Current shape: {frontend_data_preprocessed.shape}")

    #predict 'class'
    y_pred = model.predict(frontend_data_preprocessed)

    #return prediction metrics?

    return y_pred

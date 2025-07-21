import pandas as pd
import numpy as np

from neurocheck.ml_logic.preprocess import preprocess_predict
from neurocheck.ml_logic.model import predict
from neurocheck.ml_logic.registry import save_results, save_model, retrieve_model


def preprocess(frontend_data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes user input and preprocesses it for prediction.
    Returns a dataframe with one row having one value per feature.
    """

    preprocessed_prediction_data = preprocess_predict(frontend_data)

    # Ensure the output is a DataFrame
    if isinstance(preprocessed_prediction_data, pd.DataFrame):
        X_pred = preprocessed_prediction_data
    elif isinstance(preprocessed_prediction_data, dict):
        X_pred = pd.DataFrame([preprocessed_prediction_data])  # Wrap dict in list
    elif isinstance(preprocessed_prediction_data, pd.Series):
        X_pred = preprocessed_prediction_data.to_frame().T  # Transpose to make it one row
    else:
        raise TypeError("Preprocessed data must be a DataFrame, Series, or dict.")

    assert X_pred.shape[0] == 1, "Preprocessed DataFrame must have exactly one row"

    print("✅ preprocess() done \n")

    return X_pred

def prediction_result(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Takes the preprocessed user data and makes a prediciton using the current production model.
    Returns the predicted class.
    """
    model = retrieve_model(stage="Production")

    prediction = predict(X_pred, model) #TODO: this needs to be adapted with the modules (correct in model.py)

    return prediction

if __name__ == '__main__':
    preprocess()
    predict()
    #evaluate()

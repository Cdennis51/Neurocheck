from xgboost import XGBClassifier
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow.xgboost
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV # Additional step in the initialize xgb boost.

def train_model(X_train, y_train, model):

    model.fit(X_train, y_train)




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
    model  = CalibratedClassifierCV(model, method='isotonic', cv=3)

    return model





# We can remove this predict function, as the predict is directly running in api_file_MM.py
#def predict(frontend_data_preprocessed: pd.DataFrame, model) -> dict:
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

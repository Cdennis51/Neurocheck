from xgboost import XGBClassifier
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow.xgboost


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
    return model



# I think we can remove this from this module - it is in it's own python package predict_TT
def predict(frontend_data_preprocessed, expected_shape=(1, 10), model_path="trained_model.pkl"):
    """
    Predicts mental fatigue from preprocessed frontend input.

    Parameters:
    - frontend_data_preprocessed: np.array or pd.DataFrame of shape (1, n_features)
    - expected_shape: shape to validate against
    - model_path: path to the saved model

    Returns:
    - y_pred: predicted label
    - y_proba (optional): probability of class 1 (if supported)
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Check shape
    if frontend_data_preprocessed.shape != expected_shape:
        print(f"[❌] Input shape mismatch. Expected {expected_shape}, got {frontend_data_preprocessed.shape}")
        return None

    # Predict
    y_pred = model.predict(frontend_data_preprocessed)

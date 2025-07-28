from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
   #model  = CalibratedClassifierCV(model, method='isotonic', cv=3)
# improves the probability estimate by calibrating using cross-validation.
    return model

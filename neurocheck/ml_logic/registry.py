"""
MLflow Utilities for Neurocheck Project

This module provides helper functions for:
- Logging training parameters and metrics to MLflow.
- Saving and versioning XGBoost models in MLflow.
- Retrieving models from local storage or MLflow registry.
- Wrapping functions with MLflow autologging for TensorFlow.

Dependencies:
    - glob
    - os
    - colorama
    - mlflow
    - mlflow.xgboost
    - params.py (should define MODEL_TARGET, LOCAL_REGISTRY_PATH, MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME, MLFLOW_EXPERIMENT)

Usage:
    from mlflow_utils import save_results, save_model, retrieve_model, mlflow_run
"""
import glob
import os
from colorama import Fore, Style
from ml_logic.params import *
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

def save_results(params: dict, metrics: dict):
    """
    Log training parameters and metrics to MLflow.

    Args:
        params (dict): Dictionary of hyperparameters used for training.
        metrics (dict): Dictionary of evaluation metrics (e.g., accuracy, loss).

    Returns:
        None
    """
    mlflow.set_experiment("neurocheck_experiment")  # Set correct experiment by name
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        print("Results logged to MLflow")
    return None

def save_model(model):
    """
    Save an XGBoost model to MLflow and transition it to the Production stage.

    Logs all model parameters (unless too long) and registers the model
    in the MLflow Model Registry as `neurocheck_model`. If a previous
    version exists in Production, it will be archived.

    Args:
        model: Trained XGBoost model instance.

    Returns:
        None
    """
    mlflow.set_experiment("neurocheck_experiment")
    with mlflow.start_run():
        for k, v in model.get_params().items():
            val_str = str(v)
            if len(val_str) <= 500:
                mlflow.log_param(k, v)
            else:
                print(f"Skipping param '{k}': too long or complex")

        mlflow.xgboost.log_model(model,
                                 artifact_path="model",
                                 registered_model_name="neurocheck_model")

        print("✅ Model saved to MLflow")

        # Transition to production
        client = MlflowClient()
        latest_version = client.get_latest_versions("neurocheck_model", stages=["None"])[0].version
        client.transition_model_version_stage(
            name="neurocheck_model",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )

        print("Model logged and saved to MLflow.")

def retrieve_model(stage="Production"):
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + "\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + "\nLoad latest model from disk..." + Style.RESET_ALL)

        # ✅ Load as XGBoost model, not PyFunc
        latest_model = mlflow.xgboost.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "mlflow":
        print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
            model_uri = model_versions[0].source

            assert model_uri is not None
        except:
            print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")
            return None

        # ✅ Load as XGBoost model, not PyFunc
        latest_model = mlflow.xgboost.load_model(model_uri)

        print("✅ Model loaded from MLflow")
        return latest_model
    else:
        return None

def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper

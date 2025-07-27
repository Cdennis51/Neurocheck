"""
Environment Configuration Loader and Validator

This script loads environment variables with defaults for both local development
and cloud deployment (e.g., Cloud Run). It supports optional loading of a local `.env`
file if present, without breaking on platforms where dotenv is not installed.

Main features:
- Loads key environment variables related to MLflow tracking, GCP settings, and local paths.
- Constructs user-specific MLflow experiment and model names to avoid conflicts during development.
- Defines default paths for storing local data and MLflow registry outputs.
- Validates environment variables against predefined allowed values, raising an error if invalid.

Usage:
- Intended to be imported and used to provide configuration variables.
- Will raise ValueError if certain environment variables contain invalid values.

Variables loaded:
- MODEL_TARGET: Deployment target environment, either "local" or "mlflow" (default "local").
- MLFLOW_TRACKING_URI: URI for MLflow tracking server (default "http://127.0.0.1:5000").
- USER: Username for isolating experiments locally (default "defaultuser").
- MLFLOW_EXPERIMENT and MLFLOW_MODEL_NAME: MLflow experiment and model names scoped by user.
- GCP_PROJECT and GCP_REGION: Google Cloud project and region settings (with local defaults).
- LOCAL_DATA_PATH and LOCAL_REGISTRY_PATH: File system paths for data and model artifacts.

Raises:
- ValueError: if any validated environment variable has an invalid value.
"""
import os
# Load .env if detected in current environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

##################  ENV VARIABLES ##################
MODEL_TARGET = os.getenv("MODEL_TARGET", "local")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "neurocheck", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "neurocheck", "training_outputs")

################## VALIDATIONS #################
env_valid_options = dict(
    MODEL_TARGET=["local", "mlflow"],
)

def validate_env_value(environment, valid_options):
    """
    Validate that an environment variable has an allowed value.

    This function retrieves the value of a given environment variable and checks
    whether it is included in the list of valid options. If the value is invalid,
    it raises a NameError with a descriptive message.

    Args:
        environment (str): The name of the environment variable to validate.
        valid_options (list[str]): A list of allowed values for the environment variable.

    Raises:
        KeyError: If the environment variable is not set in the environment.
        NameError: If the environment variable is set but its value is not in `valid_options`.
    """
    env_value = os.environ[environment]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {environment} in `.env` file:\
                        {env_value} must be in {valid_options}")

for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)

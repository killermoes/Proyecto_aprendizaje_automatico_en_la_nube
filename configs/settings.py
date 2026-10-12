import os
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_preparado.csv"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
MLFLOW_EXPERIMENT_NAME = "telco-churn-advanced"
RANDOM_STATE = 31
TEST_SIZE = 0.2
OPTUNA_TRIALS = 20
CV_FOLDS = 3
REGISTERED_MODEL_NAME = "telco-churn-classifier"
REGISTERED_MODEL_ALIAS = "champion"


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_mlops.pipeline import register_champion_model
from configs import REGISTERED_MODEL_ALIAS, REGISTERED_MODEL_NAME


class ModelRegistrationTests(unittest.TestCase):
    @patch("churn_mlops.pipeline.get_run_logger")
    @patch("churn_mlops.pipeline.MlflowClient")
    @patch("churn_mlops.pipeline.mlflow.register_model")
    def test_registers_version_and_updates_champion_alias(
        self,
        register_model: object,
        client_class: object,
        get_run_logger: object,
    ) -> None:
        register_model.return_value = SimpleNamespace(version="4")
        client = client_class.return_value

        register_champion_model.fn("run-id-123", 0.641)

        register_model.assert_called_once_with(
            model_uri="runs:/run-id-123/modelo",
            name=REGISTERED_MODEL_NAME,
            await_registration_for=60,
        )
        client.set_model_version_tag.assert_called_once_with(
            name=REGISTERED_MODEL_NAME,
            version="4",
            key="validation_cv_f1_yes",
            value="0.641000",
        )
        client.set_registered_model_alias.assert_called_once_with(
            name=REGISTERED_MODEL_NAME,
            alias=REGISTERED_MODEL_ALIAS,
            version="4",
        )
        get_run_logger.assert_called_once()


if __name__ == "__main__":
    unittest.main()

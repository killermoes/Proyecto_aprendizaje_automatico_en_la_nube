import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from mlflow.types import DataType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_mlops.batch_predict import predict_batch


class BatchPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.input_path = self.directory / "input.csv"
        self.output_path = self.directory / "output.csv"
        self.input_data = pd.DataFrame(
            {
                "customerID": ["customer-1", "customer-2"],
                "tenure": [1, 12],
                "gender": ["Female", "Male"],
                "Churn": ["Yes", "No"],
            }
        )
        self.input_data.to_csv(self.input_path, index=False)

        self.input_schema = SimpleNamespace(
            inputs=[
                SimpleNamespace(name="tenure", type=DataType.double),
                SimpleNamespace(name="gender", type=DataType.string),
            ],
            input_names=lambda: ["tenure", "gender"],
        )
        self.model = SimpleNamespace(
            metadata=SimpleNamespace(
                get_input_schema=lambda: self.input_schema,
            ),
            predict=lambda features: self.predict(features),
        )
        self.received_features = None

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def predict(self, features: pd.DataFrame) -> list[int]:
        self.received_features = features.copy()
        return [1, 0]

    @patch("churn_mlops.batch_predict.mlflow.set_tracking_uri")
    @patch("churn_mlops.batch_predict.mlflow.pyfunc.load_model")
    @patch("churn_mlops.batch_predict.mlflow.start_run")
    @patch("churn_mlops.batch_predict.mlflow.log_metrics")
    @patch("churn_mlops.batch_predict.mlflow.log_params")
    @patch("churn_mlops.batch_predict.mlflow.set_tags")
    @patch("churn_mlops.batch_predict.mlflow.set_experiment")
    def test_writes_predictions_and_excludes_target(
        self,
        set_experiment: object,
        set_tags: object,
        log_params: object,
        log_metrics: object,
        start_run: object,
        load_model: object,
        set_tracking_uri: object,
    ) -> None:
        load_model.return_value = self.model

        result = predict_batch(self.input_path, self.output_path)
        saved_result = pd.read_csv(self.output_path)

        self.assertEqual(result["churn_prediction"].tolist(), ["Yes", "No"])
        self.assertEqual(
            saved_result["customerID"].tolist(),
            ["customer-1", "customer-2"],
        )
        self.assertNotIn("Churn", self.received_features.columns)
        self.assertEqual(str(self.received_features["tenure"].dtype), "float64")
        self.assertEqual(saved_result["churn_prediction"].tolist(), ["Yes", "No"])
        load_model.assert_called_once()
        set_tracking_uri.assert_called_once()
        set_experiment.assert_called_once_with("telco-churn-advanced")
        set_tags.assert_called_once_with({"inference_type": "batch"})
        log_params.assert_called_once_with(
            {"model_uri": "models:/telco-churn-classifier@champion"}
        )
        metrics = log_metrics.call_args.args[0]
        self.assertEqual(metrics["batch_input_rows"], 2.0)
        self.assertEqual(metrics["predictions_yes"], 1.0)
        self.assertEqual(metrics["predictions_no"], 1.0)
        self.assertEqual(metrics["prediction_yes_rate"], 0.5)
        self.assertEqual(metrics["input_missing_values"], 0.0)
        start_run.assert_called_once_with(run_name="batch_inference")

    @patch("churn_mlops.batch_predict.mlflow.set_tracking_uri")
    @patch("churn_mlops.batch_predict.mlflow.pyfunc.load_model")
    @patch("churn_mlops.batch_predict.mlflow.set_experiment")
    def test_rejects_missing_model_features(
        self,
        set_experiment: object,
        load_model: object,
        set_tracking_uri: object,
    ) -> None:
        load_model.return_value = self.model
        pd.DataFrame({"gender": ["Female"]}).to_csv(self.input_path, index=False)

        with self.assertRaisesRegex(ValueError, "missing model features"):
            predict_batch(self.input_path, self.output_path)

        self.assertFalse(self.output_path.exists())
        set_tracking_uri.assert_called_once()
        set_experiment.assert_not_called()

    def test_rejects_empty_input(self) -> None:
        pd.DataFrame(columns=["tenure", "gender"]).to_csv(
            self.input_path, index=False
        )

        with self.assertRaisesRegex(ValueError, "contains no rows"):
            predict_batch(self.input_path, self.output_path)

        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()

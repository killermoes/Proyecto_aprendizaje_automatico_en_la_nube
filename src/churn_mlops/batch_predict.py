import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from configs import MLFLOW_EXPERIMENT_NAME


MODEL_URI = "models:/telco-churn-classifier@champion"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
METADATA_COLUMNS = {"customerID", "Churn"}
PREDICTION_COLUMN = "churn_prediction"


def predict_batch(
    input_path: Path,
    output_path: Path,
    model_uri: str = MODEL_URI,
) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    input_data = pd.read_csv(input_path)
    if input_data.empty:
        raise ValueError(f"Input CSV contains no rows: {input_path}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(model_uri)
    input_schema = model.metadata.get_input_schema()
    if input_schema is None:
        raise ValueError(f"Registered model has no input signature: {model_uri}")

    feature_columns = input_schema.input_names()
    missing_columns = sorted(set(feature_columns) - set(input_data.columns))
    extra_columns = sorted(
        set(input_data.columns) - set(feature_columns) - METADATA_COLUMNS
    )
    if missing_columns:
        raise ValueError(f"Input CSV is missing model features: {missing_columns}")
    if extra_columns:
        raise ValueError(f"Input CSV has unsupported columns: {extra_columns}")

    features = input_data.loc[:, feature_columns].copy()
    numeric_columns = [
        column.name
        for column in input_schema.inputs
        if column.type.name.lower() in {"double", "float"}
    ]
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="raise").astype(
            "float64"
        )

    raw_predictions = pd.Series(model.predict(features), index=input_data.index)
    label_map = {
        0: "No",
        1: "Yes",
        "0": "No",
        "1": "Yes",
        "No": "No",
        "Yes": "Yes",
    }
    predictions = raw_predictions.map(label_map)
    if predictions.isna().any():
        raise ValueError("Model returned predictions outside the expected churn labels.")

    results = pd.DataFrame(index=input_data.index)
    if "customerID" in input_data.columns:
        results["customerID"] = input_data["customerID"]
    results[PREDICTION_COLUMN] = predictions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8")
    yes_count = int((predictions == "Yes").sum())
    row_count = len(features)
    missing_count = int(features.isna().sum().sum())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="batch_inference"):
        mlflow.log_params({"model_uri": model_uri})
        mlflow.set_tags({"inference_type": "batch"})
        mlflow.log_metrics(
            {
                "batch_input_rows": float(row_count),
                "predictions_yes": float(yes_count),
                "predictions_no": float(row_count - yes_count),
                "prediction_yes_rate": yes_count / row_count,
                "input_missing_values": float(missing_count),
                "input_missing_rate": float(missing_count / features.size),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate batch churn predictions from a CSV using MLflow."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV containing model features and optionally customerID.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the CSV containing churn predictions.",
    )
    parser.add_argument(
        "--model-uri",
        default=MODEL_URI,
        help=f"MLflow model URI (default: {MODEL_URI}).",
    )
    args = parser.parse_args()

    results = predict_batch(args.input, args.output, args.model_uri)
    print(f"Predictions written: {len(results)} rows -> {args.output}")


if __name__ == "__main__":
    main()

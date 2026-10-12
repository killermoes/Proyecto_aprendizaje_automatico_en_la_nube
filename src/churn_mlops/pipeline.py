import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from configs import (
    CV_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE,
    REGISTERED_MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
    TEST_SIZE,
    setup_mlflow,
)
from churn_mlops.data import load_and_split_data
from churn_mlops.models import (
    build_model,
    evaluate_model,
    optimize_xgboost_with_optuna,
)


@task(name="load-and-split-data")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return load_and_split_data()


@task(name="train-evaluate-and-track")
def train_evaluate_and_track(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    logger = get_run_logger()
    model = build_model(model_name, X_train)
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    cv_f1_yes = float(
        cross_val_score(
            model,
            X_train,
            y_train,
            scoring=f1_scorer,
            cv=cross_validation,
            n_jobs=1,
            error_score="raise",
        ).mean()
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    with mlflow.start_run(run_name=model_name) as active_run:
        mlflow.log_params(
            {
                "test_size": TEST_SIZE,
                "random_state_split": RANDOM_STATE,
                "imputacion_numerica": "median",
                "imputacion_categorica": "most_frequent",
                "codificacion": "OneHotEncoder",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_f1_yes_mean", cv_f1_yes)
        mlflow.set_tags(
            {
                "etapa": "model_comparison",
                "dataset": "telco-churn",
                "modelo": model_name,
                "cv_strategy": "StratifiedKFold",
            }
        )
        if model_name == "logistic_regression_baseline":
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("random_state_modelo", RANDOM_STATE)
        elif model_name == "random_forest_baseline":
            mlflow.log_param("n_estimators", 300)
            mlflow.log_param("random_state_modelo", RANDOM_STATE)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo",
            input_example=X_test.head(3),
            skops_trusted_types=(
                ["numpy.dtype", "sklearn.tree._tree.Tree"]
                if model_name == "random_forest_baseline"
                else ["numpy.dtype"]
            ),
        )

        run_id = active_run.info.run_id

    logger.info("%s: %s", model_name, metrics)
    return {
        "model": model_name,
        "run_id": run_id,
        "cv_f1_yes": cv_f1_yes,
        "metrics": metrics,
    }


@task(name="register-champion-model")
def register_champion_model(run_id: str, cv_f1_yes: float) -> None:
    logger = get_run_logger()
    model_uri = f"runs:/{run_id}/modelo"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
        await_registration_for=60,
    )
    client = MlflowClient()
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="validation_cv_f1_yes",
        value=f"{cv_f1_yes:.6f}",
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=REGISTERED_MODEL_ALIAS,
        version=model_version.version,
    )
    logger.info(
        "Registered %s version %s as alias '%s' (CV F1=%.4f)",
        REGISTERED_MODEL_NAME,
        model_version.version,
        REGISTERED_MODEL_ALIAS,
        cv_f1_yes,
    )


@flow(
    name="telco-churn-training-pipeline",
    description="Train, evaluate, and track Telco Churn baseline models.",
    log_prints=True,
)
def telco_churn_training_flow() -> list[dict[str, object]]:
    logger = get_run_logger()
    setup_mlflow()

    X_train, X_test, y_train, y_test = load_data()
    logger.info(
        "Dataset partitioned: train=%s, test=%s",
        len(X_train),
        len(X_test),
    )

    results = [
        train_evaluate_and_track(
            model_name,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        for model_name in (
            "dummy_baseline",
            "logistic_regression_baseline",
            "random_forest_baseline",
        )
    ]
    results.append(
        optimize_xgboost_with_optuna(X_train, X_test, y_train, y_test)
    )

    best_model = max(results, key=lambda result: result["cv_f1_yes"])
    logger.info(
        "Best model by cross-validation F1 for churn: %s (F1=%.4f)",
        best_model["model"],
        best_model["cv_f1_yes"],
    )
    best_run_id = best_model["run_id"]
    best_cv_f1_yes = best_model["cv_f1_yes"]
    if not isinstance(best_run_id, str) or not isinstance(
        best_cv_f1_yes, (int, float)
    ):
        raise TypeError("Best model result is missing its run ID or CV F1 score.")
    register_champion_model(best_run_id, float(best_cv_f1_yes))

    logger.info("MLflow experiment: %s", MLFLOW_EXPERIMENT_NAME)
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    telco_churn_training_flow()

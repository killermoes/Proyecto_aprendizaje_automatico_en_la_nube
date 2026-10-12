import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from prefect import get_run_logger, task
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from optuna.samplers import TPESampler

from configs import CV_FOLDS, OPTUNA_TRIALS, RANDOM_STATE
from churn_mlops.models.training import build_model, evaluate_model


@task(
    name="optimize-xgboost-with-optuna",
    description="Tune XGBoost using stratified cross-validation and F1 for churn.",
)
def optimize_xgboost_with_optuna(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    logger = get_run_logger()
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)

    with mlflow.start_run(run_name="xgboost_optuna_f1_yes") as parent_run:
        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=100),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1.0, 10.0
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                "scale_pos_weight": trial.suggest_float(
                    "scale_pos_weight", 1.0, 4.0
                ),
            }
            model = build_model("xgboost_optuna", X_train, trial_params)

            with mlflow.start_run(
                nested=True,
                run_name=f"xgboost_trial_{trial.number:02d}",
            ):
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    scoring=f1_scorer,
                    cv=cross_validation,
                    n_jobs=1,
                    error_score="raise",
                )
                mean_f1 = float(scores.mean())
                mlflow.log_params(trial_params)
                mlflow.log_metric("cv_f1_yes_mean", mean_f1)
                mlflow.log_metric("cv_f1_yes_std", float(scores.std()))
                mlflow.set_tags(
                    {
                        "etapa": "optuna",
                        "dataset": "telco-churn",
                        "metrica_objetivo": "f1_yes",
                        "trial_number": str(trial.number),
                    }
                )

            return mean_f1

        mlflow.log_params(
            {
                "modelo": "XGBClassifier",
                "optuna_trials": OPTUNA_TRIALS,
                "cv_folds": CV_FOLDS,
                "cv_strategy": "StratifiedKFold",
                "objective_metric": "f1_yes",
                "random_state": RANDOM_STATE,
            }
        )
        mlflow.set_tags(
            {
                "etapa": "hyperparameter_tuning",
                "dataset": "telco-churn",
                "modelo": "XGBoost",
                "metrica_objetivo": "f1_yes",
            }
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="telco-churn-xgboost",
            sampler=TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

        best_model = build_model("xgboost_optuna", X_train, study.best_params)
        best_model.fit(X_train, y_train)
        test_metrics = evaluate_model(best_model, X_test, y_test)
        mlflow.log_params(
            {f"best_{key}": value for key, value in study.best_params.items()}
        )
        mlflow.log_metric("best_cv_f1_yes", float(study.best_value))
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metric("test_f1_yes", test_metrics["f1"])
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="modelo",
            input_example=X_test.head(3),
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        mlflow.log_table(
            pd.DataFrame(
                [
                    {
                        "trial": trial.number,
                        "cv_f1_yes": trial.value,
                        **trial.params,
                    }
                    for trial in study.trials
                ]
            ),
            artifact_file="optuna_trials.json",
        )
        run_id = parent_run.info.run_id

    logger.info(
        "Best XGBoost CV F1 for churn: %.4f (trial %s); test metrics: %s",
        study.best_value,
        study.best_trial.number,
        test_metrics,
    )
    return {
        "model": "xgboost_optuna",
        "run_id": run_id,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "cv_f1_yes": float(study.best_value),
        "metrics": test_metrics,
    }

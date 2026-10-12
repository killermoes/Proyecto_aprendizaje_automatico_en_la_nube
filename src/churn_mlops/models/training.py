from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from configs import RANDOM_STATE


def build_model(
    model_name: str,
    features: pd.DataFrame,
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    if model_name == "dummy_baseline":
        estimator: Any = DummyClassifier(strategy="most_frequent")
        model_step = "dummy"
    elif model_name == "logistic_regression_baseline":
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model_step = "logistic_regression"
    elif model_name == "random_forest_baseline":
        estimator = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model_step = "random_forest"
    elif model_name == "xgboost_optuna":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            **(model_params or {}),
        )
        model_step = "xgboost"
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (model_step, estimator),
        ]
    )


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    positive_class_index = list(model.classes_).index(1)

    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(
            precision_score(target, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(target, predictions, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(target, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(
            roc_auc_score(target, probabilities[:, positive_class_index])
        ),
    }

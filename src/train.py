from __future__ import annotations

import argparse
import json
import logging

# save model
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from azure_mlflow_utils import configure_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "churn_mlops"

CATEGORICAL_COLUMNS = [
    "contract_type",
    "internet_service",
    "payment_method",
]

NUMERICAL_COLUMNS = [
    "age",
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_last_3m",
    "late_payments_last_6m",
    "streaming_subscription",
    "device_protection",
    "partner",
    "dependents",
]

DEFAULT_PARAMS = {
    "random_state": 42,
    "n_estimators": 200,
}


def load_and_prepare_data(path: str):
    df = pd.read_csv(path)

    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def build_pipeline(params=None):
    final_params = DEFAULT_PARAMS.copy()
    if params:
        final_params.update(params)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("num", "passthrough", NUMERICAL_COLUMNS),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**final_params)),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--params_json", default=None)
    parser.add_argument("--model_output", required=True)

    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)

    args = parser.parse_args()

    configure_mlflow(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    mlflow.set_registry_uri(None)

    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_prepare_data(args.input_path)

    params = None
    if args.params_json:
        with open(args.params_json) as f:
            params = json.load(f)

    with mlflow.start_run() as run:
        model = build_pipeline(params)
        model.fit(X_train, y_train)

        auc = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params or DEFAULT_PARAMS)
        mlflow.log_metric("roc_auc", auc)

        signature = infer_signature(
            X_train,
            model.predict_proba(X_train)[:, 1],
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=None,
        )

        logger.info(f"AUC: {auc:.4f}")
        logger.info(f"Run ID: {run.info.run_id}")

        os.makedirs(args.model_output, exist_ok=True)
        joblib.dump(model, os.path.join(args.model_output, "model.joblib"))
        logger.info(f"Model saved to {args.model_output}")

        print("Model saved successfully")

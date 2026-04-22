from __future__ import annotations

import argparse

import mlflow
import optuna

from azure_mlflow_utils import configure_mlflow
from train import build_pipeline, evaluate_model, load_and_prepare_data

N_TRIALS_DEFAULT = 20
EXPERIMENT_NAME = "churn_mlops"


def objective(trial: optuna.Trial, input_path: str) -> float:
    """Fonction objectif Optuna sur pipeline complet."""

    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True,
        ),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    X_train, X_test, y_train, y_test = load_and_prepare_data(input_path)

    with mlflow.start_run(
        run_name=f"trial_{trial.number}",
        nested=True,
    ) as run:
        pipeline = build_pipeline(params)
        pipeline.fit(X_train, y_train)

        auc = evaluate_model(pipeline, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", auc)

        # Store the run_id as a trial user attribute
        trial.set_user_attr("mlflow_run_id", run.info.run_id)

    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS_DEFAULT)

    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)

    args = parser.parse_args()

    configure_mlflow(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="optuna_study"):

        study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda trial: objective(trial, args.input_path),
            n_trials=args.n_trials,
        )

        mlflow.log_metric("best_auc", study.best_value)

        best_run_id = study.best_trial.user_attrs["mlflow_run_id"]

        print("Best Trial:")
        print(study.best_trial.number)

        print("Best AUC:")
        print(round(study.best_value, 4))

        print("Best Params:")
        print(study.best_trial.params)

        print("Best MLflow Run ID:")
        print(best_run_id)
        # ← pass this to register_model.py

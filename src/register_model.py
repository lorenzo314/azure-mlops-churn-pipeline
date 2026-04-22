from __future__ import annotations

import argparse
import logging

import mlflow

from azure_mlflow_utils import configure_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model_name", default="churn_model")

    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)

    args = parser.parse_args()

    configure_mlflow(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    model_uri = f"runs:/{args.run_id}/model"

    model = mlflow.register_model(model_uri, args.model_name)

    logger.info(f"Registered model: {model.name} v{model.version}")


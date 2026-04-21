from __future__ import annotations

import mlflow
import logging

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)


def configure_mlflow(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
) -> MLClient:
    """Configure MLflow to use Azure ML workspace."""

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    workspace = ml_client.workspaces.get(workspace_name)

    mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)

    logger.info("MLflow tracking URI set to Azure ML workspace")

    return ml_client


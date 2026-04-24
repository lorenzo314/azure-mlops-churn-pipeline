from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.identity import DefaultAzureCredential

default_instance = "Standard_DS1_v2"


def deploy_model(
    subscription_id,
    resource_group,
    workspace_name,
    endpoint_name,
    model_name,
    model_version,
    instance_type=default_instance,
):
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential,
        subscription_id,
        resource_group,
        workspace_name,
    )

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key",
    )

    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file="./conda.yaml",
    )

    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=f"azureml:{model_name}:{model_version}",
        environment=env,
        code_configuration=CodeConfiguration(
            code="./src",
            scoring_script="score.py",
        ),
        instance_type=instance_type,
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()

    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print("Deployment successful")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy model to Azure ML endpoint")

    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)

    parser.add_argument("--endpoint_name", default="churn-endpoint")
    parser.add_argument("--model_name", default="churn-model")
    parser.add_argument("--model_version", type=int, default=1)
    parser.add_argument("--compute", default="cpu-cluster")

    parser.add_argument("--instance_type", default=default_instance)

    args = parser.parse_args()

    deploy_model(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        endpoint_name=args.endpoint_name,
        model_name=args.model_name,
        model_version=args.model_version,
        instance_type=args.instance_type,
    )

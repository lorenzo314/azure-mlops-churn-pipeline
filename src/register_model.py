import argparse

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


def main(args):
    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    model = Model(
        path=args.model_path,
        name=args.model_name,
        type="custom_model",
    )

    ml_client.models.create_or_update(model)

    print(f"Model {args.model_name} registered successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)

    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)

    args = parser.parse_args()

    main(args)

import argparse

from azure.ai.ml import MLClient, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def get_ml_client(subscription_id, resource_group, workspace_name):
    return MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name,
    )


def create_environment():
    return Environment(
        name="churn-pipeline-env",
        description="Environment for churn pipeline",
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )


def build_pipeline(env, args):

    # -------------------------
    # STEP 1 — Data validation
    # -------------------------
    data_validation = command(
        name="data_validation",
        display_name="Data Validation",
        code="src",
        command="""
        python data_validation.py \
            --input_path ${{inputs.input_data}} \
            --output_path ${{outputs.validated_data}}
        """,
        inputs={
            "input_data": {
                "type": "uri_file",
            }
        },
        outputs={
            "validated_data": {
                "type": "uri_file",
            }
        },
        environment=env,
        compute="cpu-cluster",
    )

    # -------------------------
    # STEP 2 — Training
    # -------------------------
    training = command(
        name="train_model",
        display_name="Train Model",
        code="src",
        command="""
        python train.py \
            --input_path ${{inputs.input_data}} \
            --model_output ${{outputs.model}} \
            --subscription_id ${{inputs.subscription_id}} \
            --resource_group ${{inputs.resource_group}} \
            --workspace_name ${{inputs.workspace_name}}
        """,
        inputs={
            "input_data": {"type": "uri_file"},
            "subscription_id": {"type": "string"},
            "resource_group": {"type": "string"},
            "workspace_name": {"type": "string"},
        },
        outputs={
            "model": {"type": "uri_folder"},
        },
        environment=env,
        compute="cpu-cluster",
    )

    # -------------------------
    # STEP 3 — Register model
    # -------------------------
    register = command(
        name="register_model",
        display_name="Register Model",
        code="src",
        command="""
        python register_model.py \
            --model_path ${{inputs.model_path}} \
            --model_name ${{inputs.model_name}} \
            --subscription_id ${{inputs.subscription_id}} \
            --resource_group ${{inputs.resource_group}} \
            --workspace_name ${{inputs.workspace_name}}
        """,
        inputs={
            "model_path": {"type": "uri_folder"},
            "model_name": {"type": "string"},
            "subscription_id": {"type": "string"},
            "resource_group": {"type": "string"},
            "workspace_name": {"type": "string"},
        },
        environment=env,
        compute="cpu-cluster",
    )

    # -------------------------
    # PIPELINE DEFINITION
    # -------------------------
    @pipeline()
    def churn_pipeline(subscription_id, resource_group, workspace_name):

        validation_step = data_validation(input_data="data/raw/churn_train.csv")

        training_step = training(
            input_data=validation_step.outputs.validated_data,
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
        )

        register(
            model_path=training_step.outputs.model,
            model_name="churn-model",
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
        )

        return {}

    return churn_pipeline(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
    )


def main(args):
    ml_client = get_ml_client(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    env = create_environment()

    pipeline_job = build_pipeline(env, args)

    pipeline_job.settings.default_compute = "cpu-cluster"

    ml_client.jobs.create_or_update(pipeline_job)

    print("Pipeline submitted successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)

    args = parser.parse_args()

    main(args)

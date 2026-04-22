import json
import logging
import os

import mlflow.pyfunc
import pandas as pd

logger = logging.getLogger(__name__)

model = None


def init():
    global model

    try:
        model_dir = os.getenv("AZUREML_MODEL_DIR")

        print(f"MODEL DIR: {model_dir}")

        model_path = os.path.join(model_dir, "model")

        print(f"LOADING MODEL FROM: {model_path}")

        model = mlflow.pyfunc.load_model(model_path)

        print("MODEL LOADED SUCCESSFULLY")

    except Exception as e:
        print(f"INIT ERROR: {str(e)}")
        raise


def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data)

        preds = model.predict(df)

        return {"predictions": preds.tolist()}

    except Exception as e:
        logger.error(str(e))
        return {"error": str(e)}


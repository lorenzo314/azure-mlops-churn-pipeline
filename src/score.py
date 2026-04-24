import json
import logging
import os

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

model = None


def init():
    global model

    try:
        model_dir = os.getenv("AZUREML_MODEL_DIR")

        print(f"MODEL DIR: {model_dir}")

        # Find model dynamically (robust)
        model_path = None
        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".joblib"):
                    model_path = os.path.join(root, file)

        if model_path is None:
            raise ValueError("No model.joblib found")

        print(f"LOADING MODEL FROM: {model_path}")

        model = joblib.load(model_path)

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

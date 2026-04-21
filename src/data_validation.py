from __future__ import annotations

import argparse
import logging
import pandas as pd
import great_expectations as gx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data(input_path: str) -> None:
    """Validate dataset before training."""

    df = pd.read_csv(input_path)
    validator = gx.from_pandas(df)

    validator.expect_column_to_exist("churn")

    validator.expect_column_values_to_be_between(
        "monthly_charges",
        min_value=0,
    )

    validator.expect_column_values_to_be_between(
        "tenure_months",
        min_value=0,
    )

    results = validator.validate()

    if not results["success"]:
        failed = [
            r["expectation_config"]["expectation_type"]
            for r in results["results"]
            if not r["success"]
        ]

        raise ValueError(f"Validation failed: {failed}")

    logger.info("Data validation successful")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)

    args = parser.parse_args()

    validate_data(args.input_path)


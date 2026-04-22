from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ID_COLUMNS = ["customer_id"]


def run_drift(reference_path, current_path, output_path):
    ref = pd.read_csv(reference_path).drop(columns=ID_COLUMNS)
    cur = pd.read_csv(current_path).drop(columns=ID_COLUMNS)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)

    logger.info(f"Drift report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference_path", required=True)
    parser.add_argument("--current_path", required=True)
    parser.add_argument("--output_path", default="reports/drift.html")

    args = parser.parse_args()

    run_drift(
        args.reference_path,
        args.current_path,
        args.output_path,
    )

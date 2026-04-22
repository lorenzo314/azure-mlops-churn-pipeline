from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RANDOM_STATE = 42


def generate_churn_data(n_samples: int, drift: bool = False) -> pd.DataFrame:
    """Generate synthetic churn dataset.

    Args:
        n_samples: Number of rows to generate
        drift: If True, introduces distribution shift (for monitoring demo)

    Returns:
        pandas DataFrame
    """

    rng = np.random.default_rng(RANDOM_STATE)

    # Numerical features
    age = rng.integers(18, 80, size=n_samples)
    tenure_months = rng.integers(0, 72, size=n_samples)

    monthly_charges = rng.normal(70, 30, size=n_samples)
    monthly_charges = np.clip(monthly_charges, 10, 200)

    if drift:
        # Introduce drift
        monthly_charges *= 1.3
        tenure_months = tenure_months * 0.7

    total_charges = monthly_charges * tenure_months

    # Binary features
    senior_citizen = (age > 65).astype(int)
    partner = rng.integers(0, 2, size=n_samples)
    dependents = rng.integers(0, 2, size=n_samples)
    streaming_subscription = rng.integers(0, 2, size=n_samples)
    device_protection = rng.integers(0, 2, size=n_samples)

    # Behavioral features
    support_tickets_last_3m = rng.poisson(1.5, size=n_samples)
    late_payments_last_6m = rng.poisson(1.0, size=n_samples)

    # Categorical features
    contract_type = rng.choice(
        ["Monthly", "One year", "Two year"],
        size=n_samples,
        p=[0.6, 0.25, 0.15],
    )

    internet_service = rng.choice(
        ["DSL", "Fiber", "None"],
        size=n_samples,
        p=[0.4, 0.4, 0.2],
    )

    payment_method = rng.choice(
        ["Credit Card", "Bank Transfer", "Electronic Check"],
        size=n_samples,
    )

    # Target (simple rule-based churn logic)
    churn_prob = (
        0.3
        + 0.002 * monthly_charges
        + 0.3 * (contract_type == "Monthly")
        + 0.2 * (late_payments_last_6m > 2)
        - 0.003 * tenure_months
    )

    churn_prob = np.clip(churn_prob, 0, 1)

    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(n_samples),
            "age": age,
            "senior_citizen": senior_citizen,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_tickets_last_3m": support_tickets_last_3m,
            "late_payments_last_6m": late_payments_last_6m,
            "streaming_subscription": streaming_subscription,
            "device_protection": device_protection,
            "partner": partner,
            "dependents": dependents,
            "churn": churn,
        }
    )

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--drift", action="store_true")

    args = parser.parse_args()

    df = generate_churn_data(
        n_samples=args.n_samples,
        drift=args.drift,
    )

    df.to_csv(args.output_path, index=False)

    logger.info(f"Generated dataset with {len(df)} rows")
    logger.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()


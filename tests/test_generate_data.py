import pandas as pd

from src.data.generate_data import generate_churn_data


def test_generate_data_shape():
    df = generate_churn_data(n_samples=100)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100


def test_generate_data_columns():
    df = generate_churn_data(n_samples=10)

    expected_columns = {
        "age",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "internet_service",
        "payment_method",
        "churn",
    }

    assert expected_columns.issubset(set(df.columns))


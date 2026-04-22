import pandas as pd


def test_no_missing_values():
    df = pd.DataFrame(
        {
            "age": [30, 40],
            "tenure_months": [12, 24],
            "monthly_charges": [50.0, 70.0],
        }
    )

    assert not df.isnull().values.any()


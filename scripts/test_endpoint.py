import json
import os

import requests

ENDPOINT_URL = os.getenv("ENDPOINT_URL")
API_KEY = os.getenv("AZUREML_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


# Sample payload (must match training schema)
data = [
    {
        "age": 42,
        "senior_citizen": 0,
        "tenure_months": 12,
        "contract_type": "Monthly",
        "internet_service": "Fiber",
        "payment_method": "Credit Card",
        "monthly_charges": 89.5,
        "total_charges": 1074.0,
        "support_tickets_last_3m": 2,
        "late_payments_last_6m": 1,
        "streaming_subscription": 1,
        "device_protection": 0,
        "partner": 1,
        "dependents": 0,
    }
]


response = requests.post(
    ENDPOINT_URL,
    headers=headers,
    data=json.dumps(data),
)

print("Status code:", response.status_code)
print("Response:", response.text)


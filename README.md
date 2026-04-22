# 🚀 Azure MLOps Churn Prediction Pipeline

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)]()
[![Azure ML](https://img.shields.io/badge/Azure%20ML-v2-orange)]()
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

End-to-end **MLOps pipeline** for churn prediction using **Azure Machine Learning**, including training, experiment tracking, model registry, deployment, and monitoring.

---

## 📌 Overview

This project demonstrates a **production-grade machine learning workflow**:

* Data validation
* Model training & hyperparameter tuning
* MLflow experiment tracking
* Model registration
* Deployment to Azure ML managed endpoints
* Real-time inference via REST API
* Data drift monitoring

---

## 🧠 Architecture

High-level overview of the end-to-end MLOps pipeline.

```text
Data → Validation → Training → MLflow → Registry → Deployment → Endpoint → Monitoring
```

---

## ⚙️ Tech Stack

* Python 3.10
* Azure Machine Learning (SDK v2)
* MLflow
* LightGBM
* Pandas / NumPy
* Evidently (monitoring)

---

## 📂 Project Structure

```
src/
├── data/
│   └── generate_data.py
├── azure_mlflow_utils.py   # Azure ML + MLflow integration utilities
├── data_validation.py # data quality checks using Great Expectation
├── train.py
├── hpo.py
├── register_model.py
├── deploy.py
├── score.py
├── monitor_drift.py

scripts/
└── test_endpoint.py

data/
├── raw/
├── production/

tests/
├── test_generate_data.py
├── test_data_validation.py
pyproject.toml

conda.yaml
pyproject.toml
requirements.txt
```

### Azure ML Integration

This project uses a helper module:

```bash
src/azure_mlflow_utils.py
```  

to:

- connect MLflow to Azure ML workspace
- configure tracking URI
- simplify authentication

---

## 📁 Data

This project uses synthetic data generated locally.

Data is **not stored in the repository**. Instead, generate it using:

```bash
python src/data/generate_data.py --output_path data/raw/churn_train.csv
```

The expected structure is:

```
data/
├── raw/         # training data
├── production/  # simulated production data (for monitoring)
```

---

## ✅ Data Validation

This project includes a validation step using **Great Expectations** to ensure data quality before training.

Validation checks include:

* schema consistency
* missing values
* basic feature constraints

### ▶️ Run validation

```bash
python src/data_validation.py \
  --input_path data/raw/churn_train.csv
```

### ✔ Example output

```text
Validation successful: dataset passed all checks
```

If validation fails, the pipeline stops before training.

---

## 🏋️ Training

```bash
python src/train.py
```

---

## 🔍 Hyperparameter Tuning

```bash
python src/hpo.py
```

---

## 📦 Register Model

```bash
python src/register_model.py
```

---

## 🚀 Deploy Model

```bash
python src/deploy.py \
  --subscription_id $SUBSCRIPTION_ID \
  --resource_group $RESOURCE_GROUP \
  --workspace_name $WORKSPACE_NAME \
  --endpoint_name churn-endpoint \
  --model_name churn_model \
  --model_version 4
```

---

## 🔌 Test Endpoint

```bash
python scripts/test_endpoint.py
```

Example response:

```json
{"predictions": [1]}
```

---

## 📊 Monitoring

```bash
python src/monitor_drift.py \
  --reference_path data/raw/churn_train_clean.csv \
  --current_path data/production/churn_prod_drifted.csv
```

---

## 🧹 Code Quality
  
This project uses Ruff for linting:  

```bash
ruff check .
```

---

## 🧪 Testing
  
Run unit tests with:
  
```bash
pytest
```

---

## ⚠️ Key Learnings

* Environment consistency between training and inference is critical
* MLflow model structure requires correct loading (`/model` path)
* Azure ML environments can have non-obvious dependency resolution behavior
* Proper dependency pinning avoids runtime failures

---

## 📈 Future Improvements

* CI/CD pipeline (GitHub Actions)
* Automated retraining
* Alerting on drift detection
* Batch inference pipeline

---

## 📄 License

MIT License

---

## 👤 Author

Lorenzo


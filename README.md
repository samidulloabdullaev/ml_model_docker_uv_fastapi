This repository demonstrates a simple MLOps workflow for customer churn prediction using **FastAPI**, **scikit-learn**, and modern Python packaging.

## Features

  - **Model Training**: `train.py` downloads the Telco Customer Churn dataset, preprocesses it, trains a logistic regression model, and saves it as `model.bin`.
  - **Prediction API**: `predict.py` loads the trained model and serves predictions via a **FastAPI** web service at the `/predict` endpoint. Input validation is enforced using **Pydantic** models.
  - **Marketing Automation Example**: `marketing.py` shows how to call the prediction API and take action based on the result.
  - **Containerization**: A `Dockerfile` for building a containerized API service.
  - **CI/CD**: **GitHub Actions** workflows for code formatting (`black`) and linting (`ruff`).

-----

## Quickstart

### 1\. Train the Model

```sh
uv pip install -r requirements.txt
uv run python train.py
```

### 2\. Run the API

You can run the API directly or using Docker.

#### Direct Execution

```sh
uvicorn predict:app --host 0.0.0.0 --port 8080
```

#### Docker

```sh
docker build -t predict-churn .
docker run -it --rm -p 8080:8080 predict-churn
```

### 3\. Make a Prediction

```sh
uv pip install requests
uv run python marketing.py
```

-----

## Development

  - **Format code**:

    ```sh
    black .
    ```

  - **Lint code**:

    ```sh
    ruff check .
    ```

  - **CI/CD**: See the `.github/workflows/` directory for automated checks.

-----

## Project Structure

```
.
├── .github/
│   └── workflows/        # GitHub Actions CI/CD workflows
├── Dockerfile            # Containerization configuration
├── marketing.py          # Example API client
├── predict.py            # FastAPI prediction service
├── pyproject.toml        # Project dependencies (can be used with uv)
├── requirements.txt      # Project dependencies (alternative)
└── train.py              # Model training script
```
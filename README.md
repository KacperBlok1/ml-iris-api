# Iris ML API

![Python](https://img.shields.io/badge/python-3.x-blue)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-green)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Pytest](https://img.shields.io/badge/tested%20with-pytest-yellow)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)
![Status](https://img.shields.io/badge/build-passing-brightgreen)

FastAPI service that exposes a **machine learning model for Iris flower classification** using **scikit-learn**.

The goal of this project is to demonstrate how I:

* train a machine learning model
* serve it through a REST API
* automatically test the API
* integrate the project with **continuous integration**

---

# 🚀 Tech Stack

| Tool           | Purpose                           |
| -------------- | --------------------------------- |
| Python 3.x     | Programming language              |
| FastAPI        | Web framework for serving the API |
| scikit-learn   | Machine learning library          |
| joblib         | Model serialization               |
| pytest         | Automated testing framework       |
| GitHub Actions | Continuous Integration            |

---

# 📁 Project Structure

```text id="a5q7pn"
ml-iris-api/

├── app/
│   ├── __init__.py
│   └── main.py          # FastAPI app loading the trained model and exposing /predict
│
├── models/
│   └── iris_rf.joblib   # trained RandomForest model
│
├── tests/
│   ├── __init__.py
│   └── test_api.py      # API tests using TestClient
│
├── train_model.py       # script to train and save the model
│
├── requirements.txt
├── pytest.ini
│
└── .github/
    └── workflows/
        └── tests.yml    # CI pipeline
```

### Directory Overview

**app/**
Contains the FastAPI application and prediction endpoint.

**models/**
Stores the trained machine learning model.

**tests/**
Automated API tests using pytest and FastAPI TestClient.

**train_model.py**
Script used to train the machine learning model and save it.

**GitHub Actions workflow**
Continuous Integration pipeline configuration.

---

# 🤖 Training the Model

To train the machine learning model locally:

```id="ow9tce"
pip install -r requirements.txt
python train_model.py
```

This script:

1. Downloads the **Iris dataset**
2. Trains a **RandomForest classifier**
3. Saves the trained model to:

```id="6jzh31"
models/iris_rf.joblib
```

---

# ⚡ Running the API Locally

Start the FastAPI server:

```id="we6jdx"
uvicorn app.main:app --reload
```

The API will run at:

```id="2owhgj"
http://127.0.0.1:8000
```

---

# 📚 API Endpoints

### Healthcheck

```id="t6k93c"
GET /health
```

Returns a simple response confirming that the service is running.

---

### Prediction Endpoint

```id="s0s76r"
POST /predict
```

Example request:

```json id="t4p8td"
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The API returns:

* predicted Iris class
* class probabilities

Possible classes:

* setosa
* versicolor
* virginica

---

# 📊 Interactive API Documentation

FastAPI automatically generates interactive documentation.

### Swagger UI

```id="fg5j9q"
http://127.0.0.1:8000/docs
```

### OpenAPI schema

```id="jrszh6"
http://127.0.0.1:8000/openapi.json
```

These allow testing the API directly from the browser.

---

# 🧪 Tests

The test suite verifies key functionality of the API.

Covered scenarios:

* `/health` endpoint
* valid prediction request
* invalid input validation
* API response correctness

Run all tests:

```id="9k1m9n"
pytest
```

Tests use **FastAPI TestClient**, which allows testing the API without starting a separate server.

---

# 🔁 Continuous Integration

The repository includes a **GitHub Actions CI pipeline**.

Workflow location:

```id="hyk3nu"
.github/workflows/tests.yml
```

### Pipeline steps

Triggered on:

* push
* pull_request to the `main` branch

Pipeline actions:

1. Install project dependencies
2. Train the ML model using `train_model.py`
3. Run the full pytest test suite

This ensures that:

* the model trains correctly
* the API works
* the tests pass on every commit

You can view pipeline runs in the **Actions tab** of this repository.

---

# 🛠 Future Improvements

Potential improvements for the project:

* add Docker containerization
* add model versioning
* integrate ML experiment tracking (MLflow)
* add input schema validation
* deploy the API to a cloud environment

---

# 👨‍💻 Author

**Kacper Blok**

Machine Learning / Backend / QA Automation Portfolio Project

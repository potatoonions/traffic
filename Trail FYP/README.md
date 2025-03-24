# AI Stock Market Prediction System
=====================================

An AI-powered stock market prediction system using LSTM neural networks to predict stock prices.

## Table of Contents
-----------------

* [Features](#features)
* [Setup](#setup)
* [API Endpoints](#api-endpoints)
  * [Health Check](#health-check)
  * [Stock Prediction](#stock-prediction)
  * [Model Training](#model-training)
  * [Get Historical Data](#get-historical-data)
* [Usage Example](#usage-example)
* [Requirements](#requirements)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [License](#license)

## Features
------------

### Real-time Stock Price Predictions

*   Utilizes LSTM neural networks to predict stock prices
*   Provides accurate predictions based on historical data

### Historical Data Visualization

*   Displays historical stock data for better understanding
*   Helps in identifying trends and patterns

### Model Training

*   Trains the model for specific stocks
*   Improves the accuracy of predictions

### RESTful API Endpoints

*   Provides a simple and intuitive API for interacting with the system
*   Supports multiple endpoints for different functionalities

## Setup
--------

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Create a `.env` File

Create a new file named `.env` with your Alpha Vantage API key:

```makefile
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

### Step 3: Start the Server

```bash
uvicorn main:app --reload
```

## API Endpoints
----------------

### Health Check

*   **Endpoint:** `GET /`
*   **Description:** Returns a 200 OK response if the server is running.

### Stock Prediction

*   **Endpoint:** `POST /predict-stock`
*   **Request Body:**

```json
{
    "symbol": "AAPL",
    "days_to_predict": 1
}
```

*   **Description:** Returns a JSON response with the predicted stock price.

### Model Training

*   **Endpoint:** `POST /train-model`
*   **Query Parameters:**

```sql
symbol: str  # Stock symbol to train on
```

*   **Description:** Returns a JSON response with the training results.

### Get Historical Data

*   **Endpoint:** `GET /stock-data/{symbol}`
*   **Description:** Returns a JSON response with the historical stock data.

## Usage Example
----------------

### Train the Model for a Specific Stock

```python
import requests

response = requests.post("http://localhost:8000/train-model", params={"symbol": "AAPL"})
print(response.json())
```

### Get Stock Prediction

```python
import requests
from datetime import datetime

response = requests.post(
    "http://localhost:8000/predict-stock",
    json={
        "symbol": "AAPL",
        "days_to_predict": 1
    }
)
print(response.json())
```

## Requirements
------------

*   Python 3.8+
*   Alpha Vantage API key
*   Required packages listed in requirements.txt

## Troubleshooting
-----------------

*   Make sure to install all dependencies before running the server.
*   Check the API key in the `.env` file for any typos or incorrect formatting.
*   If the server is not responding, try restarting it.

## Contributing
------------

Contributions are welcome! Please submit a pull request with your changes.

## License
-------

This project is licensed under the MIT License. See LICENSE.txt for details.

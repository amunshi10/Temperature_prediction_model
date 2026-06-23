# Temperature Prediction in London Using ML Models

Predict the daily mean temperature in London using historical weather data and scikit-learn regression pipelines tracked with MLflow.

## Project Structure

```
.
├── data/               # Raw CSV data (not committed — see data/README.md)
├── notebooks/
│   └── temperature_prediction.ipynb   # Full EDA → training → evaluation notebook
├── src/
│   ├── data_preprocessing.py          # Loading, cleaning, feature engineering
│   ├── models.py                      # sklearn Pipeline definitions
│   ├── train.py                       # CLI training script with MLflow logging
│   └── predict.py                     # Load saved model and predict on new data
├── outputs/            # Saved plots
├── models/             # Saved model artifacts (.pkl)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Data

Download [London Weather Data](https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data) from Kaggle and place `london_weather.csv` in the `data/` folder.

## Run Training

```bash
python src/train.py --data data/london_weather.csv --output models
```

This will:
- Engineer time and lag features
- Train 5 regression models (Ridge, Lasso, Random Forest, Gradient Boosting, SVR)
- Log all metrics and artifacts to MLflow
- Save the best model to `models/best_model.pkl`

## View MLflow UI

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

## Predict on New Data

```bash
python src/predict.py --model models/best_model.pkl --input data/new_data.csv
```

## Models Compared

| Model | Notes |
|-------|-------|
| Ridge Regression | Strong baseline with L2 regularisation |
| Lasso Regression | Sparse solution via L1 regularisation |
| Random Forest | Ensemble of decision trees, captures non-linearity |
| Gradient Boosting | Sequential boosting, typically best accuracy |
| SVR (RBF kernel) | Support vector approach for regression |

## Feature Engineering

- **Cyclical encoding** of month and day-of-year (sin/cos) to preserve periodicity
- **Lag features**: previous day's temperature, 3-day and 7-day rolling mean
- **Calendar features**: year, week of year

## Results

Run `mlflow ui` after training to compare all experiments interactively.  
Key plots are saved to `outputs/` after running the notebook.

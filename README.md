# London Temperature Prediction

A machine learning project that predicts daily mean temperature in London using historical weather data. Multiple regression models are trained and compared using scikit-learn pipelines, with experiment tracking via MLflow.

## Overview

- **Dataset:** Historical daily weather records for London (1979–2020)
- **Target:** Daily mean temperature (°C)
- **Models:** Ridge, Lasso, Random Forest, Gradient Boosting, SVR
- **Best RMSE:** ~1.2°C (Gradient Boosting)

## Repository Structure

```
├── data/
│   └── london_weather.csv
├── notebooks/
│   └── temperature_prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── predict.py
├── outputs/
├── models/
└── requirements.txt
```

## Features

**Time-series feature engineering:**
- Cyclical encoding of month and day-of-year (sin/cos transforms)
- Lag features: previous day's temperature, 3-day and 7-day rolling mean
- Calendar features: year, week of year

**Model evaluation:**
- Time-series aware cross-validation (`TimeSeriesSplit`)
- Metrics: RMSE, MAE, R²
- All runs logged to MLflow for reproducibility

## Results

| Model | Test RMSE | Test MAE | R² |
|-------|-----------|----------|----|
| Gradient Boosting | ~1.20 | ~0.91 | ~0.98 |
| Random Forest | ~1.35 | ~1.02 | ~0.97 |
| Ridge Regression | ~1.89 | ~1.48 | ~0.95 |
| SVR | ~1.95 | ~1.51 | ~0.94 |
| Lasso | ~2.01 | ~1.57 | ~0.94 |

## Tech Stack

`Python` `scikit-learn` `pandas` `numpy` `MLflow` `matplotlib` `seaborn`

"""
Train all models, log experiments with MLflow, and save the best model.

Usage:
    python src/train.py --data data/london_weather.csv
"""

import argparse
import os
import joblib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_data, engineer_features, clean_data, get_X_y
from models import build_pipelines


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def train_and_log(data_path: str, output_dir: str = "models") -> str:
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path)
    df = engineer_features(df)
    df = clean_data(df)
    X, y = get_X_y(df)

    # Time-series aware split: last 20 % as hold-out test set
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    tscv = TimeSeriesSplit(n_splits=5)
    pipelines = build_pipelines()

    best_name, best_rmse, best_pipeline = None, float("inf"), None

    mlflow.set_experiment("london_temperature_prediction")

    for name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=name):
            cv_rmse = -cross_val_score(
                pipeline, X_train, y_train,
                cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            mlflow.log_param("model", name)
            mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
            mlflow.log_metric("cv_rmse_std", cv_rmse.std())

            pipeline.fit(X_train, y_train)
            metrics = evaluate(y_test, pipeline.predict(X_test))

            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"{name:20s}  CV RMSE={cv_rmse.mean():.3f}  Test RMSE={metrics['rmse']:.3f}  R²={metrics['r2']:.3f}")

            if metrics["rmse"] < best_rmse:
                best_rmse, best_name, best_pipeline = metrics["rmse"], name, pipeline

    best_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_pipeline, best_path)
    print(f"\nBest model: {best_name} (RMSE={best_rmse:.3f}) saved to {best_path}")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/london_weather.csv")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    train_and_log(args.data, args.output)

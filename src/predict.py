"""
Load the saved best model and predict for new input.

Usage:
    python src/predict.py --model models/best_model.pkl --input data/new_data.csv
"""

import argparse
import joblib
import pandas as pd
from data_preprocessing import engineer_features, clean_data, get_X_y


def predict(model_path: str, input_path: str) -> pd.Series:
    pipeline = joblib.load(model_path)
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = engineer_features(df)
    df = clean_data(df)
    X, _ = get_X_y(df)
    preds = pipeline.predict(X)
    return pd.Series(preds, name="predicted_mean_temp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    results = predict(args.model, args.input)
    print(results.to_string())

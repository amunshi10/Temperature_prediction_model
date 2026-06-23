import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Cyclical encoding for month and day-of-year so the model understands periodicity
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Rolling statistics (3-day and 7-day lag averages)
    df = df.sort_values("date").reset_index(drop=True)
    df["mean_temp_lag1"] = df["mean_temp"].shift(1)
    df["mean_temp_roll3"] = df["mean_temp"].shift(1).rolling(3).mean()
    df["mean_temp_roll7"] = df["mean_temp"].shift(1).rolling(7).mean()

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(-999, np.nan)
    df = df.dropna(subset=["mean_temp"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


FEATURE_COLS = [
    "cloud_cover",
    "sunshine",
    "global_radiation",
    "precipitation",
    "pressure",
    "snow_depth",
    "year",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "mean_temp_lag1",
    "mean_temp_roll3",
    "mean_temp_roll7",
]

TARGET_COL = "mean_temp"


def get_X_y(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df.dropna(subset=available + [TARGET_COL])
    return df_clean[available], df_clean[TARGET_COL]

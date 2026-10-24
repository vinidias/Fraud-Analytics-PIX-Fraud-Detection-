"""
Semana 3-4 — Feature engineering para detecção de fraude.

Uso:
    python scripts/feature_engineering.py --input data/transactions.csv \
        --output data/features.parquet

Features produzidas (por transação):
- valor em relação ao histórico do cliente (média, desvio, z-score)
- velocidade: nº tx nas últimas 1h/24h
- ticket médio do cliente vs. tx atual
- horário, dia da semana
- flags de anomalia simples
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["event_time"])


def customer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agregações por cliente (histórico até a transação atual — point-in-time)."""
    df = df.sort_values(["customer_id", "event_time"]).copy()
    g = df.groupby("customer_id")["amount"]
    df["cust_mean_amt"] = g.transform(lambda s: s.shift().expanding().mean())
    df["cust_std_amt"] = g.transform(lambda s: s.shift().expanding().std())
    df["cust_max_amt"] = g.transform(lambda s: s.shift().expanding().max())
    df["cust_tx_count"] = df.groupby("customer_id").cumcount()
    df["amt_vs_cust_mean"] = df["amount"] / df["cust_mean_amt"].replace(0, np.nan)
    df["amt_z_score"] = (df["amount"] - df["cust_mean_amt"]) / df["cust_std_amt"].replace(0, np.nan)
    return df


def velocity_features(df: pd.DataFrame, windows: tuple[str, ...] = ("1h", "24h")) -> pd.DataFrame:
    """Contagem de transações em janelas móveis por cliente."""
    df = df.sort_values(["customer_id", "event_time"]).copy()
    df = df.set_index("event_time")
    for w in windows:
        col = f"tx_{w.replace('h', 'h')}_count"
        df[col] = (
            df.groupby("customer_id")["amount"]
              .rolling(w)
              .count()
              .reset_index(level=0, drop=True)
              .astype("Int64")
        )
    return df.reset_index()


def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["event_time"].dt.hour
    df["dow"] = df["event_time"].dt.dayofweek
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def anomaly_flags(df: pd.DataFrame, z_thr: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    df["flag_amount_outlier"] = (df["amt_z_score"].abs() > z_thr).astype(int)
    df["flag_high_velocity_1h"] = (df.get("tx_1h_count", 0) >= 3).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = temporal_features(df)
    df = customer_history_features(df)
    df = velocity_features(df)
    df = anomaly_flags(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = load(src)
    print(f"Loaded {len(df):,} rows from {src}")
    feats = build_features(df)
    print(f"Features: {feats.shape[1]} cols")
    feats.to_parquet(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

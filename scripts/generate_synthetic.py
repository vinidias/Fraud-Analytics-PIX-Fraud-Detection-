"""
Gera dataset sintético de transações com padrões de fraude para testes offline.

Uso:
    python scripts/generate_synthetic.py --n 50000 --output data/synthetic/transactions.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 50000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    start = datetime(2025, 1, 1)

    def _ts(k):
        offsets = rng.integers(0, 90 * 24 * 3600, size=k)
        return [start + timedelta(seconds=int(s)) for s in offsets]

    legit = pd.DataFrame({
        "tx_id": rng.integers(1, 10**9, size=n_legit),
        "customer_id": rng.integers(1, 5000, size=n_legit),
        "merchant_id": rng.integers(1, 300, size=n_legit),
        "device_id": [f"dev_{i}" for i in rng.integers(1, 4000, size=n_legit)],
        "amount": np.round(rng.lognormal(mean=4.0, sigma=0.8, size=n_legit), 2),
        "event_time": _ts(n_legit),
        "lat": np.round(rng.uniform(-30, -3, size=n_legit), 4),
        "lon": np.round(rng.uniform(-60, -35, size=n_legit), 4),
        "is_fraud": 0,
        "is_chargeback": 0,
    })

    fraud = pd.DataFrame({
        "tx_id": rng.integers(1, 10**9, size=n_fraud),
        "customer_id": rng.integers(1, 5000, size=n_fraud),
        "merchant_id": rng.integers(1, 300, size=n_fraud),
        "device_id": [f"dev_{i}" for i in rng.integers(1, 50, size=n_fraud)],
        "amount": np.round(rng.lognormal(mean=5.5, sigma=1.4, size=n_fraud), 2),
        "event_time": [
            (start + timedelta(days=int(d), hours=int(rng.choice([0, 1, 2, 3, 4, 23]))))
            for d in rng.integers(0, 90, size=n_fraud)
        ],
        "lat": np.round(rng.uniform(-30, -3, size=n_fraud), 4),
        "lon": np.round(rng.uniform(-60, -35, size=n_fraud), 4),
        "is_fraud": 1,
        "is_chargeback": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["Class"] = df["is_fraud"]
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--fraud-rate", type=float, default=0.02)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.fraud_rate)
    df.to_csv(out, index=False)
    print(f"[OK] {len(df):,} transacoes salvas em {out}")
    print(f"     fraude: {df['Class'].sum():,} ({100*df['Class'].mean():.2f}%)")
    print(f"     periodo: {df['event_time'].min()} -> {df['event_time'].max()}")


if __name__ == "__main__":
    main()

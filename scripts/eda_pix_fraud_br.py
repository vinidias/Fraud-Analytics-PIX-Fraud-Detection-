"""
Análise exploratória — PIX Fraud BR.
Gera métricas comparativas entre transações fraudulentas e legítimas.

Saída:
    docs/s07_pix_fraud_br/eda_metrics.json
    docs/s07_pix_fraud_br/top_insights.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT_DIR = Path("docs/s07_pix_fraud_br")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rate_by(df: pd.DataFrame, col: str, min_n: int = 100) -> pd.DataFrame:
    g = df.groupby(col, dropna=False)["Class"].agg(["count", "sum", "mean"]).reset_index()
    g = g.rename(columns={"count": "n_tx", "sum": "n_fraud", "mean": "fraud_rate"})
    g = g[g["n_tx"] >= min_n].sort_values("fraud_rate", ascending=False)
    return g


def main() -> None:
    df = pd.read_parquet(SRC)
    n = len(df)
    n_fraud = int(df["Class"].sum())
    rate = n_fraud / n
    print(f"[OK] {n:,} tx · fraude {n_fraud:,} ({100*rate:.4f}%)")

    metrics: dict = {
        "dataset": "andremessina/pix-fraud-br (Hugging Face)",
        "n_transactions": n,
        "n_fraud": n_fraud,
        "fraud_rate_pct": round(100 * rate, 4),
        "period": [str(df["event_time"].min()), str(df["event_time"].max())],
    }

    # Taxa por tipo de transação
    by_type = rate_by(df, "tx_type")
    metrics["by_tx_type"] = by_type.round(4).to_dict(orient="records")
    print("\n=== Taxa de fraude por tipo de transação ===")
    print(by_type.round(4).to_string(index=False))

    # Noturno vs diurno
    by_night = rate_by(df, "is_night")
    metrics["by_night"] = by_night.round(4).to_dict(orient="records")
    print("\n=== Noturno vs diurno ===")
    print(by_night.round(4).to_string(index=False))

    # Dia útil vs fim de semana
    by_work = rate_by(df, "is_workday")
    metrics["by_workday"] = by_work.round(4).to_dict(orient="records")
    print("\n=== Dia útil vs fim de semana ===")
    print(by_work.round(4).to_string(index=False))

    # Acima do limite noturno
    by_limit = rate_by(df, "above_night_limit")
    metrics["by_above_night_limit"] = by_limit.round(4).to_dict(orient="records")
    print("\n=== Acima do limite noturno ===")
    print(by_limit.round(4).to_string(index=False))

    # Estatísticas de valor
    stats = df.groupby("Class")["amount"].agg(["count", "mean", "median", "std", "min", "max"]).round(2)
    metrics["amount_stats"] = stats.to_dict(orient="index")
    print("\n=== Estatísticas de valor (R$) ===")
    print(stats.to_string())

    # Razão saldo residual
    stats2 = df.groupby("Class")["residual_balance_ratio"].agg(["mean", "median", "std"]).round(4)
    metrics["residual_balance_ratio_stats"] = stats2.to_dict(orient="index")
    print("\n=== Razão saldo residual ===")
    print(stats2.to_string())

    # Proporção valor recebedor
    stats3 = df.groupby("Class")["receiver_amount_ratio"].agg(["mean", "median", "std"]).round(4)
    metrics["receiver_amount_ratio_stats"] = stats3.to_dict(orient="index")
    print("\n=== Proporção valor recebedor ===")
    print(stats3.to_string())

    # Hora do dia — taxa
    by_hour = rate_by(df, "hora_dia", min_n=1000)
    metrics["by_hour"] = by_hour.round(4).to_dict(orient="records")
    print("\n=== Top 5 horas com maior taxa de fraude ===")
    print(by_hour.head(5).round(4).to_string(index=False))

    # Top pagadores (potenciais laranjas / contas comprometidas)
    top_pag = (
        df[df["Class"] == 1]
        .groupby("customer_id")
        .agg(n_fraude=("Class", "size"), valor_total=("amount", "sum"))
        .sort_values("valor_total", ascending=False)
        .head(20)
        .round(2)
        .reset_index()
    )
    metrics["top_fraudulent_payers"] = top_pag.to_dict(orient="records")
    print("\n=== Top 10 pagadores com maior valor fraudulento ===")
    print(top_pag.head(10).to_string(index=False))

    # Top recebedores
    top_rec = (
        df[df["Class"] == 1]
        .groupby("merchant_id")
        .agg(n_fraude=("Class", "size"), valor_total=("amount", "sum"))
        .sort_values("valor_total", ascending=False)
        .head(20)
        .round(2)
        .reset_index()
    )
    metrics["top_fraudulent_receivers"] = top_rec.to_dict(orient="records")
    print("\n=== Top 10 recebedores com maior valor fraudulento ===")
    print(top_rec.head(10).to_string(index=False))

    (OUT_DIR / "eda_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[OK] salvo em {OUT_DIR / 'eda_metrics.json'}")


if __name__ == "__main__":
    main()

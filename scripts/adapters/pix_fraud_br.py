"""
Adaptador — PIX Fraud BR (Hugging Face: andremessina/pix-fraud-br).

Mapeia o schema original para o schema padrão do projeto:
    valor_brl          -> amount
    datetime_brasilia  -> event_time
    id_pagador         -> customer_id
    id_recebedor       -> merchant_id (recebedor)
    fraude             -> Class

Uso:
    python scripts/adapters/pix_fraud_br.py
    python scripts/adapters/pix_fraud_br.py --sample 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SRC = Path("data/pix_fraud_br/data/train-00000-of-00001.parquet")
OUT = Path("data/pix_fraud_br/transactions.parquet")

MAP = {
    "id_pagador": "customer_id",
    "id_recebedor": "merchant_id",
    "tipo_transacao": "tx_type",
    "valor_brl": "amount",
    "datetime_brasilia": "event_time",
    "dia_semana": "weekday",
    "dia_util": "is_workday",
    "horario_noturno": "is_night",
    "acima_limite_noturno": "above_night_limit",
    "razao_saldo_residual": "residual_balance_ratio",
    "proporcao_valor_recebedor": "receiver_amount_ratio",
    "fraude": "Class",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None, help="amostra aleatória")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_parquet(SRC)
    print(f"[OK] carregado {len(df):,} linhas, {df.shape[1]} colunas")

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"[OK] amostra: {len(df):,} linhas")

    df = df.rename(columns=MAP)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values(["customer_id", "event_time"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[OK] {OUT} — {len(df):,} linhas, {df.shape[1]} colunas")
    print(f"     fraude: {int(df['Class'].sum()):,} ({100*df['Class'].mean():.4f}%)")
    print(f"     período: {df['event_time'].min()} -> {df['event_time'].max()}")


if __name__ == "__main__":
    main()

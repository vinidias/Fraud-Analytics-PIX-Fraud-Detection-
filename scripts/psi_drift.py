"""
PSI (Population Stability Index) — monitoramento de drift de features.

Mede: quanto a distribuição de cada feature mudou entre treino (referencia)
e janelas de teste. Threshold classico:
    < 0.1   → sem drift
    0.1-0.25 → drift moderado (monitorar)
    > 0.25  → drift severo (investigar retreino)

Saida:
    docs/s09_walk_forward/psi_features.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT_DIR = Path("docs/s09_walk_forward")

FEATURES = [
    "amount", "log_amount", "residual_balance_ratio", "receiver_amount_ratio",
    "hora_dia", "is_night", "drain_score",
]

JANELAS = {
    "W1": (15, 18),
    "W2": (19, 22),
    "W3": (23, 26),
    "W4": (27, 31),
}
TRAIN_MAX_DAY = 14


def psi(reference: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI com bins por quantil da referência."""
    ref = reference[~np.isnan(reference)]
    act = actual[~np.isnan(actual)]
    if len(ref) == 0 or len(act) == 0:
        return float("nan")

    # Bins por quantil da referência
    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(ref, quantiles))
    if len(breaks) < 3:
        return 0.0

    ref_hist, _ = np.histogram(ref, bins=breaks)
    act_hist, _ = np.histogram(act, bins=breaks)

    # Normalizar
    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    act_pct = act_hist / max(act_hist.sum(), 1)

    # Evitar log(0)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, 1 - eps)
    act_pct = np.clip(act_pct, eps, 1 - eps)

    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)
    for c in ["is_night", "is_workday", "above_night_limit"]:
        df[c] = df[c].astype(int)
    df["night_x_residual"] = df["is_night"] * (1 - df["residual_balance_ratio"])
    df["night_x_ratio"] = df["is_night"] * df["receiver_amount_ratio"]
    df["limit_x_residual"] = df["above_night_limit"] * (1 - df["residual_balance_ratio"])
    df["drain_score"] = (1 - df["residual_balance_ratio"]) * df["receiver_amount_ratio"] * (df["is_night"] + 1)
    df["log_amount"] = np.log1p(df["amount"])
    return df


def classify(p: float) -> str:
    if np.isnan(p):
        return "n/d"
    if p < 0.1:
        return "sem_drift"
    if p < 0.25:
        return "moderado"
    return "severo"


def main() -> None:
    df = pd.read_parquet(SRC).sample(500_000, random_state=42)
    df = build_features(df)
    df["dia"] = df["event_time"].dt.day

    train = df[df["dia"] <= TRAIN_MAX_DAY]
    print(f"[OK] referencia: {len(train):,} tx")

    rows = []
    for jan_nome, (d0, d1) in JANELAS.items():
        test = df[(df["dia"] >= d0) & (df["dia"] <= d1)]
        for feat in FEATURES:
            p = psi(train[feat].values, test[feat].values)
            rows.append({
                "janela": jan_nome,
                "dias": f"{d0}-{d1}",
                "feature": feat,
                "psi": round(p, 4),
                "classificacao": classify(p),
            })

    df_psi = pd.DataFrame(rows)
    df_psi.to_csv(OUT_DIR / "psi_features.csv", index=False)

    # Pivot para visualização
    pivot = df_psi.pivot_table(index="feature", columns="janela", values="psi")
    print("\n=== PSI por feature x janela ===")
    print(pivot.round(4).to_string())

    # Features com drift maximo por janela
    print("\n=== Features com drift >= moderado ===")
    flagged = df_psi[df_psi["classificacao"].isin(["moderado", "severo"])]
    if len(flagged) > 0:
        print(flagged.to_string(index=False))
    else:
        print("Nenhuma — todas as features estaveis (PSI < 0.1)")

    # Sumario por feature
    print("\n=== PSI medio por feature (4 janelas) ===")
    media = df_psi.groupby("feature")["psi"].mean().sort_values(ascending=False).round(4)
    print(media.to_string())


if __name__ == "__main__":
    main()

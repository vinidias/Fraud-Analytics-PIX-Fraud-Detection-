"""
Semana 5-6 — Detecção de anomalias com pyod.

Compara IsolationForest, LOF, OCSVM e AutoEncoder em dataset de fraude.
Métricas apropriadas para classes desbalanceadas:
- PR-AUC (precision-recall)
- Recall@k (k = % de transações revisáveis manualmente)
- Custo de FP vs FN

Uso:
    python scripts/anomaly_detection.py --input data/features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM


MODELS = {
    "IForest": lambda: IForest(contamination=0.01, random_state=42, n_estimators=200),
    "LOF":     lambda: LOF(contamination=0.01, n_neighbors=20),
    "OCSVM":   lambda: OCSVM(contamination=0.01, kernel="rbf", gamma="auto"),
    "AutoEnc": lambda: AutoEncoder(contamination=0.01, epoch_num=20, hidden_neuron_list=[32, 16, 32], random_state=42),
}


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k_pct: float) -> float:
    """Recall quando apenas top k% transações são revisadas (ordenadas por score)."""
    n = len(scores)
    top_k = int(n * k_pct)
    order = np.argsort(scores)[::-1][:top_k]
    return float(y_true[order].sum() / max(y_true.sum(), 1))


def eval_model(name: str, y_true: np.ndarray, scores: np.ndarray) -> dict:
    ap = average_precision_score(y_true, scores)
    auc = roc_auc_score(y_true, scores) if y_true.sum() > 0 else np.nan
    prec, rec, _ = precision_recall_curve(y_true, scores)
    return {
        "model": name,
        "pr_auc": round(ap, 4),
        "roc_auc": round(float(auc), 4),
        "recall@1%": round(recall_at_k(y_true, scores, 0.01), 4),
        "recall@5%": round(recall_at_k(y_true, scores, 0.05), 4),
        "recall@10%": round(recall_at_k(y_true, scores, 0.10), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", default="Class")
    parser.add_argument("--drop", nargs="*", default=["event_time", "customer_id"])
    args = parser.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    y = df[args.target].astype(int).to_numpy()
    drop_cols = [c for c in args.drop + [args.target] if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).fillna(0).to_numpy()

    print(f"Dataset: X={X.shape}, fraude={y.sum()} ({100*y.mean():.3f}%)")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    results = []
    for name, ctor in MODELS.items():
        print(f"\n>>> Treinando {name}...")
        model = ctor()
        model.fit(Xs)
        scores = model.decision_function(Xs)
        results.append(eval_model(name, y, scores))

    result_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n=== Ranking ===")
    print(result_df.to_string(index=False))

    out = Path("docs/s04/benchmark_anomaly_models.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out, index=False)
    print(f"\nSalvo em {out}")


if __name__ == "__main__":
    main()

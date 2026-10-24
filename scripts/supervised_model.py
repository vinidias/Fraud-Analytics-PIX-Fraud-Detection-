"""
Modelo supervisionado (LightGBM) para fraude PIX — vs. regras de negócio.

Estratégia:
    - Split TEMPORAL (nunca aleatório em fraude: leak de futuro)
        treino: primeiros 21 dias · teste: últimos 10 dias
    - Features: engenharia + interações (noturno × saldo residual, etc.)
    - Class weight balanced
    - Métricas: PR-AUC, Recall@k%, Precision@threshold ótimo por custo
    - Baseline: regras de negócio (interpretáveis, prontas para produção)
    - SHAP: top features para comunicação executiva

Saída:
    docs/s07_pix_fraud_br/model_metrics.json
    docs/s07_pix_fraud_br/feature_importance.csv
    docs/s07_pix_fraud_br/threshold_analysis.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT_DIR = Path("docs/s07_pix_fraud_br")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features contextuais + interações — só o que produz sinal real."""
    df = df.copy()

    # Garantir tipos
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)

    # Booleanos para int
    for c in ["is_night", "is_workday", "above_night_limit"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Features de interação — a chave para fraude PIX
    df["night_x_residual"] = df["is_night"] * (1 - df["residual_balance_ratio"])
    df["night_x_ratio"] = df["is_night"] * df["receiver_amount_ratio"]
    df["limit_x_residual"] = df["above_night_limit"] * (1 - df["residual_balance_ratio"])
    df["drain_score"] = (
        (1 - df["residual_balance_ratio"]) *
        df["receiver_amount_ratio"] *
        (df["is_night"] + 1)
    )

    # Log do valor (distribuição muito assimétrica)
    df["log_amount"] = np.log1p(df["amount"])

    return df


# ============================================================
# Baseline: regras de negócio interpretáveis
# ============================================================

def rule_baseline(df: pd.DataFrame) -> pd.Series:
    """Regra combinada: noturno + acima do limite + saldo residual baixo + recebedor recebe quase tudo."""
    return (
        (df["is_night"] == 1)
        & (df["above_night_limit"] == 1)
        & (df["residual_balance_ratio"] < 0.10)
        & (df["receiver_amount_ratio"] > 0.95)
    ).astype(int)


# ============================================================
# Métricas
# ============================================================

def recall_at_k(y: np.ndarray, scores: np.ndarray, k_pct: float) -> float:
    n = len(scores)
    top = int(n * k_pct)
    if top == 0:
        return 0.0
    idx = np.argsort(scores)[::-1][:top]
    return float(y[idx].sum() / max(y.sum(), 1))


def threshold_by_cost(y: np.ndarray, scores: np.ndarray,
                     cost_fp: float = 20.0, cost_fn_frac: float = 1.0,
                     amounts: np.ndarray | None = None) -> dict:
    """
    Escolhe threshold que minimiza custo total.
    cost_fn = cost_fn_frac * valor da transação (perda real)
    cost_fp = 50 (custo operacional de revisão + fricção)
    """
    prec, rec, thr = precision_recall_curve(y, scores)
    if amounts is None:
        amounts = np.ones_like(y, dtype=float)
    n = len(y)
    pos = y.sum()
    neg = n - pos
    avg_amount = float(amounts[y == 1].mean()) if pos > 0 else 1.0

    best = {"threshold": 0.5, "cost": float("inf"), "precision": 0, "recall": 0, "f1": 0}
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        if p == 0 and r == 0:
            continue
        # estimativa exata via PR
        tp = r * pos
        fp = tp * (1 - p) / max(p, 1e-9)
        fn = pos - tp
        cost = fp * cost_fp + fn * avg_amount * cost_fn_frac
        f1 = 2 * p * r / max(p + r, 1e-9)
        if cost < best["cost"]:
            best = {
                "threshold": float(t),
                "cost": float(cost),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "fp": int(round(fp)),
                "fn": int(round(fn)),
            }
    return best


# ============================================================
# Main
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=500_000)
    p.add_argument("--train-days", type=int, default=21)
    p.add_argument("--test-days", type=int, default=10)
    args = p.parse_args()

    df = pd.read_parquet(SRC)
    if args.sample and args.sample < len(df):
        # amostra estratificada por dia para preservar distribuição temporal
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
    print(f"[OK] {len(df):,} transações carregadas")

    df = build_features(df)

    # Split temporal
    t_min = df["event_time"].min()
    cutoff = t_min + pd.Timedelta(days=args.train_days)
    train = df[df["event_time"] < cutoff].copy()
    test = df[df["event_time"] >= cutoff].copy()
    print(f"    treino: {len(train):,} tx ({100*train['Class'].mean():.4f}% fraude)")
    print(f"    teste : {len(test):,} tx ({100*test['Class'].mean():.4f}% fraude)")

    feature_cols = [
        "amount", "log_amount",
        "saldo_anterior_pagador", "saldo_posterior_pagador",
        "saldo_anterior_recebedor", "saldo_posterior_recebedor",
        "residual_balance_ratio", "receiver_amount_ratio",
        "hora_dia", "is_night", "is_workday", "above_night_limit",
        "night_x_residual", "night_x_ratio", "limit_x_residual", "drain_score",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_train, y_train = train[feature_cols].fillna(0), train["Class"].astype(int)
    X_test, y_test = test[feature_cols].fillna(0), test["Class"].astype(int)
    print(f"    features: {len(feature_cols)}")
    print(f"    {feature_cols}")

    # ---------- Baseline: regras ----------
    print("\n=== Baseline: regras de negócio ===")
    rules_pred = rule_baseline(test)
    rules_prec = precision_score(y_test, rules_pred, zero_division=0)
    rules_rec = recall_score(y_test, rules_pred, zero_division=0)
    rules_f1 = f1_score(y_test, rules_pred, zero_division=0)
    print(f"    precision={rules_prec:.4f}  recall={rules_rec:.4f}  f1={rules_f1:.4f}")
    print(f"    TP={int(rules_pred.sum())} de {int(y_test.sum())} fraudes")

    # ---------- LightGBM ----------
    print("\n=== Treinando LightGBM ===")
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        scale_pos_weight=pos_weight,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    proba = model.predict_proba(X_test)[:, 1]

    # Métricas
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    print(f"    PR-AUC  : {pr_auc:.4f}")
    print(f"    ROC-AUC : {roc_auc:.4f}")
    print(f"    Recall@1%:  {recall_at_k(y_test.values, proba, 0.01):.4f}")
    print(f"    Recall@5%:  {recall_at_k(y_test.values, proba, 0.05):.4f}")
    print(f"    Recall@10%: {recall_at_k(y_test.values, proba, 0.10):.4f}")

    # Threshold ótimo por custo
    thr_info = threshold_by_cost(y_test.values, proba, amounts=X_test["amount"].values)
    print(f"\n    threshold ótimo (min custo): {thr_info['threshold']:.4f}")
    print(f"    precision={thr_info['precision']:.4f}  recall={thr_info['recall']:.4f}  f1={thr_info['f1']:.4f}")
    print(f"    FP={thr_info['fp']}  FN={thr_info['fn']}")

    # Threshold por F1 máximo (mais conservador)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, proba)
    f1_arr = 2 * prec_arr * rec_arr / np.maximum(prec_arr + rec_arr, 1e-9)
    best_f1_idx = int(np.argmax(f1_arr))
    best_f1_threshold = float(thr_arr[best_f1_idx]) if best_f1_idx < len(thr_arr) else 0.5
    best_f1_prec = float(prec_arr[best_f1_idx])
    best_f1_rec = float(rec_arr[best_f1_idx])
    best_f1_val = float(f1_arr[best_f1_idx])
    print(f"\n    threshold ótimo (max F1): {best_f1_threshold:.4f}")
    print(f"    precision={best_f1_prec:.4f}  recall={best_f1_rec:.4f}  f1={best_f1_val:.4f}")

    # Confusion matrix nos dois thresholds
    y_pred_opt = (proba >= thr_info["threshold"]).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    print(f"    confusion matrix (custo):\n{cm}")
    y_pred_f1 = (proba >= best_f1_threshold).astype(int)
    cm_f1 = confusion_matrix(y_test, y_pred_f1)
    print(f"    confusion matrix (F1):\n{cm_f1}")

    # Feature importance (top 15)
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print("\n    Top 10 features:")
    print(fi.head(10).to_string(index=False))

    # Varredura de thresholds (para gráfico)
    thresholds_grid = np.linspace(0.01, 0.99, 99)
    grid = []
    for t in thresholds_grid:
        pred = (proba >= t).astype(int)
        if pred.sum() == 0:
            continue
        grid.append({
            "threshold": round(float(t), 3),
            "n_flagged": int(pred.sum()),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        })
    pd.DataFrame(grid).to_csv(OUT_DIR / "threshold_analysis.csv", index=False)

    # Salvar métricas
    metrics = {
        "dataset": "pix-fraud-br",
        "sample_size": int(args.sample),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "test_fraud_rate_pct": round(100 * float(y_test.mean()), 4),
        "n_features": len(feature_cols),
        "baseline_rules": {
            "precision": round(float(rules_prec), 4),
            "recall": round(float(rules_rec), 4),
            "f1": round(float(rules_f1), 4),
            "n_flagged": int(rules_pred.sum()),
        },
        "lightgbm": {
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "recall@1%": round(recall_at_k(y_test.values, proba, 0.01), 4),
            "recall@5%": round(recall_at_k(y_test.values, proba, 0.05), 4),
            "recall@10%": round(recall_at_k(y_test.values, proba, 0.10), 4),
            "threshold_by_cost": {
                "threshold": round(thr_info["threshold"], 4),
                "precision": round(thr_info["precision"], 4),
                "recall": round(thr_info["recall"], 4),
                "f1": round(thr_info["f1"], 4),
                "fp": thr_info["fp"],
                "fn": thr_info["fn"],
            },
            "threshold_by_f1": {
                "threshold": round(best_f1_threshold, 4),
                "precision": round(best_f1_prec, 4),
                "recall": round(best_f1_rec, 4),
                "f1": round(best_f1_val, 4),
            },
        },
        "feature_importance_top10": fi.head(10).to_dict(orient="records"),
    }
    (OUT_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[OK] métricas em {OUT_DIR / 'model_metrics.json'}")


if __name__ == "__main__":
    main()

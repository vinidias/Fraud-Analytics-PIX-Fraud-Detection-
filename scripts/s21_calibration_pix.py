"""
s21 - Calibracao + cost-sensitive threshold no PIX Fraud BR.

Usa nosso melhor modelo (CatBoost PR-AUC 0.9614) em vez do BAF fraco.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import train_test_split
import catboost as cb

warnings.filterwarnings("ignore")

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT = Path("docs/s21_calibration_pix")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "amount", "log_amount",
    "saldo_anterior_pagador", "saldo_posterior_pagador",
    "saldo_anterior_recebedor", "saldo_posterior_recebedor",
    "residual_balance_ratio", "receiver_amount_ratio",
    "hora_dia", "is_night", "is_workday", "above_night_limit",
    "night_x_residual", "night_x_ratio", "limit_x_residual", "drain_score",
]


def build_features(df):
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
    fator = df["amount"].median() / 300.0
    for col in ["amount", "saldo_anterior_pagador", "saldo_posterior_pagador",
                "saldo_anterior_recebedor", "saldo_posterior_recebedor"]:
        df[col] = df[col] / fator
    return df


def main():
    print("[1] carregando PIX Fraud BR...")
    df = pd.read_parquet(SRC).sample(500_000, random_state=42)
    df = build_features(df)

    cutoff = df["event_time"].min() + pd.Timedelta(days=21)
    train = df[df["event_time"] < cutoff]
    test = df[df["event_time"] >= cutoff].reset_index(drop=True)
    print(f"    train: {len(train):,} | test: {len(test):,}")

    X_tr = train[FEATURES].fillna(0)
    y_tr = train["Class"].astype(int)
    X_te = test[FEATURES].fillna(0)
    y_te = test["Class"].astype(int)

    # Split interno para calibracao
    X_tr2, X_cal, y_tr2, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
    )

    print("[2] treinando modelo base...")
    pos_weight = (y_tr2 == 0).sum() / max((y_tr2 == 1).sum(), 1)
    base = cb.CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        scale_pos_weight=pos_weight, random_seed=42,
        verbose=False, thread_count=-1, bootstrap_type="Bernoulli",
    )
    base.fit(X_tr2, y_tr2)

    print("[3] calibrando (isotonic)...")
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    except ImportError:
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    def eval_model(name, proba):
        pr_auc = average_precision_score(y_te, proba)
        brier = brier_score_loss(y_te, proba)
        prec, rec, thr = precision_recall_curve(y_te, proba)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        idx = int(np.argmax(f1))
        return {
            "modelo": name,
            "pr_auc": round(float(pr_auc), 4),
            "brier": round(float(brier), 5),
            "f1": round(float(f1[idx]), 4),
            "threshold_f1": round(float(thr[idx]) if idx < len(thr) else 0.5, 4),
        }

    results = []
    results.append(eval_model("CatBoost (base)", base.predict_proba(X_te)[:, 1]))
    results.append(eval_model("CatBoost (isotonic)", calibrated.predict_proba(X_te)[:, 1]))
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT / "calibration_results.csv", index=False)
    print(df_res.to_string(index=False))

    print("[4] cost-sensitive threshold...")
    proba = calibrated.predict_proba(X_te)[:, 1]
    amounts = test["amount"].values
    prec, rec, thr = precision_recall_curve(y_te, proba)
    cost_fp = 20.0
    recovery = 0.85
    n_pos = int(y_te.sum())
    avg_fraud_amount = float(amounts[y_te == 1].mean()) if n_pos > 0 else 1000.0
    cost_fn = avg_fraud_amount * recovery

    costs = []
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        tp = r * n_pos
        fp = tp * (1 - p) / max(p, 1e-9)
        fn = n_pos - tp
        cost = fp * cost_fp + fn * cost_fn
        costs.append(cost)
    best_idx = int(np.argmin(costs))
    best_thr = float(thr[best_idx])
    best_prec = float(prec[best_idx])
    best_rec = float(rec[best_idx])

    lines = []
    lines.append("# s21 - Calibracao + threshold otimo (PIX Fraud BR)")
    lines.append("")
    lines.append("**Data:** 2026-10-24 | **Dataset:** PIX Fraud BR (500k) | **Modelo:** CatBoost")
    lines.append("")
    lines.append("## 1. Calibracao (Brier score)")
    lines.append("")
    lines.append("| Modelo | PR-AUC | Brier | F1 |")
    lines.append("|---|---:|---:|---:|")
    for _, r in df_res.iterrows():
        lines.append(f"| {r['modelo']} | {r['pr_auc']:.4f} | {r['brier']:.5f} | {r['f1']:.4f} |")
    lines.append("")
    delta_brier = (df_res.iloc[1]["brier"] - df_res.iloc[0]["brier"]) / df_res.iloc[0]["brier"] * 100
    lines.append(f"**Reducao do Brier com isotonic**: {delta_brier:+.1f}%")
    lines.append("")
    lines.append("> **Por que importa**: Brier mede calibracao das probabilidades. Um modelo com PR-AUC alto mas Brier alto tem thresholds **errados** — o score 0.9 pode significar 0.6 na realidade.")
    lines.append("")
    lines.append("## 2. Cost-sensitive threshold")
    lines.append("")
    lines.append(f"- **Custo FP** (revisao manual): R$ {cost_fp:.0f}")
    lines.append(f"- **Custo FN** (fraude nao detectada, recuperavel {recovery*100:.0f}%): R$ {cost_fn:.0f}")
    lines.append(f"- **Threshold otimo**: **{best_thr:.4f}**")
    lines.append(f"- **Precision**: {best_prec:.4f}")
    lines.append(f"- **Recall**: {best_rec:.4f}")
    lines.append("")
    lines.append("## 3. Licao de negocio")
    lines.append("")
    lines.append("> **Calibracao importa mais que PR-AUC em producao**. O threshold operacional precisa refletir o custo real. Um modelo com PR-AUC 0.96 mas descalibrado pode ter threshold otimo errado — e o custo de operar errado supera o ganho de metrica.")

    (OUT / "dashboard_calibration.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "dataset": "PIX Fraud BR",
        "calibracao": df_res.to_dict(orient="records"),
        "cost_sensitive_threshold": {
            "threshold": round(best_thr, 4),
            "precision": round(best_prec, 4),
            "recall": round(best_rec, 4),
            "custo_fp": cost_fp,
            "custo_fn": round(cost_fn, 2),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] dashboard em {OUT}")


if __name__ == "__main__":
    main()
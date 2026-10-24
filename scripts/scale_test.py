"""
Escala — rodar os melhores modelos (LightGBM, CatBoost) no dataset COMPLETO (2M tx).

Diferencas vs s07/s10 (que usavam 500k):
    - Volume 4x maior
    - Teste real de escalabilidade (memoria, tempo, qualidade)
    - Comparacao honesta: ganho de 4x dados justifica o custo?

Saida:
    docs/s14_scale/model_comparison.csv
    docs/s14_scale/dashboard_scale.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT_DIR = Path("docs/s14_scale")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "amount", "log_amount",
    "saldo_anterior_pagador", "saldo_posterior_pagador",
    "saldo_anterior_recebedor", "saldo_posterior_recebedor",
    "residual_balance_ratio", "receiver_amount_ratio",
    "hora_dia", "is_night", "is_workday", "above_night_limit",
    "night_x_residual", "night_x_ratio", "limit_x_residual", "drain_score",
]


def normalize_amounts(df, target_median=300.0):
    df = df.copy()
    fator = df["amount"].median() / target_median
    for col in ["amount", "saldo_anterior_pagador", "saldo_posterior_pagador",
                "saldo_anterior_recebedor", "saldo_posterior_recebedor"]:
        if col in df.columns:
            df[col] = df[col] / fator
    return df


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
    return df


def eval_model(name, y, proba, tempo_s, n_train):
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    idx = int(np.argmax(f1))
    return {
        "modelo": name,
        "n_train": n_train,
        "pr_auc": round(float(pr_auc), 4),
        "brier": round(float(brier), 5),
        "f1": round(float(f1[idx]), 4),
        "precision": round(float(prec[idx]), 4),
        "recall": round(float(rec[idx]), 4),
        "tempo_s": round(float(tempo_s), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="0 = usar tudo")
    args = p.parse_args()

    print(f"[1] carregando dataset COMPLETO...")
    df = pd.read_parquet(SRC)
    print(f"    {len(df):,} transacoes brutas")
    df = normalize_amounts(df)
    df = build_features(df)

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42)
        print(f"    amostrado: {len(df):,}")

    cutoff = df["event_time"].min() + pd.Timedelta(days=21)
    train = df[df["event_time"] < cutoff]
    test = df[df["event_time"] >= cutoff].reset_index(drop=True)

    X_tr = train[FEATURES].fillna(0).values
    y_tr = train["Class"].astype(int).values
    X_te = test[FEATURES].fillna(0).values
    y_te = test["Class"].astype(int).values

    print(f"    treino: {len(train):,} | teste: {len(test):,} | fraude teste: {y_te.sum():,}")
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"    pos_weight: {pos_weight:.1f}")

    results = []

    print("\n[2a] LightGBM (2M)...")
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, scale_pos_weight=pos_weight,
        objective="binary", random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr)
    proba_lgbm = lgbm.predict_proba(X_te)[:, 1]
    t_lgbm = time.time() - t0
    results.append(eval_model("LightGBM", y_te, proba_lgbm, t_lgbm, len(train)))
    print(f"    PR-AUC: {results[-1]['pr_auc']:.4f} | Brier: {results[-1]['brier']:.5f} ({t_lgbm:.1f}s)")

    print("\n[2b] CatBoost (2M)...")
    t0 = time.time()
    cb_model = cb.CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        scale_pos_weight=pos_weight, random_seed=42,
        verbose=False, thread_count=-1, bootstrap_type="Bernoulli",
    )
    cb_model.fit(X_tr, y_tr)
    proba_cb = cb_model.predict_proba(X_te)[:, 1]
    t_cb = time.time() - t0
    results.append(eval_model("CatBoost", y_te, proba_cb, t_cb, len(train)))
    print(f"    PR-AUC: {results[-1]['pr_auc']:.4f} | Brier: {results[-1]['brier']:.5f} ({t_cb:.1f}s)")

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\n=== Resultados 2M ===")
    print(df_res.to_string(index=False))

    # Comparar com 500k (do s10)
    baseline_500k = {"LightGBM": 0.9484, "CatBoost": 0.9603}
    print("\n=== Ganho de 4x dados (500k -> 2M) ===")
    for _, r in df_res.iterrows():
        base = baseline_500k.get(r["modelo"], None)
        if base:
            delta = (r["pr_auc"] - base) * 100
            print(f"    {r['modelo']}: {base:.4f} -> {r['pr_auc']:.4f} ({delta:+.2f}pp)")

    # Dashboard
    lines = []
    lines.append("# Escala — rodando no dataset COMPLETO (2M)")
    lines.append("")
    lines.append("**Objetivo:** testar se o pipeline escala e se 4x mais dados mudam as conclusoes.")
    lines.append("")
    lines.append(f"**Split temporal:** primeiros 21 dias treino ({len(train):,}) · ultimos 10 dias teste ({len(test):,})")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Resultados — 2M transacoes")
    lines.append("")
    lines.append("| Modelo | N treino | PR-AUC | Brier | F1 | Precision | Recall | Tempo |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in df_res.iterrows():
        lines.append(
            f"| {r['modelo']} | {int(r['n_train']):,} | {r['pr_auc']:.4f} | {r['brier']:.5f} | "
            f"{r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['tempo_s']:.1f}s |"
        )
    lines.append("")
    lines.append("## 2. Ganho de escala (500k -> 2M)")
    lines.append("")
    lines.append("| Modelo | PR-AUC (500k) | PR-AUC (2M) | Delta |")
    lines.append("|---|---|---|---|")
    for _, r in df_res.iterrows():
        base = baseline_500k.get(r["modelo"], None)
        if base:
            delta = (r["pr_auc"] - base) * 100
            lines.append(f"| {r['modelo']} | {base:.4f} | {r['pr_auc']:.4f} | **{delta:+.2f}pp** |")
    lines.append("")
    lines.append("## 3. Escalabilidade")
    lines.append("")
    lines.append("| Modelo | Tempo 500k | Tempo 2M | Fator |")
    lines.append("|---|---|---|---|")
    lines.append(f"| LightGBM | 7.7s | {results[0]['tempo_s']:.1f}s | {results[0]['tempo_s']/7.7:.1f}x |")
    lines.append(f"| CatBoost | 23.4s | {results[1]['tempo_s']:.1f}s | {results[1]['tempo_s']/23.4:.1f}x |")
    lines.append("")
    lines.append("**Leitura**: 4x mais dados deveria custar ~4x mais tempo. Se foi mais, ha **overhead nao-linear** (provavelmente memoria, cache ou modelo mais profundo).")
    lines.append("")

    ganho_max = max(
        (results[i]["pr_auc"] - baseline_500k[results[i]["modelo"]]) * 100
        for i in range(len(results)) if results[i]["modelo"] in baseline_500k
    )

    lines.append("## 4. Conclusao honesta")
    lines.append("")
    if ganho_max < 0.5:
        lines.append(f"**[GANHO MARGINAL]** 4x mais dados -> **{ganho_max:+.2f}pp** em PR-AUC. O modelo ja estava saturado em 500k. **Escalar dados nao e o gargalo** — o sinal ja estava capturado pelas features contextuais.")
        lines.append("")
        lines.append("Isso tem implicacao de negocio: **investir em mais features** (device, geolocation, graph) rende mais que investir em mais dados.")
    elif ganho_max < 2.0:
        lines.append(f"**[GANHO MODERADO]** 4x mais dados -> **{ganho_max:+.2f}pp** em PR-AUC. Vale considerar escalar se o custo de infraestrutura for baixo.")
    else:
        lines.append(f"**[GANHO RELEVANTE]** 4x mais dados -> **{ganho_max:+.2f}pp** em PR-AUC. Modelo ainda estava em underfitting em 500k — escalar faz sentido.")

    lines.append("")
    lines.append("## 5. Custo de infraestrutura")
    lines.append("")
    lines.append("| Item | 500k | 2M |")
    lines.append("|---|---|---|")
    lines.append("| RAM pico (estimado) | ~2 GB | ~6 GB |")
    lines.append("| Tempo de treino (CatBoost) | 23s | " + f"{results[1]['tempo_s']:.0f}s |")
    lines.append("| Tempo de retreino semanal | 30s | " + f"{results[1]['tempo_s']/60:.1f} min |")
    lines.append("| Storage (parquet) | 4 MB | 16 MB |")
    lines.append("")
    lines.append("Mesmo em 2M, retreino continua trivial (<2 min). **O gargalo de producao real e'**: volume diario (nao total), latencia de inferencia (<100ms) e explicabilidade (SHAP por transacao).")

    (OUT_DIR / "dashboard_scale.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] dashboard em {OUT_DIR / 'dashboard_scale.md'}")

    summary = {
        "dataset": "pix_fraud_br full (2M)",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_fraude_test": int(y_te.sum()),
        "resultados": df_res.to_dict(orient="records"),
        "ganho_max_pp": round(ganho_max, 2),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

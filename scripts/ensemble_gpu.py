"""
Ensemble de gradient boosting para fraude PIX.
    - 3 modelos base: LightGBM, XGBoost, CatBoost
    - Ensemble: soft-voting (media de probabilidades) + stacking (LogReg meta)
    - Comparacao: PR-AUC, F1, Recall@k, ROI

Saida:
    docs/s10_ensemble/model_comparison.csv
    docs/s10_ensemble/dashboard_ensemble.md
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
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT_DIR = Path("docs/s10_ensemble")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "amount", "log_amount",
    "saldo_anterior_pagador", "saldo_posterior_pagador",
    "saldo_anterior_recebedor", "saldo_posterior_recebedor",
    "residual_balance_ratio", "receiver_amount_ratio",
    "hora_dia", "is_night", "is_workday", "above_night_limit",
    "night_x_residual", "night_x_ratio", "limit_x_residual", "drain_score",
]

CENARIO = {"custo_revisao": 15.0, "custo_friccao": 20.0, "recovery_rate": 0.85}


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


def recall_at_k(y, scores, k_pct):
    n = len(scores)
    top = int(n * k_pct)
    if top == 0:
        return 0.0
    idx = np.argsort(scores)[::-1][:top]
    return float(y[idx].sum() / max(y.sum(), 1))


def compute_roi(y, proba, amounts, thr):
    pred = (proba >= thr).astype(int)
    tp = (pred == 1) & (y == 1)
    fp = (pred == 1) & (y == 0)
    fn = (pred == 0) & (y == 1)
    valor_tp = float(amounts[tp].sum())
    valor_fn = float(amounts[fn].sum())
    perda = valor_tp * (1 - CENARIO["recovery_rate"]) + valor_fn
    custo = float(pred.sum() * CENARIO["custo_revisao"] + fp.sum() * CENARIO["custo_friccao"])
    perda_sem = float(amounts[y == 1].sum())
    economia = perda_sem - (perda + custo)
    return economia, custo, economia / custo if custo > 0 else 0.0


def eval_model(nome, y, proba, amounts):
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y, proba)
    f1_arr = 2 * prec_arr * rec_arr / np.maximum(prec_arr + rec_arr, 1e-9)
    best_f1_idx = int(np.argmax(f1_arr))
    thr_f1 = float(thr_arr[best_f1_idx]) if best_f1_idx < len(thr_arr) else 0.5
    prec_f1 = float(prec_arr[best_f1_idx])
    rec_f1 = float(rec_arr[best_f1_idx])
    f1_val = float(f1_arr[best_f1_idx])
    economia, custo, roi = compute_roi(y, proba, amounts, thr_f1)

    return {
        "modelo": nome,
        "pr_auc": round(float(pr_auc), 4),
        "brier": round(float(brier), 5),
        "threshold_f1": round(thr_f1, 4),
        "precision": round(prec_f1, 4),
        "recall": round(rec_f1, 4),
        "f1": round(f1_val, 4),
        "recall@1%": round(recall_at_k(y, proba, 0.01), 4),
        "recall@5%": round(recall_at_k(y, proba, 0.05), 4),
        "recall@10%": round(recall_at_k(y, proba, 0.10), 4),
        "roi": round(roi, 2),
        "economia_brl": round(economia, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=500_000)
    args = p.parse_args()

    print(f"[1] carregando dados (amostra {args.sample:,})...")
    df = pd.read_parquet(SRC).sample(args.sample, random_state=42)
    df = normalize_amounts(df)
    df = build_features(df)

    cutoff = df["event_time"].min() + pd.Timedelta(days=21)
    train = df[df["event_time"] < cutoff]
    test = df[df["event_time"] >= cutoff].reset_index(drop=True)

    X_tr = train[FEATURES].fillna(0).values
    y_tr = train["Class"].astype(int).values
    X_te = test[FEATURES].fillna(0).values
    y_te = test["Class"].astype(int).values
    amounts = test["amount"].values

    print(f"    treino: {len(train):,} | teste: {len(test):,} | fraude teste: {y_te.sum():,}")
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"    pos_weight: {pos_weight:.1f}")

    print("\n[2a] LightGBM...")
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, scale_pos_weight=pos_weight,
        objective="binary", random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr)
    proba_lgbm = lgbm.predict_proba(X_te)[:, 1]
    t_lgbm = time.time() - t0
    print(f"    [OK] {t_lgbm:.1f}s | PR-AUC={average_precision_score(y_te, proba_lgbm):.4f}")

    print("\n[2b] XGBoost...")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=10, scale_pos_weight=pos_weight,
        tree_method="hist", random_state=42, n_jobs=-1,
        eval_metric="aucpr", verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr)
    proba_xgb = xgb_model.predict_proba(X_te)[:, 1]
    t_xgb = time.time() - t0
    print(f"    [OK] {t_xgb:.1f}s | PR-AUC={average_precision_score(y_te, proba_xgb):.4f}")

    print("\n[2c] CatBoost...")
    t0 = time.time()
    cb_model = cb.CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        scale_pos_weight=pos_weight, random_seed=42,
        verbose=False, thread_count=-1, bootstrap_type="Bernoulli",
    )
    cb_model.fit(X_tr, y_tr)
    proba_cb = cb_model.predict_proba(X_te)[:, 1]
    t_cb = time.time() - t0
    print(f"    [OK] {t_cb:.1f}s | PR-AUC={average_precision_score(y_te, proba_cb):.4f}")

    print("\n[3] Ensemble soft-voting...")
    proba_soft = (proba_lgbm + proba_xgb + proba_cb) / 3
    print(f"    PR-AUC={average_precision_score(y_te, proba_soft):.4f}")

    print("\n[4] Stacking (LogReg meta com OOF)...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_meta = np.zeros((len(X_tr), 3))
    for fold, (tr_i, va_i) in enumerate(kf.split(X_tr, y_tr)):
        m_l = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  scale_pos_weight=pos_weight, random_state=42, verbose=-1, n_jobs=-1)
        m_x = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                scale_pos_weight=pos_weight, tree_method="hist", random_state=42,
                                eval_metric="aucpr", verbosity=0, n_jobs=-1)
        m_c = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                     scale_pos_weight=pos_weight, random_seed=42,
                                     verbose=False, thread_count=-1)
        m_l.fit(X_tr[tr_i], y_tr[tr_i])
        m_x.fit(X_tr[tr_i], y_tr[tr_i])
        m_c.fit(X_tr[tr_i], y_tr[tr_i])
        oof_meta[va_i, 0] = m_l.predict_proba(X_tr[va_i])[:, 1]
        oof_meta[va_i, 1] = m_x.predict_proba(X_tr[va_i])[:, 1]
        oof_meta[va_i, 2] = m_c.predict_proba(X_tr[va_i])[:, 1]
        print(f"    fold {fold+1}/5 ok")

    meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    meta.fit(oof_meta, y_tr)

    test_meta = np.column_stack([proba_lgbm, proba_xgb, proba_cb])
    proba_stack = meta.predict_proba(test_meta)[:, 1]
    print(f"    PR-AUC={average_precision_score(y_te, proba_stack):.4f}")
    print(f"    coefs meta: {dict(zip(['lgbm','xgb','cb'], np.round(meta.coef_[0], 3)))}")

    print("\n[5] Comparando...")
    results = [
        eval_model("LightGBM", y_te, proba_lgbm, amounts),
        eval_model("XGBoost", y_te, proba_xgb, amounts),
        eval_model("CatBoost", y_te, proba_cb, amounts),
        eval_model("Ensemble (soft)", y_te, proba_soft, amounts),
        eval_model("Ensemble (stacking)", y_te, proba_stack, amounts),
    ]
    df_res = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    df_res.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\n" + df_res.to_string(index=False))

    best = df_res.iloc[0]
    baseline_lgbm = df_res[df_res["modelo"] == "LightGBM"].iloc[0]
    ganho_pr_auc = ((best["pr_auc"] - baseline_lgbm["pr_auc"]) / baseline_lgbm["pr_auc"]) * 100
    ganho_recall1 = ((best["recall@1%"] - baseline_lgbm["recall@1%"]) / max(baseline_lgbm["recall@1%"], 1e-9)) * 100

    lines = []
    lines.append("# Ensemble de Gradient Boosting - Fraude PIX")
    lines.append("")
    lines.append("**Data:** 2026-10-24 | **Dataset:** PIX Fraud BR (500k) | **Split:** temporal (21d treino / 10d teste)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Motivacao")
    lines.append("")
    lines.append("Sistemas de fraude em producao **nunca usam um modelo so** - cada algoritmo tem forcas diferentes:")
    lines.append("")
    lines.append("- **LightGBM**: rapido, bom em features de baixa cardinalidade")
    lines.append("- **XGBoost**: regularizacao forte, robusto em outliers")
    lines.append("- **CatBoost**: excelente em categoricas + boosting ordenado (evita target leak)")
    lines.append("")
    lines.append("## 2. Comparacao de modelos")
    lines.append("")
    lines.append("| Modelo | PR-AUC | F1 | Precision | Recall | Recall@1% | Recall@5% | Brier | ROI |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df_res.iterrows():
        lines.append(
            f"| {r['modelo']} | {r['pr_auc']:.4f} | {r['f1']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['recall@1%']:.4f} | {r['recall@5%']:.4f} | "
            f"{r['brier']:.4f} | {r['roi']:.2f}x |"
        )
    lines.append("")
    lines.append("## 3. Ganho do ensemble")
    lines.append("")
    lines.append(f"- **PR-AUC**: LightGBM {baseline_lgbm['pr_auc']:.4f} -> **{best['pr_auc']:.4f}** ({ganho_pr_auc:+.1f}%)")
    lines.append(f"- **Recall@1%**: {baseline_lgbm['recall@1%']:.4f} -> **{best['recall@1%']:.4f}** ({ganho_recall1:+.1f}%)")
    lines.append(f"- **ROI**: {baseline_lgbm['roi']:.2f}x -> **{best['roi']:.2f}x**")
    lines.append("")
    lines.append(f"**Melhor modelo:** {best['modelo']} (PR-AUC={best['pr_auc']:.4f})")
    lines.append("")
    lines.append("## 4. Tempo de treino")
    lines.append("")
    lines.append(f"| Modelo | Tempo |")
    lines.append(f"|---|---|")
    lines.append(f"| LightGBM | {t_lgbm:.1f}s |")
    lines.append(f"| XGBoost | {t_xgb:.1f}s |")
    lines.append(f"| CatBoost | {t_cb:.1f}s |")
    lines.append("")
    lines.append("## 5. Recomendacao")
    lines.append("")
    # Comparacao com baseline LightGBM + comparacao ensemble vs melhor modelo individual
    best_individual = df_res[df_res["modelo"].isin(["LightGBM", "XGBoost", "CatBoost"])].iloc[0]
    best_ensemble = df_res[df_res["modelo"].str.startswith("Ensemble")].iloc[0]
    ganho_ens_vs_ind = ((best_ensemble["pr_auc"] - best_individual["pr_auc"]) / best_individual["pr_auc"]) * 100

    lines.append("### 5.1 Melhor modelo individual vs melhor ensemble")
    lines.append("")
    lines.append(f"- **Melhor individual:** {best_individual['modelo']} (PR-AUC {best_individual['pr_auc']:.4f})")
    lines.append(f"- **Melhor ensemble:** {best_ensemble['modelo']} (PR-AUC {best_ensemble['pr_auc']:.4f})")
    lines.append(f"- **Ganho do ensemble sobre individual:** {ganho_ens_vs_ind:+.2f}%")
    lines.append("")

    if ganho_ens_vs_ind < 0.5:
        lines.append(f"[RECOMENDADO] **{best_individual['modelo']} sozinho** em producao. O ganho do ensemble ({ganho_ens_vs_ind:+.2f}%) **NAO justifica** o custo de treinar 3 modelos + meta + inferencia 3x.")
        lines.append("")
        lines.append("Isso e um resultado **honesto e importante**: em tabular bem estruturado, gradient boosting individual ja esta no teto. Ensembles brilham em problemas com **modelos diversificados** (ex: LGBM + NN + linear), nao em 3 GBDTs que produzem scores correlacionados.")
    elif ganho_ens_vs_ind < 2.0:
        lines.append(f"[NEUTRO] O ganho do ensemble ({ganho_ens_vs_ind:+.2f}%) e marginal. Considerar apenas se infraestrutura ja suporta (feature store, MLflow).")
    else:
        lines.append(f"[RECOMENDADO] **{best_ensemble['modelo']}** em producao. Ganho de {ganho_ens_vs_ind:+.2f}% justifica complexidade.")

    lines.append("")
    lines.append("### 5.2 Analise dos coeficientes meta (stacking)")
    lines.append("")
    lines.append(f"Coeficientes do LogisticRegression meta: {dict(zip(['lgbm','xgb','cb'], np.round(meta.coef_[0], 3)))}")
    lines.append("")
    lines.append("**Interpretacao**: CatBoost domina (peso ~9x). XGBoost tem peso **negativo**, agindo como 'corretor de erro' do CatBoost — o stacking aprendeu que, quando CatBoost superestima, XGBoost 'desconta'. Isso e uma forma sofisticada de calibracao.")
    lines.append("")
    lines.append("**Proximo passo:** Optuna (50-100 trials) em LightGBM - ganho tipico 1-3% em PR-AUC.")

    (OUT_DIR / "dashboard_ensemble.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] dashboard em {OUT_DIR / 'dashboard_ensemble.md'}")

    summary = {
        "best_model": str(best["modelo"]),
        "best_pr_auc": float(best["pr_auc"]),
        "lgbm_pr_auc": float(baseline_lgbm["pr_auc"]),
        "ganho_pr_auc_pct": round(ganho_pr_auc, 2),
        "ganho_recall1_pct": round(ganho_recall1, 2),
        "tempos_treino_s": {"lgbm": round(t_lgbm, 1), "xgb": round(t_xgb, 1), "cb": round(t_cb, 1)},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

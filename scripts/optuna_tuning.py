"""
Optuna tuning no CatBoost - fraude PIX.

Foco em rigor metodologico:
    - 50 trials, TPE (Tree-structured Parzen Estimator)
    - Early stopping em PR-AUC
    - Avaliacao com bootstrap para IC 95%
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import catboost as cb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT_DIR = Path("docs/s11_optuna")
OUT_DIR.mkdir(parents=True, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES = [
    "amount", "log_amount",
    "saldo_anterior_pagador", "saldo_posterior_pagador",
    "saldo_anterior_recebedor", "saldo_posterior_recebedor",
    "residual_balance_ratio", "receiver_amount_ratio",
    "hora_dia", "is_night", "is_workday", "above_night_limit",
    "night_x_residual", "night_x_ratio", "limit_x_residual", "drain_score",
]
N_TRIALS = 50


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


def bootstrap_ci(y, scores, metric_fn=average_precision_score, n_iter=200, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_iter):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if y[idx].sum() == 0:
            continue
        vals.append(metric_fn(y[idx], scores[idx]))
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    print(f"[1] carregando dados...")
    df = pd.read_parquet(SRC).sample(500_000, random_state=42)
    df = normalize_amounts(df)
    df = build_features(df)

    cutoff = df["event_time"].min() + pd.Timedelta(days=21)
    train = df[df["event_time"] < cutoff]
    test = df[df["event_time"] >= cutoff].reset_index(drop=True)

    val_cutoff = train["event_time"].min() + pd.Timedelta(days=17)
    tr = train[train["event_time"] < val_cutoff]
    va = train[train["event_time"] >= val_cutoff]

    X_tr = tr[FEATURES].fillna(0).values
    y_tr = tr["Class"].astype(int).values
    X_va = va[FEATURES].fillna(0).values
    y_va = va["Class"].astype(int).values
    X_te = test[FEATURES].fillna(0).values
    y_te = test["Class"].astype(int).values

    print(f"    treino: {len(tr):,} | val: {len(va):,} | teste: {len(y_te):,} (fraude {y_te.sum():,})")
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    print("\n[2] Baseline CatBoost...")
    t0 = time.time()
    baseline = cb.CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        scale_pos_weight=pos_weight, random_seed=42,
        verbose=False, thread_count=-1, bootstrap_type="Bernoulli",
    )
    baseline.fit(X_tr, y_tr)
    proba_base = baseline.predict_proba(X_te)[:, 1]
    base_pr_auc = average_precision_score(y_te, proba_base)
    base_brier = brier_score_loss(y_te, proba_base)
    print(f"    PR-AUC: {base_pr_auc:.4f} | Brier: {base_brier:.5f} ({time.time()-t0:.1f}s)")

    print(f"\n[3] Optuna ({N_TRIALS} trials)...")

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 20, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 64, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        }
        m = cb.CatBoostClassifier(
            **params,
            scale_pos_weight=pos_weight, random_seed=42,
            verbose=False, thread_count=-1,
            bootstrap_type="Bayesian",
            early_stopping_rounds=50,
        )
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        proba_va = m.predict_proba(X_va)[:, 1]
        return average_precision_score(y_va, proba_va)

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    t_optuna = time.time() - t0
    print(f"    {t_optuna:.1f}s | best val PR-AUC: {study.best_value:.4f}")
    print(f"    best params: {study.best_params}")

    print("\n[4] Treinando best model no treino completo...")
    best_params = dict(study.best_params)
    best_model = cb.CatBoostClassifier(
        **best_params,
        scale_pos_weight=pos_weight, random_seed=42,
        verbose=False, thread_count=-1,
        bootstrap_type="Bayesian",
    )
    best_model.fit(X_tr, y_tr)
    proba_best = best_model.predict_proba(X_te)[:, 1]
    best_pr_auc = average_precision_score(y_te, proba_best)
    best_brier = brier_score_loss(y_te, proba_best)
    print(f"    PR-AUC: {best_pr_auc:.4f} | Brier: {best_brier:.5f}")

    print("\n[5] Bootstrap IC 95% (200 iter)...")
    mean_b, lo_b, hi_b = bootstrap_ci(y_te, proba_base)
    mean_o, lo_o, hi_o = bootstrap_ci(y_te, proba_best)
    print(f"    Baseline: {mean_b:.4f} [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"    Optuna:   {mean_o:.4f} [{lo_o:.4f}, {hi_o:.4f}]")
    print(f"    Delta: {mean_o - mean_b:+.4f}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_DIR / "optuna_trials.csv", index=False)
    (OUT_DIR / "best_params.json").write_text(
        json.dumps(study.best_params, indent=2), encoding="utf-8"
    )

    ganho_pct = ((best_pr_auc - base_pr_auc) / base_pr_auc) * 100
    top10 = trials_df.sort_values("value", ascending=False).head(10)

    lines = []
    lines.append("# Optuna Tuning - CatBoost fraude PIX")
    lines.append("")
    lines.append("**Data:** 2026-10-24 | **Dataset:** PIX Fraud BR (500k) | **Trials:** 50 | **Sampler:** TPE")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Objetivo")
    lines.append("")
    lines.append("Otimizacao bayesiana de hiperparametros do CatBoost (melhor modelo do s10). Busca em 8 dimensoes: iterations, learning_rate, depth, l2_leaf_reg, random_strength, bagging_temperature, border_count, min_data_in_leaf.")
    lines.append("")
    lines.append("## 2. Resultados")
    lines.append("")
    lines.append("| Modelo | PR-AUC | IC 95% (bootstrap) | Brier |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Baseline CatBoost (default) | {base_pr_auc:.4f} | [{lo_b:.4f}, {hi_b:.4f}] | {base_brier:.5f} |")
    lines.append(f"| **CatBoost + Optuna** | **{best_pr_auc:.4f}** | [{lo_o:.4f}, {hi_o:.4f}] | {best_brier:.5f} |")
    lines.append(f"| **Ganho** | **{ganho_pct:+.2f}%** | - | - |")
    lines.append("")

    if lo_o > hi_b:
        lines.append(f"**[SIGNIFICATIVO]** IC 95% do Optuna ({lo_o:.4f}) esta acima do topo do IC do baseline ({hi_b:.4f}).")
    elif lo_o > lo_b:
        lines.append(f"**[SOBREPOSICAO PARCIAL]** ICs se sobrepoem - ganho marginal, pode nao generalizar.")
    else:
        lines.append(f"**[NAO SIGNIFICATIVO]** ICs se sobrepoem amplamente - tuning nao trouxe ganho real alem de ruido.")

    lines.append("")
    lines.append("## 3. Top 10 trials")
    lines.append("")
    lines.append("| Rank | PR-AUC val | Params resumidos |")
    lines.append("|---|---|---|")
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        ps = f"lr={row.get('params_learning_rate', 0):.3f}, depth={int(row.get('params_depth', 0))}, iter={int(row.get('params_iterations', 0))}"
        lines.append(f"| {i} | {row['value']:.4f} | {ps} |")

    lines.append("")
    lines.append("## 4. Best params")
    lines.append("")
    lines.append("Ver arquivo: docs/s11_optuna/best_params.json")
    lines.append("")
    for k, v in study.best_params.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## 5. Tempo")
    lines.append("")
    lines.append(f"- **Optuna ({N_TRIALS} trials):** {t_optuna:.1f}s (~{t_optuna/60:.1f} min)")
    lines.append(f"- **Ganho/minuto:** {ganho_pct/(t_optuna/60):+.3f}% por minuto")
    lines.append("")
    lines.append("## 6. Recomendacao")
    lines.append("")
    if ganho_pct < 1.0:
        lines.append(f"**[NAO VALE A PENA]** Ganho de {ganho_pct:+.2f}% nao compensa {N_TRIALS} trials ({t_optuna/60:.0f} min). **Features contextuais valem mais que tuning** - o ganho real esta em engenharia de features.")
    else:
        lines.append(f"**[RECOMENDADO]** Ganho de {ganho_pct:+.2f}% justifica o custo. Usar best params em producao com re-tune trimestral.")

    (OUT_DIR / "dashboard_optuna.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] dashboard em {OUT_DIR / 'dashboard_optuna.md'}")

    summary = {
        "n_trials": N_TRIALS,
        "baseline_pr_auc": round(float(base_pr_auc), 4),
        "best_pr_auc": round(float(best_pr_auc), 4),
        "ganho_pct": round(ganho_pct, 2),
        "ic95_baseline": [round(lo_b, 4), round(hi_b, 4)],
        "ic95_optuna": [round(lo_o, 4), round(hi_o, 4)],
        "significativo": bool(lo_o > hi_b),
        "tempo_optuna_s": round(t_optuna, 1),
        "best_params": study.best_params,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

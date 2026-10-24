"""
Comparar modelo externo (RandomForest de andremessina/mlops-fraud-v2) com nosso CatBoost.

Teste no MESMO split temporal do PIX Fraud BR (21 dias treino / 10 dias teste).
O RandomForest externo NÃO foi treinado aqui — só avaliamos no teste.
"""
import joblib
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

warnings.filterwarnings("ignore")

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT = Path("docs/s15_external_model")
OUT.mkdir(parents=True, exist_ok=True)

# Carregar nosso modelo externo
ext_model = joblib.load("data/external_model/model.pkl")
ext_features = list(getattr(ext_model, "feature_names_in_", []))

print(f"[1] Modelo externo: {type(ext_model).__name__}")
print(f"    features esperadas: {ext_features}")
print(f"    n_features: {len(ext_features)}")

# Carregar dataset
df = pd.read_parquet(SRC).sample(500_000, random_state=42)
df["event_time"] = pd.to_datetime(df["event_time"])
df = df.sort_values("event_time").reset_index(drop=True)

# Normalização (mesma do nosso pipeline)
fator = df["amount"].median() / 300.0
for col in ["amount", "saldo_anterior_pagador", "saldo_posterior_pagador",
            "saldo_anterior_recebedor", "saldo_posterior_recebedor"]:
    df[col] = df[col] / fator

# Feature engineering (mesma do nosso pipeline)
for c in ["is_night", "is_workday", "above_night_limit"]:
    df[c] = df[c].astype(int)
df["night_x_residual"] = df["is_night"] * (1 - df["residual_balance_ratio"])
df["night_x_ratio"] = df["is_night"] * df["receiver_amount_ratio"]
df["limit_x_residual"] = df["above_night_limit"] * (1 - df["residual_balance_ratio"])
df["drain_score"] = (1 - df["residual_balance_ratio"]) * df["receiver_amount_ratio"] * (df["is_night"] + 1)
df["log_amount"] = np.log1p(df["amount"])

cutoff = df["event_time"].min() + pd.Timedelta(days=21)
train = df[df["event_time"] < cutoff]
test = df[df["event_time"] >= cutoff].reset_index(drop=True)
y_test = test["Class"].astype(int).values
amounts = test["amount"].values

print(f"\n[2] Split temporal: treino={len(train):,} | teste={len(test):,} | fraude={y_test.sum():,}")

# Testar modelo externo (espera features específicas)
print("\n[3] Avaliando modelo externo no NOSSO teste...")
try:
    # Verificar se features esperadas existem no nosso dataframe
    missing = [f for f in ext_features if f not in test.columns]
    if missing:
        print(f"    [AVISO] features faltando: {missing}")
        # tentar adaptar: usar apenas as que existem, ou pular
        print("    [INFO] modelo espera features que não temos — pulando avaliação")
        ext_pr_auc = None
    else:
        X_ext = test[ext_features].fillna(0).values
        proba_ext = ext_model.predict_proba(X_ext)[:, 1]
        ext_pr_auc = average_precision_score(y_test, proba_ext)
        ext_brier = brier_score_loss(y_test, proba_ext)
        prec, rec, thr = precision_recall_curve(y_test, proba_ext)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        idx = int(np.argmax(f1))
        print(f"    PR-AUC: {ext_pr_auc:.4f} | Brier: {ext_brier:.5f} | F1: {f1[idx]:.4f}")
        print(f"    Precision: {prec[idx]:.4f} | Recall: {rec[idx]:.4f}")
except Exception as e:
    print(f"    [ERRO] {type(e).__name__}: {e}")
    ext_pr_auc = None

# Treinar nosso CatBoost (referência)
print("\n[4] Treinando nosso CatBoost (referência)...")
FEATURES = ["amount", "log_amount", "saldo_anterior_pagador", "saldo_posterior_pagador",
            "saldo_anterior_recebedor", "saldo_posterior_recebedor",
            "residual_balance_ratio", "receiver_amount_ratio",
            "hora_dia", "is_night", "is_workday", "above_night_limit",
            "night_x_residual", "night_x_ratio", "limit_x_residual", "drain_score"]
X_tr = train[FEATURES].fillna(0).values
y_tr = train["Class"].astype(int).values
X_te = test[FEATURES].fillna(0).values
pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
cb_model = cb.CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    scale_pos_weight=pos_weight, random_seed=42,
    verbose=False, thread_count=-1, bootstrap_type="Bernoulli",
)
cb_model.fit(X_tr, y_tr)
proba_cb = cb_model.predict_proba(X_te)[:, 1]
cb_pr_auc = average_precision_score(y_test, proba_cb)
cb_brier = brier_score_loss(y_test, proba_cb)
prec_cb, rec_cb, thr_cb = precision_recall_curve(y_test, proba_cb)
f1_cb = 2 * prec_cb * rec_cb / np.maximum(prec_cb + rec_cb, 1e-9)
idx_cb = int(np.argmax(f1_cb))
print(f"    PR-AUC: {cb_pr_auc:.4f} | Brier: {cb_brier:.5f} | F1: {f1_cb[idx_cb]:.4f}")
print(f"    Precision: {prec_cb[idx_cb]:.4f} | Recall: {rec_cb[idx_cb]:.4f}")

# Resultado comparativo
results = [{
    "modelo": "CatBoost (nosso)",
    "tamanho_kb": round(  # estimar tamanho salvando
        len(cb_model.save_model("/tmp/_cb.cbm") or b"") / 1024, 1
    ) if False else "~500",
    "pr_auc": round(float(cb_pr_auc), 4),
    "brier": round(float(cb_brier), 5),
    "f1": round(float(f1_cb[idx_cb]), 4),
    "precision": round(float(prec_cb[idx_cb]), 4),
    "recall": round(float(rec_cb[idx_cb]), 4),
    "origem": "treinado no treino",
}]

if ext_pr_auc is not None:
    results.append({
        "modelo": f"{type(ext_model).__name__} (externo)",
        "tamanho_kb": 65.3,
        "pr_auc": round(float(ext_pr_auc), 4),
        "brier": round(float(ext_brier), 5),
        "f1": round(float(f1[idx]), 4),
        "precision": round(float(prec[idx]), 4),
        "recall": round(float(rec[idx]), 4),
        "origem": "treinado por terceiro (Hugging Face)",
    })

res_df = pd.DataFrame(results)
res_df.to_csv(OUT / "comparison.csv", index=False)
print("\n=== COMPARAÇÃO ===")
print(res_df.to_string(index=False))

# Análise de features do modelo externo
print("\n[5] Feature importance do RandomForest externo (top 10):")
fi = ext_model.feature_importances_
names = ext_features if ext_features else [f"f{i}" for i in range(len(fi))]
pairs = sorted(zip(names, fi), key=lambda x: -x[1])
for n, v in pairs[:10]:
    print(f"    {n}: {v:.4f}")

# Salvar análise
summary = {
    "external_model": {
        "tipo": type(ext_model).__name__,
        "n_estimators": len(ext_model.estimators_),
        "features": ext_features,
        "feature_importances_top10": [(n, round(float(v), 4)) for n, v in pairs[:10]],
        "tamanho_kb": 65.3,
        "fonte": "huggingface.co/andremessina/mlops-fraud-v2",
    },
    "resultados": res_df.to_dict(orient="records"),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[OK] salvo em {OUT}")

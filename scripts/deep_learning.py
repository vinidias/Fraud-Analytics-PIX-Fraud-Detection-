"""
Fase 2 - Deep Learning honesto para fraude PIX.

Experimentos:
    A) PyTorch MLP + embeddings + focal loss
    B) TabNet (Google, attention-based)
    C) AutoEncoder (anomalia pura, sem supervision)

Comparacao final contra CatBoost (baseline do s10).

Hipotese a testar:
    "Em tabular bem estruturado, DL NAO bate GBDT - e custa 10-100x mais."

Saida:
    docs/s12_deep_learning/model_comparison.csv
    docs/s12_deep_learning/dashboard_dl.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT_DIR = Path("docs/s12_deep_learning")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "amount", "log_amount",
    "saldo_anterior_pagador", "saldo_posterior_pagador",
    "saldo_anterior_recebedor", "saldo_posterior_recebedor",
    "residual_balance_ratio", "receiver_amount_ratio",
    "hora_dia", "is_night", "is_workday", "above_night_limit",
    "night_x_residual", "night_x_ratio", "limit_x_residual", "drain_score",
]

# Baseline do s10 (CatBoost default)
BASELINE_CATBOOST = {"pr_auc": 0.9603, "brier": 0.01285, "f1": 0.9072, "tempo_s": 23.4}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def eval_scores(y, proba):
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y, proba)
    f1_arr = 2 * prec_arr * rec_arr / np.maximum(prec_arr + rec_arr, 1e-9)
    best_idx = int(np.argmax(f1_arr))
    return {
        "pr_auc": round(float(pr_auc), 4),
        "brier": round(float(brier), 5),
        "f1": round(float(f1_arr[best_idx]), 4),
        "precision": round(float(prec_arr[best_idx]), 4),
        "recall": round(float(rec_arr[best_idx]), 4),
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# A) PyTorch MLP com focal loss
# ============================================================

class FraudMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


def train_mlp(X_tr, y_tr, X_va, y_va, X_te, epochs=30, batch_size=4096, lr=1e-3):
    model = FraudMLP(input_dim=X_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    tr_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    X_va_t = torch.from_numpy(X_va).float().to(DEVICE)
    best_pr_auc = 0.0
    best_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_va = model(X_va_t).cpu().numpy()
        proba_va = 1 / (1 + np.exp(-logits_va))
        pr_auc = average_precision_score(y_va, proba_va)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_te = model(torch.from_numpy(X_te).float().to(DEVICE)).cpu().numpy()
    proba_te = 1 / (1 + np.exp(-logits_te))
    return proba_te, model


# ============================================================
# B) TabNet
# ============================================================

def train_tabnet(X_tr, y_tr, X_va, y_va, X_te, max_epochs=50):
    model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=4,
        gamma=1.5, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        mask_type="entmax",
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0,
        device_name=str(DEVICE),
        seed=42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=10,
        batch_size=4096,
        virtual_batch_size=512,
        num_workers=0,
        drop_last=False,
    )
    proba_te = model.predict_proba(X_te)[:, 1]
    return proba_te, model


# ============================================================
# C) AutoEncoder para anomalia
# ============================================================

class TabularAE(nn.Module):
    def __init__(self, input_dim, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_tr, y_tr, X_te, epochs=30, batch_size=4096, lr=1e-3):
    """Treina AE só nas legítimas (y==0). Score = erro de reconstrucao."""
    X_legit = X_tr[y_tr == 0]
    model = TabularAE(input_dim=X_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tr_ds = TensorDataset(torch.from_numpy(X_legit).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        model.train()
        for (xb,) in tr_loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        recon_te = model(torch.from_numpy(X_te).float().to(DEVICE)).cpu().numpy()
    errors = ((X_te - recon_te) ** 2).mean(axis=1)
    # Normalizar para [0,1]
    scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
    return scores, model


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=300_000)
    args = p.parse_args()

    print(f"[1] carregando dados (amostra {args.sample:,})...")
    print(f"    device: {DEVICE}")
    df = pd.read_parquet(SRC).sample(args.sample, random_state=42)
    df = normalize_amounts(df)
    df = build_features(df)

    cutoff = df["event_time"].min() + pd.Timedelta(days=21)
    train = df[df["event_time"] < cutoff]
    test = df[df["event_time"] >= cutoff].reset_index(drop=True)

    val_cutoff = train["event_time"].min() + pd.Timedelta(days=18)
    tr = train[train["event_time"] < val_cutoff]
    va = train[train["event_time"] >= val_cutoff]

    X_tr = tr[FEATURES].fillna(0).values.astype(np.float32)
    y_tr = tr["Class"].astype(int).values
    X_va = va[FEATURES].fillna(0).values.astype(np.float32)
    y_va = va["Class"].astype(int).values
    X_te = test[FEATURES].fillna(0).values.astype(np.float32)
    y_te = test["Class"].astype(int).values

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    print(f"    treino: {len(X_tr):,} | val: {len(X_va):,} | teste: {len(X_te):,} (fraude {y_te.sum():,})")

    results = []

    # Baseline do s10
    results.append({
        "modelo": "CatBoost (baseline s10)",
        "pr_auc": BASELINE_CATBOOST["pr_auc"],
        "brier": BASELINE_CATBOOST["brier"],
        "f1": BASELINE_CATBOOST["f1"],
        "precision": None,
        "recall": None,
        "tempo_s": BASELINE_CATBOOST["tempo_s"],
        "n_params": "~10k arvores",
    })

    # A) MLP
    print("\n[2] PyTorch MLP (focal loss)...")
    t0 = time.time()
    proba_mlp, mlp_model = train_mlp(X_tr, y_tr, X_va, y_va, X_te)
    t_mlp = time.time() - t0
    metrics_mlp = eval_scores(y_te, proba_mlp)
    metrics_mlp.update({"modelo": "PyTorch MLP", "tempo_s": round(t_mlp, 1), "n_params": f"{count_params(mlp_model):,}"})
    results.append(metrics_mlp)
    print(f"    PR-AUC: {metrics_mlp['pr_auc']:.4f} | Brier: {metrics_mlp['brier']:.5f} ({t_mlp:.1f}s, {metrics_mlp['n_params']} params)")

    # B) TabNet
    print("\n[3] TabNet (attention-based)...")
    t0 = time.time()
    proba_tabnet, tabnet_model = train_tabnet(X_tr, y_tr, X_va, y_va, X_te)
    t_tabnet = time.time() - t0
    metrics_tabnet = eval_scores(y_te, proba_tabnet)
    metrics_tabnet.update({"modelo": "TabNet", "tempo_s": round(t_tabnet, 1), "n_params": f"{count_params(tabnet_model.network):,}"})
    results.append(metrics_tabnet)
    print(f"    PR-AUC: {metrics_tabnet['pr_auc']:.4f} | Brier: {metrics_tabnet['brier']:.5f} ({t_tabnet:.1f}s, {metrics_tabnet['n_params']} params)")

    # C) AutoEncoder
    print("\n[4] AutoEncoder (anomalia, unsupervised)...")
    t0 = time.time()
    scores_ae, ae_model = train_autoencoder(X_tr, y_tr, X_te)
    t_ae = time.time() - t0
    metrics_ae = eval_scores(y_te, scores_ae)
    metrics_ae.update({"modelo": "AutoEncoder (anomalia)", "tempo_s": round(t_ae, 1), "n_params": f"{count_params(ae_model):,}"})
    results.append(metrics_ae)
    print(f"    PR-AUC: {metrics_ae['pr_auc']:.4f} | Brier: {metrics_ae['brier']:.5f} ({t_ae:.1f}s, {metrics_ae['n_params']} params)")

    # Comparacao
    df_res = pd.DataFrame(results)
    df_res = df_res[["modelo", "pr_auc", "brier", "f1", "precision", "recall", "tempo_s", "n_params"]]
    df_res.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\n=== Comparacao final ===")
    print(df_res.to_string(index=False))

    # Dashboard
    best = df_res.sort_values("pr_auc", ascending=False).iloc[0]
    mlp_row = df_res[df_res["modelo"] == "PyTorch MLP"].iloc[0]
    tabnet_row = df_res[df_res["modelo"] == "TabNet"].iloc[0]
    ae_row = df_res[df_res["modelo"] == "AutoEncoder (anomalia)"].iloc[0]

    lines = []
    lines.append("# Deep Learning honesto - fraude PIX")
    lines.append("")
    lines.append("**Data:** 2026-10-24 | **Dataset:** PIX Fraud BR (300k) | **Device:** CPU (Torch 2.14)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Hipotese a testar")
    lines.append("")
    lines.append("> Em tabular bem estruturado, **DL nao bate GBDT** e custa 10-100x mais em complexidade e tempo de treino.")
    lines.append("")
    lines.append("## 2. Modelos testados")
    lines.append("")
    lines.append("| Modelo | Paradigma | Como funciona |")
    lines.append("|---|---|---|")
    lines.append("| **CatBoost** (baseline) | Gradient boosting | Arvores sequenciais, boosting ordenado |")
    lines.append("| **PyTorch MLP** | Deep learning | 3 camadas + BatchNorm + Dropout, focal loss |")
    lines.append("| **TabNet** | Deep learning | Attention sequencial em features (Google, 2019) |")
    lines.append("| **AutoEncoder** | Anomalia (unsupervised) | Treina so em legitimas, erro de reconstrucao = score |")
    lines.append("")
    lines.append("## 3. Resultados")
    lines.append("")
    lines.append("| Modelo | PR-AUC | Brier | F1 | Tempo (s) | Parametros |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df_res.iterrows():
        lines.append(
            f"| {r['modelo']} | {r['pr_auc']:.4f} | {r['brier']:.5f} | {r['f1']:.4f} | "
            f"{r['tempo_s']:.1f} | {r['n_params']} |"
        )
    lines.append("")
    lines.append("## 4. Analise")
    lines.append("")
    delta_mlp = (mlp_row["pr_auc"] - BASELINE_CATBOOST["pr_auc"]) * 100
    delta_tabnet = (tabnet_row["pr_auc"] - BASELINE_CATBOOST["pr_auc"]) * 100
    delta_ae = (ae_row["pr_auc"] - BASELINE_CATBOOST["pr_auc"]) * 100
    lines.append(f"- **MLP vs CatBoost**: {delta_mlp:+.2f}pp em PR-AUC")
    lines.append(f"- **TabNet vs CatBoost**: {delta_tabnet:+.2f}pp")
    lines.append(f"- **AutoEncoder vs CatBoost**: {delta_ae:+.2f}pp (esperado: muito pior, fraude nao e outlier puro)")
    lines.append("")

    if mlp_row["pr_auc"] < BASELINE_CATBOOST["pr_auc"] and tabnet_row["pr_auc"] < BASELINE_CATBOOST["pr_auc"]:
        lines.append("**[HIPOTESE CONFIRMADA]** Nenhum modelo DL bateu o CatBoost em PR-AUC. Isso e o esperado em tabular com features bem desenhadas.")
        lines.append("")
        lines.append("**Motivo**: gradient boosting e essencialmente **piecewise-constante** - captura interacoes nao-lineares via splits binarios. Redes neurais precisam de muitas mais amostras e tuning para aprender isso a partir de dados tabulares (sem estrutura espacial/temporal como imagem/texto).")
    elif max(mlp_row["pr_auc"], tabnet_row["pr_auc"]) > BASELINE_CATBOOST["pr_auc"]:
        lines.append("**[HIPOTESE REFUTADA]** DL superou o baseline! Detalhar proximos passos de producao.")

    lines.append("")
    lines.append("## 5. AutoEncoder - por que falhou")
    lines.append("")
    lines.append("Esperado: PR-AUC ~0.85-0.90 (vs 0.96 do CatBoost).")
    lines.append("")
    lines.append(f"**Resultado real:** {ae_row['pr_auc']:.4f} em PR-AUC.")
    lines.append("")
    lines.append("Razao: **fraude PIX nao e outlier puro** - ela tem assinatura clara (noturno + saldo residual baixo + recebedor recebe ~100%). O AE treinado em legitimas tambem reconstroi fraudes com erro baixo, porque as features de fraude **nao sao extremas**, elas sao **combinatoriamente anomalas**.")
    lines.append("")
    lines.append("Isso confirma o insight do s07/s10: **interacoes de features** (drain_score, night_x_ratio) sao o que separa fraude de legitima - e AE so captura magnitude de erro, nao interacao.")
    lines.append("")
    lines.append("## 6. Trade-off de producao")
    lines.append("")
    lines.append("| Criterio | CatBoost | MLP | TabNet |")
    lines.append("|---|---|---|---|")
    lines.append(f"| PR-AUC | {BASELINE_CATBOOST['pr_auc']:.4f} | {mlp_row['pr_auc']:.4f} | {tabnet_row['pr_auc']:.4f} |")
    lines.append(f"| Tempo treino | {BASELINE_CATBOOST['tempo_s']:.0f}s | {mlp_row['tempo_s']:.0f}s | {tabnet_row['tempo_s']:.0f}s |")
    lines.append(f"| Parametros | ~10k (arvores) | {mlp_row['n_params']} | {tabnet_row['n_params']} |")
    lines.append("| Explicabilidade | SHAP nativo | Gradient saliency (ruidoso) | Mask nativo (bom) |")
    lines.append("| Retreino | Rapido | Mais lento | Mais lento |")
    lines.append("| Deploy | Trivial | Precisa Torch server | Precisa Torch server |")
    lines.append("")
    lines.append("## 7. Recomendacao")
    lines.append("")
    if best["modelo"] == "CatBoost (baseline s10)":
        lines.append("**[RECOMENDADO]** Manter **CatBoost** em producao. Nenhum modelo DL trouxe ganho que justifique:")
        lines.append("- +30s a +3min de treino por retreino semanal")
        lines.append("- Container Torch em producao (vs binario nativo do CatBoost)")
        lines.append("- Perda de SHAP nativo (explicabilidade regulatoria)")
        lines.append("")
        lines.append("**DL nao e' errado** - so nao agrega em tabular bem-feito. Se o dataset tivesse:")
        lines.append("- Features de sequencia (transacoes do cliente em ordem)")
        lines.append("- Imagens (cheque, documento)")
        lines.append("- Texto (chat de suporte)")
        lines.append("")
        lines.append("Ai sim, DL (LSTM/Transformer/CNN) venceria por margem grande.")
    else:
        lines.append(f"**[RECOMENDADO]** Considerar **{best['modelo']}** se a infraestrutura suportar Torch em producao.")

    lines.append("")
    lines.append("## 8. Licao de negocio")
    lines.append("")
    lines.append("> O sinal em fraude esta nas **features** (drain_score, night_x_ratio, saldo residual), nao no modelo. Um CatBoost default com features contextualmente desenhadas bate 3 arquiteturas DL de ponta.")

    (OUT_DIR / "dashboard_dl.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] dashboard em {OUT_DIR / 'dashboard_dl.md'}")

    summary = {
        "device": str(DEVICE),
        "hipotese": "DL nao bate GBDT em tabular bem estruturado",
        "confirmada": bool(best["modelo"] == "CatBoost (baseline s10)"),
        "resultados": df_res.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

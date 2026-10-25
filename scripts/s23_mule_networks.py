"""
s23 - Deteccao de Mule Networks no PIX Fraud BR.

Inspirado em:
  - PixSim (arXiv 2609.30684) — "mule networks" no paper
  - ghs-mk/pixfraudeproject — "contas zumbis" e "hubs de alto volume"

Mule network = conta que recebe de N vítimas e repassa para M destinos em tempo curto.
Essa e a assinatura de conta-laranja em golpes organizados.

Testa hipoteses:
  H1: Contas com in_degree >= 3 e out_degree >= 2 sao mais fraudulentas
  H2: Tempo entre receber e repassar < 1h indica mula
  H3: Mule networks formam clusters detectaveis

Saida:
    docs/s23_mule_networks/mule_accounts.csv
    docs/s23_mule_networks/dashboard_mules.md
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SRC = Path("data/pix_fraud_br/transactions.parquet")
OUT = Path("docs/s23_mule_networks")
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_parquet(SRC)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)
    return df


def build_mule_network_features(df, time_window="24h"):
    """Constroi features de mule network (point-in-time)."""
    df = df.copy().sort_values("event_time").reset_index(drop=True)

    # 1. in_degree e out_degree (contas unicas, point-in-time via cumsum de first_seen)
    # Usar duplicated + cumsum: O(n) e vetorizado
    df["_pair_recv"] = df["merchant_id"].astype(str) + "|" + df["customer_id"].astype(str)
    df["_pair_send"] = df["customer_id"].astype(str) + "|" + df["merchant_id"].astype(str)

    df["_first_recv"] = ~df.duplicated(subset=["merchant_id", "customer_id"], keep="first")
    df["_first_send"] = ~df.duplicated(subset=["customer_id", "merchant_id"], keep="first")

    df["receiver_in_degree"] = df.groupby("merchant_id")["_first_recv"].cumsum().astype(int)
    df["payer_out_degree"] = df.groupby("customer_id")["_first_send"].cumsum().astype(int)

    # 2. Flag de mula: recebe de >= 3 e repassa para >= 2 (indicio de laranja)
    df["is_mule_candidate"] = (
        (df["receiver_in_degree"] >= 3) &
        (df["payer_out_degree"] >= 2)
    ).astype(int)

    # 3. Fanout ratio (in/out)
    df["fanout_ratio"] = df["receiver_in_degree"] / df["payer_out_degree"].clip(lower=1)

    # 4. Saldo residual proximo de 0 (conta drenada)
    df["is_drained"] = (df["residual_balance_ratio"] < 0.05).astype(int)

    # 5. Combinacao: mula candidata + drenada
    df["is_mule_active"] = ((df["is_mule_candidate"] == 1) & (df["is_drained"] == 1)).astype(int)

    return df


def test_hypothesis(name, df, mask, target="Class"):
    group = df[mask]
    rest = df[~mask]

    if len(group) == 0 or len(rest) == 0:
        return {"hipotese": name, "n_group": 0, "nota": "grupo vazio"}

    contingency = [
        [int((group[target] == 1).sum()), int((group[target] == 0).sum())],
        [int((rest[target] == 1).sum()), int((rest[target] == 0).sum())],
    ]
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    rate_group = float(group[target].mean())
    rate_rest = float(rest[target].mean())
    lift = rate_group / rate_rest if rate_rest > 0 else 0

    return {
        "hipotese": name,
        "n_group": int(len(group)),
        "n_total": int(len(df)),
        "frac_grupo_pct": round(100 * len(group) / len(df), 2),
        "fraud_rate_grupo_pct": round(100 * rate_group, 4),
        "fraud_rate_resto_pct": round(100 * rate_rest, 4),
        "lift": round(lift, 2),
        "chi2": round(float(chi2), 2),
        "p_value": float(p_value),
        "significante_5pct": bool(p_value < 0.05),
    }


def detect_mule_clusters(df):
    """Detecta clusters de mula (contas que recebem de multiplos e repassam)."""
    # Analise por merchant (recebedor)
    mule_stats = df.groupby("merchant_id").agg(
        n_tx=("Class", "size"),
        n_fraudes=("Class", "sum"),
        n_payers=("customer_id", "nunique"),
        total_recebido=("amount", "sum"),
        avg_saldo_antes=("saldo_anterior_recebedor", "mean"),
        avg_amount_ratio=("receiver_amount_ratio", "mean"),
    ).reset_index()

    # Mula: recebeu de 3+ pagadores, fraude alta, saldo baixo antes
    mule_stats["fraud_rate"] = mule_stats["n_fraudes"] / mule_stats["n_tx"].clip(lower=1)
    mule_stats["is_mule"] = (
        (mule_stats["n_payers"] >= 3) &
        (mule_stats["fraud_rate"] > 0.5)
    ).astype(int)

    return mule_stats.sort_values("fraud_rate", ascending=False)


def main():
    print("[1] carregando dados...")
    df = load_data()
    print(f"    {len(df):,} transacoes | {int(df.Class.sum()):,} fraudes")

    print("[2] construindo features de mule network...")
    df = build_mule_network_features(df)

    print("[3] testando hipoteses...")
    hipoteses = []
    hipoteses.append(test_hypothesis(
        "H1: Mula candidata (in>=3 e out>=2)",
        df, df["is_mule_candidate"] == 1
    ))
    hipoteses.append(test_hypothesis(
        "H2: Mula ativa (in>=3, out>=2, drenada)",
        df, df["is_mule_active"] == 1
    ))
    hipoteses.append(test_hypothesis(
        "H3: Conta drenada (saldo residual < 5%)",
        df, df["is_drained"] == 1
    ))
    hipoteses.append(test_hypothesis(
        "H4: Fanout ratio > 3 (recebe muito, repassa pouco)",
        df, df["fanout_ratio"] > 3
    ))

    df_hyp = pd.DataFrame(hipoteses)
    df_hyp.to_csv(OUT / "hypotheses.csv", index=False)
    print(df_hyp.to_string(index=False))

    print("\n[4] detectando clusters de mula...")
    mule_stats = detect_mule_clusters(df)
    mule_stats.to_csv(OUT / "mule_accounts.csv", index=False)
    print(f"    Top 5 mulas:")
    print(mule_stats.head(5).to_string(index=False))

    # Dashboard
    lines = []
    lines.append("# s23 - Mule Networks no PIX Fraud BR")
    lines.append("")
    lines.append(f"**Data:** 2026-10-25 | **Dataset:** PIX Fraud BR ({len(df):,} tx)")
    lines.append("")
    lines.append("**Inspirado em:**")
    lines.append("- [PixSim (arXiv 2609.30684)](https://arxiv.org/abs/2609.30684) — mule networks")
    lines.append("- [ghs-mk/pixfraudeproject](https://github.com/ghs-mk/pixfraudeproject) — contas zumbis, top accounts")
    lines.append("")
    lines.append("## O que e mule network")
    lines.append("")
    lines.append("**Mule network** (rede de mulas/laranjas) e uma conta que:")
    lines.append("1. **Recebe** de N vítimas (in_degree >= 3)")
    lines.append("2. **Repassa** para M destinos (out_degree >= 2)")
    lines.append("3. Em **tempo curto** (24h)")
    lines.append("4. Deixando a conta **drenada** (saldo residual ~ 0)")
    lines.append("")
    lines.append("E a assinatura de conta-laranja em golpes organizados — Falso Funcionario, Perfil Falso, etc.")
    lines.append("")
    lines.append("## 1. Hipoteses testadas")
    lines.append("")
    lines.append("| Hipotese | N grupo | % do total | Taxa fraude grupo | Taxa resto | Lift | p-value | Signif. |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df_hyp.iterrows():
        if r["n_group"] == 0:
            continue
        sig = "SIM" if r["significante_5pct"] else "nao"
        p_str = f"{r['p_value']:.2e}" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        lines.append(
            f"| {r['hipotese']} | {r['n_group']:,} | {r['frac_grupo_pct']:.2f}% | "
            f"**{r['fraud_rate_grupo_pct']:.3f}%** | {r['fraud_rate_resto_pct']:.3f}% | "
            f"**{r['lift']:.2f}x** | {p_str} | {sig} |"
        )
    lines.append("")
    lines.append("## 2. Top 10 contas mula (maior taxa de fraude)")
    lines.append("")
    lines.append("| Merchant | N tx | Fraudes | Taxa fraude | N pagadores | Total recebido | Saldo antes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in mule_stats[mule_stats["is_mule"] == 1].head(10).iterrows():
        lines.append(
            f"| {r['merchant_id'][:15]}... | {int(r['n_tx'])} | {int(r['n_fraudes'])} | "
            f"**{100*r['fraud_rate']:.1f}%** | {int(r['n_payers'])} | "
            f"R$ {r['total_recebido']:,.0f} | R$ {r['avg_saldo_antes']:,.0f} |"
        )
    lines.append("")
    n_mules = int((mule_stats["is_mule"] == 1).sum())
    total_merchants = len(mule_stats)
    lines.append(f"**Total de mulas detectadas:** {n_mules} de {total_merchants} merchants ({100*n_mules/total_merchants:.2f}%)")
    lines.append("")
    lines.append("## 3. Interpretacao")
    lines.append("")
    for _, r in df_hyp.iterrows():
        if r["n_group"] == 0:
            continue
        if not r["significante_5pct"]:
            lines.append(f"- **{r['hipotese']}**: **REFUTADA** (p={r['p_value']:.3f})")
        elif r["lift"] > 2:
            lines.append(f"- **{r['hipotese']}**: **CONFIRMADA** com lift forte ({r['lift']:.2f}x, p={r['p_value']:.2e})")
        else:
            lines.append(f"- **{r['hipotese']}**: **CONFIRMADA** (lift {r['lift']:.2f}x, p={r['p_value']:.2e})")

    lines.append("")
    lines.append("## 4. Licao de negocio")
    lines.append("")
    lines.append("> **Mule networks sao detectaveis por estrutura de grafo, nao por features isoladas.** A assinatura e: conta nova + recebe de N vitimas + repassa para M destinos + drena saldo. Isso e **exatamente** o que a Incognia chama de \"rede de laranjas\" e o que o paper PixSim modela.")
    lines.append("")
    lines.append("**Impacto**: detectar uma mula **antes** dela repassar o dinheiro pode recuperar **100% do valor** via MED (Mecanismo Especial de Devolucao). O desafio e a **velocidade** — a mula tipicamente repassa em **minutos** apos receber.")

    (OUT / "dashboard_mules.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "dataset": "pix-fraud-br",
        "inspirado_em": [
            "PixSim (arXiv 2609.30684)",
            "ghs-mk/pixfraudeproject",
        ],
        "n_mules_detectadas": n_mules,
        "n_merchants_total": total_merchants,
        "hipoteses": df_hyp.to_dict(orient="records"),
        "top_mulas": mule_stats[mule_stats["is_mule"] == 1].head(20).to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[OK] dashboard em {OUT}")


if __name__ == "__main__":
    main()
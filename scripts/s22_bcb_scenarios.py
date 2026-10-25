"""
s22 - Cenarios de golpe BCB no PIX Fraud BR.

Inspirado em:
  - ghs-mk/pixfraudeproject (contas zumbis, top accounts)
  - MatheusFran/pix-detection-fraud (decomposicao temporal)
  - Paper PixSim (cenarios: Falso Funcionario, Perfil Falso)

Saida:
    docs/s22_bcb_scenarios/hypotheses.csv
    docs/s22_bcb_scenarios/dashboard_scenarios.md
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
OUT = Path("docs/s22_bcb_scenarios")
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_parquet(SRC)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)
    return df


def build_scenario_features(df):
    df = df.copy()
    tx_per_payer = df.groupby("customer_id")["amount"].transform("count")
    df["is_zombie"] = (tx_per_payer == 1).astype(int)
    p95_amount = df["amount"].quantile(0.95)
    df["is_zombie_high_value"] = ((tx_per_payer == 1) & (df["amount"] >= p95_amount)).astype(int)
    receiver_volume = df.groupby("merchant_id")["amount"].transform("sum")
    p90_receiver = receiver_volume.quantile(0.90)
    df["is_high_volume_hub"] = (receiver_volume >= p90_receiver).astype(int)
    df["is_round_amount"] = ((df["amount"] >= 100) & (df["amount"] % 100 == 0)).astype(int)
    df["is_micro_amount"] = (df["amount"] < 50).astype(int)
    df["tx_number"] = df.groupby("customer_id").cumcount() + 1
    df["is_first_tx"] = (df["tx_number"] == 1).astype(int)
    df["hour"] = df["event_time"].dt.hour
    df_idx = df.set_index("event_time")
    df["payer_tx_1h"] = (
        df_idx.groupby("customer_id")["amount"]
              .rolling("1h").count()
              .reset_index(level=0, drop=True)
              .fillna(1).astype(int).values
    )
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


def main():
    print("[1] carregando dados...")
    df = load_data()
    print(f"    {len(df):,} transacoes | {int(df.Class.sum()):,} fraudes ({100*df.Class.mean():.3f}%)")

    print("[2] construindo features...")
    df = build_scenario_features(df)

    print("[3] testando hipoteses...")
    hipoteses = []
    hipoteses.append(test_hypothesis("H1: Conta zumbi (1 tx)", df, df["is_zombie"] == 1))
    hipoteses.append(test_hypothesis("H1b: Conta zumbi + alto valor (P95)", df, df["is_zombie_high_value"] == 1))
    hipoteses.append(test_hypothesis("H2: Recebedor top 10% volume", df, df["is_high_volume_hub"] == 1))
    hipoteses.append(test_hypothesis("H3: Valor redondo (multiplo de 100)", df, df["is_round_amount"] == 1))
    hipoteses.append(test_hypothesis("H4: Primeira tx da conta", df, df["is_first_tx"] == 1))
    hipoteses.append(test_hypothesis("H5: 3+ tx na ultima hora", df, df["payer_tx_1h"] >= 3))
    hipoteses.append(test_hypothesis("H6: Madrugada (0-4h)", df, df["hour"].isin([0, 1, 2, 3, 4])))

    df_hyp = pd.DataFrame(hipoteses)
    df_hyp.to_csv(OUT / "hypotheses.csv", index=False)
    print(df_hyp.to_string(index=False))

    print("\n[4] analise por cenario BCB...")
    cenarios = {}

    mask_falso_func = (
        (df["is_night"] == 1) &
        (df["residual_balance_ratio"] < 0.1) &
        (df["amount"] > df["amount"].quantile(0.7))
    )
    cenarios["Falso Funcionario"] = {
        "n_tx": int(mask_falso_func.sum()),
        "n_fraude": int(df.loc[mask_falso_func, "Class"].sum()),
        "fraud_rate_pct": round(100 * df.loc[mask_falso_func, "Class"].mean(), 3) if mask_falso_func.sum() > 0 else 0,
        "descricao": "Vitima transfere para terceiros acreditando ser o banco",
    }

    mask_perfil_falso = (df["is_zombie"] == 1) & (df["amount"] > 500)
    cenarios["Perfil Falso"] = {
        "n_tx": int(mask_perfil_falso.sum()),
        "n_fraude": int(df.loc[mask_perfil_falso, "Class"].sum()),
        "fraud_rate_pct": round(100 * df.loc[mask_perfil_falso, "Class"].mean(), 3) if mask_perfil_falso.sum() > 0 else 0,
        "descricao": "Vitima paga por produto inexistente (conta recem-criada)",
    }

    mask_mula = (df["saldo_anterior_recebedor"] < 500) & (df["receiver_amount_ratio"] > 0.95)
    cenarios["Mula (conta-laranja)"] = {
        "n_tx": int(mask_mula.sum()),
        "n_fraude": int(df.loc[mask_mula, "Class"].sum()),
        "fraud_rate_pct": round(100 * df.loc[mask_mula, "Class"].mean(), 3) if mask_mula.sum() > 0 else 0,
        "descricao": "Recebedor com saldo baixo antes (conta-laranja)",
    }

    print(json.dumps(cenarios, indent=2, ensure_ascii=False))

    # Dashboard
    lines = []
    lines.append("# s22 - Cenarios de Golpe BCB no PIX Fraud BR")
    lines.append("")
    lines.append(f"**Data:** 2026-10-25 | **Dataset:** PIX Fraud BR ({len(df):,} tx)")
    lines.append("")
    lines.append("**Inspirado em:**")
    lines.append("- [ghs-mk/pixfraudeproject](https://github.com/ghs-mk/pixfraudeproject) — contas zumbis, top accounts")
    lines.append("- [MatheusFran/pix-detection-fraud](https://github.com/MatheusFran/pix-detection-fraud) — decomposicao temporal")
    lines.append("- [PixSim (arXiv 2609.30684)](https://arxiv.org/abs/2609.30684) — cenarios: Falso Funcionario, Perfil Falso")
    lines.append("")
    lines.append("---")
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
        lines.append(f"| {r['hipotese']} | {r['n_group']:,} | {r['frac_grupo_pct']:.2f}% | **{r['fraud_rate_grupo_pct']:.3f}%** | {r['fraud_rate_resto_pct']:.3f}% | **{r['lift']:.2f}x** | {p_str} | {sig} |")
    lines.append("")
    lines.append("## 2. Cenarios BCB (mapeamento)")
    lines.append("")
    lines.append("| Cenario | N tx | N fraude | Taxa fraude | Descricao |")
    lines.append("|---|---:|---:|---:|---|")
    for nome, c in cenarios.items():
        lines.append(f"| **{nome}** | {c['n_tx']:,} | {c['n_fraude']:,} | **{c['fraud_rate_pct']:.3f}%** | {c['descricao']} |")
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
        elif r["lift"] > 1.2:
            lines.append(f"- **{r['hipotese']}**: **CONFIRMADA** com lift moderado ({r['lift']:.2f}x, p={r['p_value']:.2e})")
        else:
            lines.append(f"- **{r['hipotese']}**: **FRACA** (lift {r['lift']:.2f}x, p={r['p_value']:.2e})")

    lines.append("")
    lines.append("## 4. Licao de negocio")
    lines.append("")
    lines.append("> **Cada cenario de golpe tem uma assinatura propria.** Conta zumbi, hub de alto volume e valores redondos capturam partes diferentes do problema. O melhor sistema combina varias regras (cada uma captura um cenario) com o modelo ML (que aprende interacoes).")

    (OUT / "dashboard_scenarios.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "dataset": "pix-fraud-br",
        "inspirado_em": ["ghs-mk/pixfraudeproject", "MatheusFran/pix-detection-fraud", "PixSim arXiv 2609.30684"],
        "hipoteses": df_hyp.to_dict(orient="records"),
        "cenarios_bcb": cenarios,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[OK] dashboard em {OUT}")


if __name__ == "__main__":
    main()
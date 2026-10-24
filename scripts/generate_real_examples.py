"""
Gera grafico de exemplos reais (3 fraudes top-score com SHAP por feature).

Saida:
    docs/images/07_real_examples.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

SRC = "docs/s07_pix_fraud_br/shap_waterfall_top.csv"
OUT = Path("docs/images/07_real_examples.png")

df = pd.read_csv(SRC)
# Colunas esperadas: rank, feature, value, shap, proba, tx_index
ranks = sorted(df["rank"].unique())
n = min(3, len(ranks))

fig, axes = plt.subplots(1, n, figsize=(15, 6), sharex=False)

for i, rank in enumerate(ranks[:n]):
    ax = axes[i] if n > 1 else axes
    sub = df[df["rank"] == rank].sort_values("shap", key=abs, ascending=True)
    top = sub.tail(8)

    colors = ["#D62828" if s > 0 else "#0A66C2" for s in top["shap"]]
    bars = ax.barh(top["feature"], top["shap"], color=colors, edgecolor="white")

    proba = float(sub["proba"].iloc[0])
    ax.set_title(f"Fraude #{rank} (P={proba:.2%})", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribuicao SHAP (vermelho=aumenta risco)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)

    for b, s in zip(bars, top["shap"]):
        ax.text(s + (0.05 if s > 0 else -0.05), b.get_y() + b.get_height()/2,
                f"{s:+.2f}", va="center",
                ha="left" if s > 0 else "right", fontsize=8, fontweight="bold")

fig.suptitle("Exemplos reais — 3 fraudes com explicacao SHAP por feature",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.close()
print(f"[OK] {OUT}")

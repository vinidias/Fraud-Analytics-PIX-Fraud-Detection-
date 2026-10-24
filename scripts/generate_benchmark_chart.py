"""Gera grafico do cross-dataset benchmark."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 130

SRC = "docs/s18_benchmark/results.csv"
OUT = Path("docs/images/08_cross_dataset.png")

df = pd.read_csv(SRC)
df = df.sort_values("pr_auc", ascending=True)

# Adicionar PIX baseline
pix_baseline = {"dataset": "PIX Fraud BR\n(nosso)", "pr_auc": 0.9614}

fig, ax = plt.subplots(figsize=(11, 6))
colors = []
for _, r in df.iterrows():
    if r["pr_auc"] >= 0.95:
        colors.append("#2A9D8F")  # verde - excelente
    elif r["pr_auc"] >= 0.5:
        colors.append("#F4A261")  # laranja - medio
    else:
        colors.append("#D62828")  # vermelho - problematico

bars = ax.barh(df["dataset"], df["pr_auc"], color=colors, edgecolor="white", linewidth=1.2)
for b, v in zip(bars, df["pr_auc"]):
    ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.4f}",
            va="center", fontsize=10, fontweight="bold")

# Linha de referência PIX
ax.axvline(0.9614, color="#0A66C2", linestyle="--", linewidth=1.5, alpha=0.7,
           label="PIX Fraud BR (0,9614)")

ax.set_xlim(0, 1.1)
ax.set_xlabel("PR-AUC (0,5 = aleatório, 1,0 = perfeito)", fontsize=11)
ax.set_title("Cross-dataset benchmark — 5 datasets + PIX (mesmo pipeline CatBoost)",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(loc="lower right", fontsize=10)

# Anotações
ax.text(0.02, 0.98, "Verde: ≥0,95 (excelente)\nLaranja: 0,50-0,95 (médio)\nVermelho: <0,50 (precisa FE ou drift)",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.close()
print(f"[OK] {OUT}")
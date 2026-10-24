"""
Gera graficos-chave para o README.

Saida:
    docs/images/01_model_comparison.png
    docs/images/02_fraud_by_hour.png
    docs/images/03_shap_importance.png
    docs/images/04_roi_curve.png
    docs/images/05_walk_forward.png
    docs/images/06_optuna_trials.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 120

PRIMARY = "#0A66C2"
DANGER = "#D62828"
SUCCESS = "#2A9D8F"
GRAY = "#6C757D"


# ============================================================
# 1. Model comparison (bar) — todos os modelos testados
# ============================================================

def chart_model_comparison():
    models = {
        "AutoEncoder (anomalia)": 0.2280,
        "TabNet": 0.8753,
        "PyTorch MLP": 0.9206,
        "LightGBM (500k)": 0.9484,
        "LightGBM (2M)": 0.9553,
        "XGBoost": 0.9587,
        "Ensemble (stacking)": 0.9597,
        "CatBoost (500k)": 0.9603,
        "CatBoost (2M)": 0.9614,
    }
    names = list(models.keys())
    vals = list(models.values())

    colors = []
    for v in vals:
        if v >= 0.96:
            colors.append(SUCCESS)
        elif v >= 0.94:
            colors.append(PRIMARY)
        elif v >= 0.88:
            colors.append("#F4A261")
        else:
            colors.append(DANGER)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, vals):
        ax.text(v + 0.002, b.get_y() + b.get_height()/2, f"{v:.4f}",
                va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("PR-AUC (0.5 = aleatorio, 1.0 = perfeito)", fontsize=11)
    ax.set_title("Comparacao de 9 modelos — PIX Fraud Detection",
                 fontsize=13, fontweight="bold", pad=15)
    ax.axvline(0.96, color=DANGER, linestyle="--", alpha=0.5, label="Threshold 0.96")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "01_model_comparison.png", bbox_inches="tight")
    plt.close()
    print("[OK] 01_model_comparison.png")


# ============================================================
# 2. Fraude por hora do dia
# ============================================================

def chart_fraud_by_hour():
    eda = json.load(open("docs/s07_pix_fraud_br/eda_metrics.json"))
    by_hour = pd.DataFrame(eda["by_hour"])
    by_hour = by_hour.sort_values("hora_dia")

    fig, ax = plt.subplots(figsize=(11, 5))
    hours = by_hour["hora_dia"].values
    rates = (by_hour["fraud_rate"] * 100).values
    colors = [DANGER if h in [0, 1, 2, 3, 4, 5] else PRIMARY for h in hours]
    bars = ax.bar(hours, rates, color=colors, edgecolor="white", linewidth=1.2)

    for b, r in zip(bars, rates):
        if r > 5:
            ax.text(b.get_x() + b.get_width()/2, r + 0.5, f"{r:.0f}%",
                    ha="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Hora do dia", fontsize=11)
    ax.set_ylabel("Taxa de fraude (%)", fontsize=11)
    ax.set_title("Fraude concentra na madrugada — 46,8% as 4h vs 0,57% no diurno",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(range(24))
    ax.axvspan(-0.5, 5.5, color=DANGER, alpha=0.08, label="Janela noturna (0-5h)")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "02_fraud_by_hour.png", bbox_inches="tight")
    plt.close()
    print("[OK] 02_fraud_by_hour.png")


# ============================================================
# 3. SHAP importance
# ============================================================

def chart_shap():
    df = pd.read_csv("docs/s07_pix_fraud_br/shap_summary.csv").head(12)
    df = df.sort_values("pct_of_total", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [DANGER if "drain" in f or "night" in f else PRIMARY for f in df["feature"]]
    bars = ax.barh(df["feature"], df["pct_of_total"], color=colors, edgecolor="white")

    for b, v in zip(bars, df["pct_of_total"]):
        ax.text(v + 0.2, b.get_y() + b.get_height()/2, f"{v:.1f}%",
                va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("% do total de |SHAP|", fontsize=11)
    ax.set_title("Top 12 features — SHAP (importancia global)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.text(0.98, 0.02, "Vermelho = features de interacao (criadas)",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic", color=GRAY)

    plt.tight_layout()
    plt.savefig(OUT / "03_shap_importance.png", bbox_inches="tight")
    plt.close()
    print("[OK] 03_shap_importance.png")


# ============================================================
# 4. ROI curve by threshold
# ============================================================

def chart_roi():
    df = pd.read_csv("docs/s08_roi/roi_curves.csv")
    df = df[df["capacidade"] == "time_medio"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cen_colors = {"conservador": GRAY, "balanceado": PRIMARY, "agressivo": SUCCESS}

    for cen in ["conservador", "balanceado", "agressivo"]:
        sub = df[df["cenario"] == cen].sort_values("threshold")
        ax.plot(sub["threshold"], sub["economia_liquida_brl"] / 1000,
                label=cen.capitalize(), color=cen_colors[cen], linewidth=2.2)

    ax.set_xlabel("Threshold de classificacao", fontsize=11)
    ax.set_ylabel("Economia liquida (R$ mil)", fontsize=11)
    ax.set_title("Curva de ROI por threshold — 3 cenarios de custo (capacity=300/dia)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT / "04_roi_curve.png", bbox_inches="tight")
    plt.close()
    print("[OK] 04_roi_curve.png")


# ============================================================
# 5. Walk-forward
# ============================================================

def chart_walk_forward():
    df = pd.read_csv("docs/s09_walk_forward/windows.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for est, color in [("fixed", GRAY), ("expanding", PRIMARY)]:
        sub = df[df["estrategia"] == est].sort_values("janela")
        axes[0].plot(sub["janela"], sub["pr_auc"], marker="o", label=est.capitalize(),
                     color=color, linewidth=2.2, markersize=8)
        axes[1].plot(sub["janela"], sub["roi"], marker="s", label=est.capitalize(),
                     color=color, linewidth=2.2, markersize=8)

    axes[0].set_title("PR-AUC por janela temporal", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("PR-AUC")
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].set_title("ROI por janela temporal", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("ROI (x)")
    axes[1].legend(loc="lower right", fontsize=9)

    fig.suptitle("Walk-forward temporal — modelo NAO degrada em drift sintetico",
                 fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(OUT / "05_walk_forward.png", bbox_inches="tight")
    plt.close()
    print("[OK] 05_walk_forward.png")


# ============================================================
# 6. Optuna trials
# ============================================================

def chart_optuna():
    df = pd.read_csv("docs/s11_optuna/optuna_trials.csv")
    df = df[df["state"] == "COMPLETE"].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["number"], df["value"], c=df["value"], cmap="viridis",
               s=80, alpha=0.8, edgecolor="white")

    best_idx = df["value"].idxmax()
    best_val = df.loc[best_idx, "value"]
    ax.axhline(0.9598, color=DANGER, linestyle="--", label="Baseline CatBoost (0.9598)", linewidth=2)
    ax.axhline(best_val, color=SUCCESS, linestyle="--", label=f"Best Optuna ({best_val:.4f})", linewidth=2)

    ax.set_xlabel("Trial #", fontsize=11)
    ax.set_ylabel("PR-AUC (validation)", fontsize=11)
    ax.set_title("Optuna 50 trials — melhor trial PIOR que default",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "06_optuna_trials.png", bbox_inches="tight")
    plt.close()
    print("[OK] 06_optuna_trials.png")


if __name__ == "__main__":
    chart_model_comparison()
    chart_fraud_by_hour()
    chart_shap()
    chart_roi()
    chart_walk_forward()
    chart_optuna()
    print("\n[OK] 6 graficos gerados em", OUT)

"""
Gera tabelas como PNG para Medium (v2 — sem emojis, sem LaTeX).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("docs/medium_assets")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150


def render_table(data, out_name, title=None, col_widths=None, header_color="#0A66C2", highlight_row=None):
    """Renderiza uma tabela como PNG estilo Medium."""
    n_rows = len(data)
    n_cols = len(data[0]) if data else 0
    fig_height = 0.45 * n_rows + 0.6
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, loc="left")

    table = ax.table(
        cellText=data[1:],
        colLabels=data[0],
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Header
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    # Rows alternate
    for i in range(1, n_rows):
        for j in range(n_cols):
            cell = table[(i, j)]
            # Highlight row (0-indexed nos dados, então +1 na tabela)
            if highlight_row is not None and i == highlight_row:
                cell.set_facecolor("#E8F5E9")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#f8f8f8" if i % 2 == 0 else "white")
                cell.set_text_props(fontsize=10.5)
            cell.set_edgecolor("#e0e0e0")

    plt.tight_layout()
    plt.savefig(OUT / out_name, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    print(f"[OK] {out_name}")


# 1. Desafios técnicos (corrigido: R$ escapado)
render_table(
    [
        ["Desafio", "Por que é difícil"],
        ["Desbalanceamento extremo", "0,77% de fraude → acurácia engana"],
        ["Drift temporal", "Taxa cresce 20x ao longo do mês"],
        ["Custo assimétrico", "FN custa ~R\\$ 4.000; FP custa ~R\\$ 20"],
        ["Explicabilidade", "Regulação exige justificativa por transação"],
    ],
    "tabela_01_desafios.png",
    title="Os 4 desafios técnicos do PIX",
    col_widths=[0.35, 0.65],
)

# 2. Comparação de 9 modelos (sem emojis)
render_table(
    [
        ["Modelo", "PR-AUC"],
        ["CatBoost (2M) — MELHOR", "0,9614"],
        ["CatBoost (500k)", "0,9603"],
        ["Ensemble (stacking)", "0,9597"],
        ["XGBoost", "0,9587"],
        ["LightGBM", "0,9484"],
        ["MLP (focal loss)", "0,9206"],
        ["TabNet", "0,8753"],
        ["GNN (GraphSAGE)", "0,3928"],
        ["AutoEncoder", "0,2280"],
    ],
    "tabela_02_modelos.png",
    title="Comparação de 9 modelos — PR-AUC",
    col_widths=[0.7, 0.3],
    highlight_row=1,
)

# 3. Optuna (corrigido)
render_table(
    [
        ["Métrica", "Base", "Optuna", "Delta"],
        ["PR-AUC (teste)", "0,9598", "0,9533", "-0,64pp"],
        ["Brier", "0,0122", "0,0179", "+46%"],
    ],
    "tabela_03_optuna.png",
    title="Optuna: tuning piorou o modelo",
    col_widths=[0.4, 0.2, 0.2, 0.2],
    header_color="#D62828",
)

# 4. Cross-dataset (sem emoji)
render_table(
    [
        ["Dataset", "Tipo", "PR-AUC", "vs PIX"],
        ["MoMTSim", "Mobile money sintético", "0,9712", "+0,98pp"],
        ["Xente (REAL)", "Mobile money REAL", "0,9672", "+0,58pp"],
        ["PIX Fraud BR", "PIX sintético", "0,9614", "baseline"],
        ["IEEE-CIS", "Cartão real", "0,5067", "-45,47pp"],
        ["BAF", "Bank account sintético", "0,1689", "-79,25pp"],
    ],
    "tabela_04_cross_dataset.png",
    title="Cross-dataset benchmark — 5 datasets + PIX",
    col_widths=[0.25, 0.35, 0.2, 0.2],
    header_color="#2A9D8F",
    highlight_row=2,
)

# 5. Isotonic
render_table(
    [
        ["Métrica", "Base", "Isotonic", "Delta"],
        ["PR-AUC", "0,9598", "0,9545", "-0,53pp"],
        ["Brier", "0,01174", "0,01215", "+3,5%"],
    ],
    "tabela_05_isotonic.png",
    title="Calibração isotônica: piorou tudo",
    col_widths=[0.4, 0.2, 0.2, 0.2],
    header_color="#D62828",
)

print(f"\n[OK] 5 tabelas geradas (v2) em {OUT}")

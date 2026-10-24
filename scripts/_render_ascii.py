"""
Gera tabelas em ASCII art (para colar em code block do Medium).
"""
from pathlib import Path

OUT = Path("docs/medium_assets")
OUT.mkdir(parents=True, exist_ok=True)


def ascii_table(headers, rows, title=None, highlight_rows=None):
    """Gera tabela ASCII com alinhamento perfeito."""
    highlight_rows = highlight_rows or []
    n_cols = len(headers)

    # Calcular larguras
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    widths = [w + 2 for w in widths]  # padding

    def sep(char="-", corner="+"):
        return corner + corner.join(char * w for w in widths) + corner

    def fmt_row(row, bold=False):
        cells = []
        for i, (w, cell) in enumerate(zip(widths, row)):
            s = str(cell)
            # Alinhamento: primeira coluna à esquerda, resto também
            cells.append(" " + s.ljust(w - 2) + " ")
        return "|" + "|".join(cells) + "|"

    lines = []
    if title:
        lines.append("### " + title)
        lines.append("")
    lines.append("+" + "+".join("-" * w for w in widths) + "+")
    lines.append(fmt_row(headers))
    lines.append("+" + "+".join("=" * w for w in widths) + "+")
    for i, row in enumerate(rows):
        lines.append(fmt_row(row))
    lines.append("+" + "+".join("-" * w for w in widths) + "+")
    return "\n".join(lines)


# 1. Desafios
t1 = ascii_table(
    ["Desafio", "Por que é difícil"],
    [
        ["Desbalanceamento extremo", "0,77% de fraude -> acurácia engana"],
        ["Drift temporal", "Taxa cresce 20x ao longo do mês"],
        ["Custo assimétrico", "FN custa ~R$ 4.000; FP custa ~R$ 20"],
        ["Explicabilidade", "Regulação exige justificativa por transação"],
    ],
    title="Os 4 desafios técnicos do PIX",
)

# 2. Modelos
t2 = ascii_table(
    ["Modelo", "PR-AUC"],
    [
        ["CatBoost (2M) -- MELHOR", "0,9614"],
        ["CatBoost (500k)", "0,9603"],
        ["Ensemble (stacking)", "0,9597"],
        ["XGBoost", "0,9587"],
        ["LightGBM", "0,9484"],
        ["MLP (focal loss)", "0,9206"],
        ["TabNet", "0,8753"],
        ["GNN (GraphSAGE)", "0,3928"],
        ["AutoEncoder", "0,2280"],
    ],
    title="Comparação de 9 modelos -- PR-AUC",
)

# 3. Optuna
t3 = ascii_table(
    ["Métrica", "Base", "Optuna", "Delta"],
    [
        ["PR-AUC (teste)", "0,9598", "0,9533", "-0,64pp"],
        ["Brier", "0,0122", "0,0179", "+46%"],
    ],
    title="Optuna: tuning piorou o modelo",
)

# 4. Cross-dataset
t4 = ascii_table(
    ["Dataset", "Tipo", "PR-AUC", "vs PIX"],
    [
        ["MoMTSim", "Mobile money sintético", "0,9712", "+0,98pp"],
        ["Xente (REAL)", "Mobile money REAL", "0,9672", "+0,58pp"],
        ["PIX Fraud BR", "PIX sintético", "0,9614", "baseline"],
        ["IEEE-CIS", "Cartão real", "0,5067", "-45,47pp"],
        ["BAF", "Bank account sintético", "0,1689", "-79,25pp"],
    ],
    title="Cross-dataset benchmark -- 5 datasets + PIX",
)

# 5. Isotonic
t5 = ascii_table(
    ["Métrica", "Base", "Isotonic", "Delta"],
    [
        ["PR-AUC", "0,9598", "0,9545", "-0,53pp"],
        ["Brier", "0,01174", "0,01215", "+3,5%"],
    ],
    title="Calibração isotônica: piorou tudo",
)

all_tables = {
    "tabela_01_desafios": t1,
    "tabela_02_modelos": t2,
    "tabela_03_optuna": t3,
    "tabela_04_cross_dataset": t4,
    "tabela_05_isotonic": t5,
}

# Salvar tudo num único arquivo
output = []
output.append("# Tabelas ASCII para Medium\n")
output.append("Cole cada bloco dentro de um **code block** (bloco de código) no Medium.\n")
output.append("---\n")
for name, content in all_tables.items():
    output.append(f"## {name}\n")
    output.append("```")
    output.append(content)
    output.append("```\n")
    output.append("---\n")

(OUT / "TABELAS_ASCII.md").write_text("\n".join(output), encoding="utf-8")

# Imprimir para conferir
for name, content in all_tables.items():
    print(f"\n=== {name} ===")
    print(content)

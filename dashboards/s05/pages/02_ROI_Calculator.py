"""
Simulador de ROI interativo — pagina Streamlit.

Uso: streamlit run dashboards/s05/pages/02_ROI_Calculator.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Simulador de ROI", page_icon="ROI", layout="wide")
st.title("Simulador de ROI Operacional")
st.caption("Calibre o threshold ideal para sua operacao. Baseado em s08 (ROI simulator).")

st.markdown("---")

st.header("1. Parametros da operacao")
col1, col2, col3 = st.columns(3)

with col1:
    custo_revisao = st.number_input(
        "Custo por revisao manual (R$)",
        min_value=1.0, max_value=500.0, value=15.0, step=5.0,
        help="Quanto custa revisar manualmente uma transacao flagrada",
    )
    custo_friccao = st.number_input(
        "Custo de friccao ao cliente (R$)",
        min_value=1.0, max_value=1000.0, value=20.0, step=5.0,
        help="Custo quando um legitimo e bloqueado (NPS, churn, suporte)",
    )

with col2:
    recovery_rate = st.slider(
        "Taxa de recuperacao (%)",
        0, 100, 85,
        help="% do valor da fraude recuperado apos deteccao",
    ) / 100
    capacidade = st.number_input(
        "Capacidade diaria de revisao",
        min_value=10, max_value=5000, value=300, step=50,
        help="Quantas transacoes o time consegue revisar por dia",
    )

with col3:
    dias_teste = st.number_input("Periodo de teste (dias)", min_value=1, max_value=90, value=10)
    valor_medio_fraude = st.number_input(
        "Valor medio da fraude (R$)",
        min_value=50.0, max_value=50000.0, value=300.0, step=50.0,
    )

st.markdown("---")

st.header("2. Curva de ROI por threshold")

# Simular curva de ROI sintetica (calibrada pelos resultados de s08)
thresholds = np.linspace(0.05, 0.95, 50)

# Modelo logico: recall cai com threshold, precision sobe
recall = 0.95 * np.exp(-3 * thresholds) + 0.05
precision = 0.4 + 0.5 * thresholds
n_total = int(capacidade * dias_teste * 1.5)  # assume volume ~1.5x capacidade
n_flag = (n_total * (1 - thresholds) * 0.05).clip(min=1, max=int(capacidade * dias_teste))
tp = n_flag * recall
fp = n_flag * (1 - precision)
fn = (n_total * 0.01) - tp  # assume taxa de fraude ~1%

valor_evitado = tp * valor_medio_fraude * recovery_rate
valor_perdido = fn * valor_medio_fraude * (1 - recovery_rate)
custo_total = n_flag * custo_revisao + fp * custo_friccao
economia = valor_evitado - valor_perdido - custo_total
roi = economia / np.maximum(custo_total, 1)

df_roi = pd.DataFrame({
    "threshold": thresholds,
    "recall": recall,
    "precision": precision,
    "n_flagged": n_flag,
    "economia_brl": economia,
    "roi": roi,
})

best_idx = df_roi["economia_brl"].idxmax()
best = df_roi.loc[best_idx]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Threshold otimo", f"{best['threshold']:.3f}")
c2.metric("Economia liquida", f"R$ {best['economia_brl']:,.0f}")
c3.metric("ROI", f"{best['roi']:.2f}x")
c4.metric("Transacoes flagradas", f"{int(best['n_flagged']):,}")

fig = px.line(df_roi, x="threshold", y="economia_brl",
              title="Economia liquida por threshold",
              labels={"economia_brl": "Economia (R$)", "threshold": "Threshold"})
fig.add_vline(x=best["threshold"], line_dash="dash", line_color="red",
              annotation_text=f"Otimo: {best['threshold']:.3f}")
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.line(df_roi, x="threshold", y=["precision", "recall"],
                  title="Precision e Recall por threshold")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.line(df_roi, x="threshold", y="roi",
                  title="ROI por threshold",
                  labels={"roi": "ROI (x)"})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("3. Recomendacao operacional")
st.success(f"""
**Com sua operacao atual:**
- Threshold sugerido: **{best['threshold']:.3f}**
- Transacoes/dia para revisar: **{int(best['n_flagged'] / dias_teste):,}**
- Recall: **{best['recall']:.1%}** · Precision: **{best['precision']:.1%}**
- Economia liquida no periodo: **R$ {best['economia_brl']:,.0f}**
- ROI: **{best['roi']:.2f}x**
""")

st.caption("Modelo simplificado para demonstracao. Para calculo exato, use `scripts/roi_simulator.py` com dados reais.")

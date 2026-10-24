"""
Feature Explorer — pagina Streamlit.

SHAP global + drill-down por transacao.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Feature Explorer", page_icon="?", layout="wide")
st.title("Feature Explorer (SHAP)")
st.caption("O que o modelo aprendeu? Importancia global + explicacoes por transacao.")

st.markdown("---")

SHAP_PATH = Path("docs/s07_pix_fraud_br/shap_summary.csv")
SHAP_WATERFALL = Path("docs/s07_pix_fraud_br/shap_waterfall_top.csv")

if SHAP_PATH.exists():
    df_shap = pd.read_csv(SHAP_PATH)
    df_shap = df_shap.sort_values("pct_of_total", ascending=True)

    st.header("Importancia global das features")
    fig = px.barh(
        df_shap, x="pct_of_total", y="feature",
        title="Top features por % do |SHAP| total",
        labels={"pct_of_total": "% do total", "feature": "Feature"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Insight**: `drain_score` (feature de interacao que criamos) entra no **top 3**, 
    validando que o modelo aprendeu a combinacao noturno x drenagem x proporcao recebedor.
    """)
else:
    st.warning("Rode `python scripts/shap_explain.py` para gerar os dados.")

st.markdown("---")

if SHAP_WATERFALL.exists():
    st.header("Explicacao por transacao (fraudes top-score)")
    df_wf = pd.read_csv(SHAP_WATERFALL)
    ranks = sorted(df_wf["rank"].unique())

    selected_rank = st.selectbox("Selecione uma fraude", ranks)
    sub = df_wf[df_wf["rank"] == selected_rank].sort_values("shap", key=abs, ascending=True)

    proba = float(sub["proba"].iloc[0])
    st.metric("Probabilidade de fraude", f"{proba:.2%}")

    colors = ["red" if s > 0 else "blue" for s in sub["shap"]]
    fig = px.bar(
        sub, x="shap", y="feature", orientation="h",
        title=f"Contribuicoes SHAP — Fraude #{selected_rank}",
        labels={"shap": "Contribuicao SHAP", "feature": "Feature"},
    )
    fig.update_traces(marker_color=colors)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sub[["feature", "value", "shap"]], use_container_width=True)
else:
    st.warning("Rode `python scripts/shap_explain.py` para gerar os dados.")

st.markdown("---")
st.caption("Pipeline: CatBoost (s10) -> SHAP (s07). Metricas: docs/s07_pix_fraud_br/RESULTADOS.md")

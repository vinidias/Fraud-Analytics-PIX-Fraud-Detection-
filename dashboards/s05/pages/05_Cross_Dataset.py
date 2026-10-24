"""
Cross-Dataset Benchmark — comparação de 5 datasets.

Mostra o PR-AUC do mesmo pipeline em diferentes dominios de fraude.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Cross-Dataset", layout="wide")
st.title("Cross-Dataset Benchmark")
st.caption("Mesmo pipeline CatBoost + split temporal em 5 datasets publicos")

RESULTS = Path("docs/s18_benchmark/results.csv")

if not RESULTS.exists():
    st.warning("Rode `python scripts/s18_cross_dataset_benchmark.py` primeiro.")
    st.stop()

df = pd.read_csv(RESULTS)
df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)

# Adicionar PIX
pix = pd.DataFrame([{"dataset": "PIX Fraud BR", "pr_auc": 0.9614, "n_features": 16,
                     "n_train": 475246, "n_test": 24754, "fraud_rate_test_pct": 5.04}])
df_full = pd.concat([pix, df], ignore_index=True)

# KPIs
st.subheader("Resumo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Datasets testados", len(df_full))
c2.metric("Melhor PR-AUC", f"{df_full.pr_auc.max():.4f}", df_full.loc[df_full.pr_auc.idxmax(), "dataset"])
c3.metric("PIX (baseline)", "0,9614")
c4.metric("Pipeline funciona em", "mobile money", "Xente + MoMTSim superam PIX")

st.divider()

# Grafico principal
st.subheader("PR-AUC por dataset")
df_full["cor"] = df_full["pr_auc"].apply(lambda x: "Excelente" if x >= 0.95 else ("Medio" if x >= 0.5 else "Critico"))
df_full["tipo"] = df_full["dataset"].apply(lambda x: "nosso" if x == "PIX Fraud BR" else "externo")

fig = px.bar(
    df_full, x="pr_auc", y="dataset", color="cor", orientation="h",
    color_discrete_map={"Excelente": "#2A9D8F", "Medio": "#F4A261", "Critico": "#D62828"},
    title="PR-AUC — pipeline CatBoost em 6 cenarios",
    labels={"pr_auc": "PR-AUC", "dataset": ""},
    text="pr_auc",
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig.add_vline(x=0.9614, line_dash="dash", line_color="#0A66C2",
              annotation_text="PIX baseline")
fig.update_layout(height=500, xaxis_range=[0, 1.1])
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Tabela detalhada
st.subheader("Detalhes")
df_show = df_full[["dataset", "pr_auc", "n_features", "n_train", "n_test", "fraud_rate_test_pct"]].copy()
df_show["pr_auc"] = df_show["pr_auc"].round(4)
df_show["fraud_rate_test_pct"] = df_show["fraud_rate_test_pct"].round(3)
st.dataframe(df_show, use_container_width=True, hide_index=True)

st.divider()

# Interpretacao
st.subheader("O que aprendemos")
st.success(
    "**Pipeline generaliza para mobile money**: Xente (0,9672) e MoMTSim (0,9712) superam o PIX — mesmo com dominios diferentes. Valida que a arquitetura nao e overfit."
)
st.warning(
    "**IEEE-CIS (0,5067) exige feature engineering especifica**: IDs tratados como features, 14 grupos V, TransactionAmt com range enorme. Gap de FE, nao de metodo."
)
st.error(
    "**BAF (0,1689) tem drift adversarial**: fraude muda ao longo do tempo. 12x melhor que random, mas longe do PIX. Precisa walk-forward retraining."
)

st.caption("Pipeline: `python scripts/s18_cross_dataset_benchmark.py` | Docs: `docs/s18_benchmark/`")
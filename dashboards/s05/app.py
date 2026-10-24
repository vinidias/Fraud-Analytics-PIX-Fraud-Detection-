"""
Semana 7 — Dashboard de monitoramento de fraude (Streamlit).

Uso:
    streamlit run dashboards/s05/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Fraud Monitor — Incognia prep", page_icon="🛡️", layout="wide")

st.title("🛡️ Fraud Monitor — Dashboard de Análise")
st.caption("Estudo para Fraud Data Analyst @ Incognia · dados sintéticos/exemplo")

# ---------- Carregar dados ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    p = Path("data/features.parquet")
    if p.exists():
        return pd.read_parquet(p)
    # fallback: dados sintéticos para demonstração
    rng = np.random.default_rng(42)
    n = 5000
    return pd.DataFrame({
        "amount": rng.lognormal(mean=4, sigma=1, size=n),
        "hour": rng.integers(0, 24, size=n),
        "Class": (rng.random(n) < 0.02).astype(int),
        "customer_id": rng.integers(0, 500, size=n),
    })

df = load_data()

# ---------- Sidebar ----------
st.sidebar.header("Filtros")
hours = st.sidebar.slider("Horário", 0, 23, (0, 23))
amount_range = st.sidebar.slider(
    "Valor (R$)",
    float(df["amount"].min()),
    float(df["amount"].quantile(0.99)),
    (float(df["amount"].min()), float(df["amount"].quantile(0.99))),
)
mask = df["hour"].between(*hours) & df["amount"].between(*amount_range)
dff = df[mask]

# ---------- KPIs ----------
n_tx = len(dff)
n_fraud = int(dff["Class"].sum())
fraud_rate = n_fraud / max(n_tx, 1) * 100
ticket = dff.loc[dff.Class == 1, "amount"].mean() if n_fraud else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transações", f"{n_tx:,}")
c2.metric("Fraudes", f"{n_fraud:,}")
c3.metric("Taxa de fraude", f"{fraud_rate:.2f}%")
c4.metric("Ticket médio (fraude)", f"R$ {ticket:.2f}")

# ---------- Gráficos ----------
st.subheader("Distribuição temporal")
hourly = dff.groupby("hour").agg(n_tx=("amount", "size"), n_fraud=("Class", "sum")).reset_index()
fig1 = px.line(hourly, x="hour", y=["n_tx", "n_fraud"], markers=True,
               title="Transações vs. Fraudes por hora")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Valor por classe")
fig2 = px.box(dff, x="Class", y="amount", color="Class",
              title="Distribuição de Amount (0=legítima, 1=fraude)",
              log_y=True)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top clientes suspeitos")
top = (
    dff[dff.Class == 1]
    .groupby("customer_id")
    .agg(n_fraude=("Class", "size"), valor_total=("amount", "sum"))
    .sort_values("valor_total", ascending=False)
    .head(20)
    .reset_index()
)
st.dataframe(top, use_container_width=True)

st.info("💡 Próximo passo: adicionar threshold operacional, matriz de custo e simulação de revisão manual.")

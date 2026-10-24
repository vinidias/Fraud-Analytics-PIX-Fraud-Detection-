"""
Dashboard de monitoramento de fraude - Streamlit.

Compatível com Streamlit Community Cloud:
    - Se data/features.parquet existir, usa o dado real
    - Senão, gera dados sintéticos em memória (suficiente para demonstração)

Rodar local:
    streamlit run dashboards/s05/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Fraud Monitor — PIX",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Fraud Monitor — PIX")
st.caption(
    "Fraud Analytics · PIX · "
    "[github.com/vinic/fraud-analytics](https://github.com/vinic/fraud-analytics)"
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Carrega dados reais se existirem; senão gera sintético compatível."""
    candidates = [
        Path("data/pix_fraud_br/transactions.parquet"),
        Path("data/features.parquet"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            df["event_time"] = pd.to_datetime(df["event_time"])
            return df

    # Fallback: dataset sintético com padrões de fraude reais
    rng = np.random.default_rng(42)
    n = 5000
    is_night = rng.random(n) < 0.3
    is_fraud = (is_night & (rng.random(n) < 0.15)) | (rng.random(n) < 0.01)
    return pd.DataFrame({
        "event_time": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "amount": np.where(is_fraud,
                           rng.lognormal(6.5, 1.0, n),
                           rng.lognormal(5.5, 0.8, n)),
        "hour": pd.date_range("2024-01-01", periods=n, freq="15min").hour,
        "Class": is_fraud.astype(int),
        "customer_id": [f"user_{i}" for i in rng.integers(0, 500, n)],
        "residual_balance_ratio": np.where(is_fraud,
                                            rng.uniform(0, 0.1, n),
                                            rng.uniform(0.7, 1.0, n)),
        "receiver_amount_ratio": np.where(is_fraud,
                                           rng.uniform(0.95, 1.0, n),
                                           rng.uniform(0.85, 0.99, n)),
    })


df = load_data()
df["hour"] = df["event_time"].dt.hour


# ---------- Sidebar ----------
st.sidebar.header("Filtros")
hours = st.sidebar.slider("Horário", 0, 23, (0, 23))
amount_max = float(df["amount"].quantile(0.99))
amount_range = st.sidebar.slider("Valor (R$)", 0.0, amount_max, (0.0, amount_max))

mask = df["hour"].between(*hours) & df["amount"].between(*amount_range)
dff = df[mask]


# ---------- KPIs ----------
st.subheader("📊 KPIs do período")
n_tx = len(dff)
n_fraud = int(dff["Class"].sum())
fraud_rate = (n_fraud / max(n_tx, 1)) * 100
ticket_med = dff.loc[dff.Class == 1, "amount"].median() if n_fraud else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transações", f"{n_tx:,}")
c2.metric("Fraudes", f"{n_fraud:,}")
c3.metric("Taxa de fraude", f"{fraud_rate:.2f}%")
c4.metric("Ticket mediano (fraude)", f"R$ {ticket_med:,.0f}")


# ---------- Horário ----------
st.subheader("⏰ Distribuição por hora do dia")
hourly = (
    dff.groupby("hour")
    .agg(n_tx=("Class", "size"), n_fraud=("Class", "sum"))
    .reset_index()
)
hourly["fraud_rate_pct"] = (hourly["n_fraud"] / hourly["n_tx"]) * 100

tab1, tab2 = st.tabs(["Volume", "Taxa de fraude"])

with tab1:
    fig = px.bar(
        hourly, x="hour", y=["n_tx", "n_fraud"],
        barmode="overlay", title="Transações vs Fraudes por hora",
        labels={"value": "Transações", "hour": "Hora", "variable": "Tipo"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.line(
        hourly, x="hour", y="fraud_rate_pct", markers=True,
        title="Taxa de fraude por hora (%)",
        labels={"fraud_rate_pct": "Taxa (%)", "hour": "Hora"},
    )
    st.plotly_chart(fig, use_container_width=True)
    if hourly["fraud_rate_pct"].max() > 5:
        peak_hour = int(hourly.loc[hourly["fraud_rate_pct"].idxmax(), "hour"])
        st.warning(
            f"⚠️ Concentração de fraude às **{peak_hour}h** "
            f"({hourly['fraud_rate_pct'].max():.1f}% vs {hourly['fraud_rate_pct'].mean():.1f}% média). "
            "Recomendação: step-up auth para transações noturnas."
        )


# ---------- Valor ----------
st.subheader("💰 Distribuição de valor")
fig = px.box(
    dff, x="Class", y="amount", color="Class",
    title="Valor por classe (0=legítima, 1=fraude)",
    log_y=True,
    labels={"amount": "Valor (R$)", "Class": "Classe"},
)
st.plotly_chart(fig, use_container_width=True)


# ---------- Drenagem (se colunas existirem) ----------
if "residual_balance_ratio" in df.columns:
    st.subheader("💧 Drenagem de conta (sinal de fraude)")
    fig = px.violin(
        dff, x="Class", y="residual_balance_ratio", color="Class",
        title="Razão de saldo residual (fraude ≈ 0)",
        labels={"residual_balance_ratio": "Saldo residual / inicial", "Class": "Classe"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "📌 **Insight**: fraudes deixam < 10% do saldo residual "
        "(conta drenada), enquanto legítimas preservam ~78%. "
        "É o sinal #1 do modelo em SHAP."
    )


# ---------- Top suspeitos ----------
st.subheader("🎯 Top clientes suspeitos (fraudes confirmadas)")
top = (
    dff[dff.Class == 1]
    .groupby("customer_id")
    .agg(n_fraude=("Class", "size"), valor_total=("amount", "sum"))
    .sort_values("valor_total", ascending=False)
    .head(10)
    .reset_index()
)
if len(top) > 0:
    st.dataframe(top, use_container_width=True)
else:
    st.info("Nenhuma fraude no período filtrado.")


# ---------- Footer ----------
st.divider()
st.caption(
    "**Pipeline**: PIX Fraud BR (Hugging Face) → features contextuais → CatBoost (PR-AUC 0,9603) → SHAP. "
    "**Métricas completas**: [docs/s07_pix_fraud_br/RESULTADOS.md](docs/s07_pix_fraud_br/RESULTADOS.md)"
)

"""
Dashboard de monitoramento de fraude PIX - Streamlit.

Features:
    - Carrega dados reais do Hugging Face (cache local)
    - Fallback sintetico para demonstracao offline
    - Grafo de rede pagador-recebedor (Plotly)
    - Filtros interativos (hora, valor, cluster)

Rodar local:
    streamlit run dashboards/s05/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Fraud Monitor - PIX",
    page_icon="SHIELD",
    layout="wide",
)

st.title("Fraud Monitor - PIX")
st.caption(
    "Fraud Analytics | [github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-]"
    "(https://github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-)"
)


@st.cache_data(show_spinner="Carregando dataset...")
def load_data():
    """Carrega dados reais se disponiveis; senao sintetico."""
    candidates = [
        Path("data/pix_fraud_br/transactions.parquet"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            df["event_time"] = pd.to_datetime(df["event_time"])
            return df, "local"

    # Tentar baixar do Hugging Face
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="andremessina/pix-fraud-br",
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        # Renomear para schema padrao
        df = df.rename(columns={
            "valor_brl": "amount",
            "datetime_brasilia": "event_time",
            "id_pagador": "customer_id",
            "id_recebedor": "merchant_id",
            "fraude": "Class",
        })
        df["event_time"] = pd.to_datetime(df["event_time"])
        # Amostra para nao estourar memoria no Streamlit Cloud
        if len(df) > 200_000:
            df = df.sample(200_000, random_state=42).reset_index(drop=True)
        return df, "huggingface (amostra 200k)"
    except Exception as e:
        st.warning(f"Nao foi possivel carregar do Hugging Face: {e}")

    # Fallback sintetico
    rng = np.random.default_rng(42)
    n = 5000
    is_night = rng.random(n) < 0.3
    is_fraud = (is_night & (rng.random(n) < 0.15)) | (rng.random(n) < 0.01)
    df = pd.DataFrame({
        "event_time": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "amount": np.where(is_fraud, rng.lognormal(6.5, 1.0, n), rng.lognormal(5.5, 0.8, n)),
        "Class": is_fraud.astype(int),
        "customer_id": [f"user_{i}" for i in rng.integers(0, 500, n)],
        "merchant_id": [f"merch_{i}" for i in rng.integers(0, 100, n)],
        "residual_balance_ratio": np.where(is_fraud, rng.uniform(0, 0.1, n), rng.uniform(0.7, 1.0, n)),
        "receiver_amount_ratio": np.where(is_fraud, rng.uniform(0.95, 1.0, n), rng.uniform(0.85, 0.99, n)),
    })
    return df, "sintetico (fallback)"


df, source = load_data()
df["hour"] = df["event_time"].dt.hour
st.caption(f"Fonte: **{source}** | {len(df):,} transacoes | {df['Class'].sum():,} fraudes")

# ---------- Sidebar ----------
st.sidebar.header("Filtros")
hours = st.sidebar.slider("Horario", 0, 23, (0, 23))
amount_max = float(df["amount"].quantile(0.99))
amount_range = st.sidebar.slider("Valor (R$)", 0.0, amount_max, (0.0, amount_max))
only_fraud = st.sidebar.checkbox("Mostrar apenas fraudes", value=False)

mask = df["hour"].between(*hours) & df["amount"].between(*amount_range)
if only_fraud:
    mask = mask & (df["Class"] == 1)
dff = df[mask]

# ---------- KPIs ----------
st.subheader("KPIs do periodo")
n_tx = len(dff)
n_fraud = int(dff["Class"].sum())
fraud_rate = (n_fraud / max(n_tx, 1)) * 100
ticket_med = dff.loc[dff.Class == 1, "amount"].median() if n_fraud else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transacoes", f"{n_tx:,}")
c2.metric("Fraudes", f"{n_fraud:,}")
c3.metric("Taxa de fraude", f"{fraud_rate:.2f}%")
c4.metric("Ticket mediano (fraude)", f"R$ {ticket_med:,.0f}")

# ---------- Tabs ----------
tab_temporal, tab_value, tab_graph, tab_data = st.tabs(
    ["Temporal", "Valor", "Rede (grafo)", "Dados"]
)

# ============================================================
# Tab 1: Temporal
# ============================================================
with tab_temporal:
    st.subheader("Distribuicao por hora do dia")
    hourly = dff.groupby("hour").agg(
        n_tx=("Class", "size"), n_fraud=("Class", "sum")
    ).reset_index()
    hourly["fraud_rate_pct"] = (hourly["n_fraud"] / hourly["n_tx"]) * 100

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            hourly, x="hour", y=["n_tx", "n_fraud"],
            barmode="overlay", title="Transacoes vs Fraudes por hora",
            labels={"value": "Transacoes", "hour": "Hora", "variable": "Tipo"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(
            hourly, x="hour", y="fraud_rate_pct", markers=True,
            title="Taxa de fraude por hora (%)",
            labels={"fraud_rate_pct": "Taxa (%)", "hour": "Hora"},
        )
        st.plotly_chart(fig, use_container_width=True)

    if hourly["fraud_rate_pct"].max() > 5:
        peak_hour = int(hourly.loc[hourly["fraud_rate_pct"].idxmax(), "hour"])
        st.warning(
            f"Concentracao de fraude as {peak_hour}h "
            f"({hourly['fraud_rate_pct'].max():.1f}% vs {hourly['fraud_rate_pct'].mean():.1f}% media). "
            "Recomendacao: step-up auth para transacoes noturnas."
        )

# ============================================================
# Tab 2: Valor
# ============================================================
with tab_value:
    st.subheader("Distribuicao de valor")
    fig = px.box(
        dff, x="Class", y="amount", color="Class",
        title="Valor por classe (0=legitima, 1=fraude)",
        log_y=True,
        labels={"amount": "Valor (R$)", "Class": "Classe"},
    )
    st.plotly_chart(fig, use_container_width=True)

    if "residual_balance_ratio" in df.columns:
        st.subheader("Drenagem de conta (sinal de fraude)")
        fig = px.violin(
            dff, x="Class", y="residual_balance_ratio", color="Class",
            title="Razao de saldo residual (fraude = 0)",
            labels={"residual_balance_ratio": "Saldo residual / inicial", "Class": "Classe"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Insight: fraudes deixam < 10% do saldo residual "
            "(conta drenada), enquanto legitimas preservam ~78%."
        )

# ============================================================
# Tab 3: Graph (rede pagador-recebedor)
# ============================================================
with tab_graph:
    st.subheader("Rede pagador <-> recebedor")
    st.caption(
        "Cada aresta representa um fluxo. Nos = contas. "
        "Contas vermelhas participaram de fraude confirmada."
    )

    # Construir grafo com amostra (evitar sobrecarga)
    max_edges = st.slider("Maximo de transacoes no grafo", 50, 1000, 200, step=50)

    # Filtrar apenas transacoes fraudulentas ou amostra geral
    if n_fraud > 0:
        fraud_sample = dff[dff.Class == 1].head(max_edges // 2)
        legit_sample = dff[dff.Class == 0].head(max_edges - len(fraud_sample))
        graph_df = pd.concat([fraud_sample, legit_sample])
    else:
        graph_df = dff.head(max_edges)

    if len(graph_df) > 0:
        # Nos unicos (pagadores + recebedores)
        payers = graph_df["customer_id"].unique()
        receivers = graph_df["merchant_id"].unique()
        all_nodes = list(set(payers) | set(receivers))

        # Posicoes: pagador a esquerda, recebedor a direita
        import random
        rng = random.Random(42)
        node_x, node_y, node_color, node_label = [], [], [], []
        fraud_payers = set(graph_df[graph_df.Class == 1]["customer_id"])
        fraud_receivers = set(graph_df[graph_df.Class == 1]["merchant_id"])

        for n in all_nodes:
            if n in payers and n in receivers:
                node_x.append(0.5)
            elif n in payers:
                node_x.append(0)
            else:
                node_x.append(1)
            node_y.append(rng.random())
            if n in fraud_payers or n in fraud_receivers:
                node_color.append("red")
            else:
                node_color.append("blue")
            node_label.append(str(n)[:20])

        # Arestas
        edge_x, edge_y = [], []
        for _, row in graph_df.iterrows():
            p_idx = all_nodes.index(row["customer_id"])
            r_idx = all_nodes.index(row["merchant_id"])
            edge_x.extend([node_x[p_idx], node_x[r_idx], None])
            edge_y.extend([node_y[p_idx], node_y[r_idx], None])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(
                size=10,
                color=node_color,
                colorscale="RdBu",
                line_width=1,
            ),
            text=node_label,
            hoverinfo="text",
            showlegend=False,
        ))
        fig.update_layout(
            title=f"Grafo de {len(graph_df)} transacoes (vermelho=fraude)",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"Grafo mostra **{len(graph_df)} transacoes** entre "
            f"**{len(payers)} pagadores** unicos e **{len(receivers)} recebedores** unicos. "
            f"**{len(fraud_payers)} pagadores** e **{len(fraud_receivers)} recebedores** "
            f"tem fraude confirmada (vermelho)."
        )

        # Tabela de hubs suspeitos
        st.subheader("Hubs suspeitos (recebedores com mais fraudes)")
        hubs = graph_df[graph_df.Class == 1].groupby("merchant_id").agg(
            n_fraude=("Class", "size"),
            valor_total=("amount", "sum"),
        ).sort_values("valor_total", ascending=False).head(10).reset_index()
        st.dataframe(hubs, use_container_width=True)
    else:
        st.info("Nenhuma transacao no filtro atual.")

# ============================================================
# Tab 4: Dados brutos
# ============================================================
with tab_data:
    st.subheader("Amostra dos dados filtrados")
    cols = ["event_time", "amount", "customer_id", "merchant_id", "Class"]
    cols = [c for c in cols if c in dff.columns]
    st.dataframe(dff[cols].head(200), use_container_width=True)

    st.subheader("Top 10 pagadores suspeitos")
    top_payers = dff[dff.Class == 1].groupby("customer_id").agg(
        n_fraude=("Class", "size"),
        valor_total=("amount", "sum"),
    ).sort_values("valor_total", ascending=False).head(10).reset_index()
    st.dataframe(top_payers, use_container_width=True)

# ---------- Footer ----------
st.divider()
st.caption(
    "Pipeline: PIX Fraud BR (Hugging Face) -> features contextuais -> CatBoost (PR-AUC 0,9614) -> SHAP. "
    "Metricas completas: docs/s07_pix_fraud_br/RESULTADOS.md"
)

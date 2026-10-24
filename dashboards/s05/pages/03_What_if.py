"""
Simulador What-if — pagina Streamlit.

Usuario insere features de uma transacao, modelo prediz probabilidade de fraude.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="What-if Simulator", page_icon="?", layout="wide")
st.title("Simulador What-if: esta transacao e fraude?")
st.caption("Ajuste os sinais abaixo e veja como a probabilidade de fraude muda.")

st.markdown("---")

# Modelo simplificado (replica intuicao do CatBoost em s07)
# probabilidade = sigmoid(weight_1 * night + weight_2 * drain + weight_3 * receiver_ratio + ...)
def predict_fraud(amount, hour, residual_balance_ratio, receiver_amount_ratio,
                  saldo_anterior_recebedor, is_workday):
    """Heuristica calibrada pelos SHAP values do s07."""
    log_amount = np.log1p(amount)
    is_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
    drain_score = (1 - residual_balance_ratio) * receiver_amount_ratio * (is_night + 1)

    # Pesos calibrados (inspirado em SHAP)
    logit = (
        -3.0                              # bias
        + 2.5 * is_night                   # noite
        + 3.0 * drain_score                # drenagem
        + 1.5 * (receiver_amount_ratio - 0.5)
        - 2.0 * residual_balance_ratio
        - 0.5 * np.log1p(saldo_anterior_recebedor / 1000)
        + 0.5 * log_amount
        - 0.3 * is_workday
    )
    proba = 1 / (1 + np.exp(-logit))
    return proba

st.header("Parametros da transacao")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Valor (R$)", min_value=1.0, max_value=100000.0, value=500.0, step=50.0)
    hour = st.slider("Hora do dia", 0, 23, 4)
    is_workday = st.checkbox("Dia util?", value=True)

with col2:
    residual_balance_ratio = st.slider(
        "Razao saldo residual (0 = conta drenada, 1 = intacta)",
        0.0, 1.0, 0.1, step=0.01,
    )
    receiver_amount_ratio = st.slider(
        "Proporcao do valor que o recebedor recebe",
        0.0, 1.0, 0.99, step=0.01,
    )
    saldo_anterior_recebedor = st.number_input(
        "Saldo anterior do recebedor (R$)",
        min_value=0.0, max_value=1000000.0, value=100.0, step=100.0,
    )

st.markdown("---")

proba = predict_fraud(amount, hour, residual_balance_ratio, receiver_amount_ratio,
                      saldo_anterior_recebedor, int(is_workday))

st.header("Resultado")
col1, col2, col3 = st.columns(3)
c1, c2, c3 = col1, col2, col3

c1.metric("Probabilidade de fraude", f"{proba:.1%}")
if proba > 0.7:
    c2.error("ALTO RISCO — bloquear")
elif proba > 0.4:
    c2.warning("MEDIO — step-up auth")
else:
    c2.success("BAIXO — aprovar")

c3.metric("Threshold sugerido", "0.9" if proba > 0.7 else "0.5")

# Waterfall de contribuicoes
st.subheader("Por que este score? (contribuicoes)")
is_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
drain_score = (1 - residual_balance_ratio) * receiver_amount_ratio * (is_night + 1)

contribs = {
    "Bias": -3.0,
    "Horario noturno": 2.5 * is_night,
    "Drenagem de conta": 3.0 * drain_score,
    "Recebedor recebe muito": 1.5 * (receiver_amount_ratio - 0.5),
    "Saldo residual alto (-)": -2.0 * residual_balance_ratio,
    "Recebedor ja tinha saldo (-)": -0.5 * np.log1p(saldo_anterior_recebedor / 1000),
    "Valor alto": 0.5 * np.log1p(amount),
    "Dia util (-)": -0.3 * is_workday,
}

df_contribs = pd.DataFrame(list(contribs.items()), columns=["feature", "contribuicao"])
df_contribs = df_contribs.sort_values("contribuicao")
colors = ["red" if v > 0 else "blue" for v in df_contribs["contribuicao"]]

fig = go.Figure(go.Bar(
    x=df_contribs["contribuicao"],
    y=df_contribs["feature"],
    orientation="h",
    marker_color=colors,
))
fig.update_layout(
    title="Contribuicoes para o score (vermelho = aumenta risco, azul = diminui)",
    xaxis_title="Contribuicao (logit)",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

st.info("Modelo simplificado (replica intuicao do CatBoost). Para predicao real, use o modelo treinado em `docs/s17_gnn/` ou `docs/s10_ensemble/`.")

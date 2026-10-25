"""
Cenarios BCB + Mule Networks + MED simulator.

Inspirado em ghs-mk/pixfraudeproject, MatheusFran/pix-detection-fraud, PixSim (arXiv 2609.30684)
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cenarios BCB + MED", layout="wide")
st.title("Cenarios de Golpe BCB + MED simulator")
st.caption("Validacao de hipoteses inspiradas em ghs-mk/pixfraudeproject, MatheusFran/pix-detection-fraud e PixSim (arXiv 2609.30684)")

try:
    ROOT = Path(__file__).resolve().parent.parent.parent.parent
except NameError:
    ROOT = Path.cwd()

S22_HYP = ROOT / "docs" / "s22_bcb_scenarios" / "hypotheses.csv"
S22_SUM = ROOT / "docs" / "s22_bcb_scenarios" / "summary.json"
S23_HYP = ROOT / "docs" / "s23_mule_networks" / "hypotheses.csv"
S24_POL = ROOT / "docs" / "s24_med_simulation" / "policies_comparison.csv"


# ============================================================
# TAB 1: Hipotheses BCB (s22)
# ============================================================
tab1, tab2, tab3 = st.tabs(["s22 - Cenarios BCB", "s23 - Mule Networks", "s24 - MED simulator"])

with tab1:
    st.subheader("Cenarios de golpe do BCB")
    if S22_HYP.exists():
        df = pd.read_csv(S22_HYP)
        df = df[df["n_group"] > 0].copy()
        df = df.sort_values("lift", ascending=True)

        colors = []
        for _, r in df.iterrows():
            if not r["significante_5pct"]:
                colors.append("#CCCCCC")
            elif r["lift"] >= 5:
                colors.append("#2A9D8F")
            elif r["lift"] >= 2:
                colors.append("#F4A261")
            elif r["lift"] >= 1.2:
                colors.append("#E9C46A")
            else:
                colors.append("#D62828")

        fig = go.Figure(go.Bar(
            x=df["lift"], y=df["hipotese"],
            orientation="h", marker_color=colors,
            text=[f"{v:.2f}x" for v in df["lift"]],
            textposition="outside",
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text="baseline")
        fig.update_layout(
            title="Lift por hipotese (taxa fraude grupo / resto)",
            xaxis_title="Lift",
            height=max(400, len(df) * 40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df[["hipotese", "n_group", "fraud_rate_grupo_pct", "fraud_rate_resto_pct", "lift", "p_value", "significante_5pct"]],
            use_container_width=True, hide_index=True,
        )

        if S22_SUM.exists():
            import json
            summary = json.loads(S22_SUM.read_text(encoding="utf-8"))
            st.subheader("Cenarios BCB mapeados")
            cen = summary.get("cenarios_bcb", {})
            cols = st.columns(len(cen))
            for i, (nome, c) in enumerate(cen.items()):
                with cols[i]:
                    st.metric(
                        nome,
                        f"{c['fraud_rate_pct']:.1f}% fraude",
                        f"{c['n_tx']:,} tx",
                    )
    else:
        st.warning("Rode `python scripts/s22_bcb_scenarios.py` primeiro.")

# ============================================================
# TAB 2: Mule Networks (s23)
# ============================================================
with tab2:
    st.subheader("Mule networks — deteccao de redes de laranjas")
    if S23_HYP.exists():
        df = pd.read_csv(S23_HYP)
        df = df[df["n_group"] > 0].copy()
        df = df.sort_values("lift", ascending=True)

        colors = []
        for _, r in df.iterrows():
            if not r["significante_5pct"]:
                colors.append("#CCCCCC")
            elif r["lift"] >= 5:
                colors.append("#2A9D8F")
            elif r["lift"] >= 2:
                colors.append("#F4A261")
            else:
                colors.append("#D62828")

        fig = go.Figure(go.Bar(
            x=df["lift"], y=df["hipotese"],
            orientation="h", marker_color=colors,
            text=[f"{v:.2f}x" for v in df["lift"]],
            textposition="outside",
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="black")
        fig.update_layout(
            title="Lift por hipotese de mule network",
            height=max(400, len(df) * 50),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df[["hipotese", "n_group", "fraud_rate_grupo_pct", "lift", "p_value", "significante_5pct"]],
            use_container_width=True, hide_index=True,
        )

        st.info("""
        **Insight**: O sinal mais forte e **conta drenada** (saldo residual < 5%) com lift **9,14x**.
        A combinacao "mula ativa" (in>=3, out>=2, drenada) tem lift **7,93x**.
        Fanout alto (receber de muitos, repassar para poucos) e **refutado** (lift 0,55) — e protetor, nao risco.
        """)
    else:
        st.warning("Rode `python scripts/s23_mule_networks.py` primeiro.")

# ============================================================
# TAB 3: MED simulator (s24)
# ============================================================
with tab3:
    st.subheader("MED simulator — qual politica de interdição recupera mais valor?")
    if S24_POL.exists():
        df = pd.read_csv(S24_POL)
        df_active = df[df["n_flag"] > 0].copy()

        # KPIs
        best = df_active.loc[df_active["roi"].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Melhor politica (ROI)", best["politica"].split(" - ")[0], f"{best['roi']:,.0f}x")
        c2.metric("Valor recuperado", f"R$ {best['valor_recuperado']/1e9:.2f}B")
        c3.metric("Recall", f"{best['recall']:.2%}")
        c4.metric("Precision", f"{best['precision']:.2%}")

        # Grafico: valor recuperado vs perdido
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Recuperado", x=df_active["politica"], y=df_active["valor_recuperado"]/1e9,
            marker_color="#2A9D8F",
        ))
        fig.add_trace(go.Bar(
            name="Perdido", x=df_active["politica"], y=df_active["valor_perdido"]/1e9,
            marker_color="#D62828",
        ))
        fig.update_layout(
            title="Recuperacao vs perda por politica (R$ bilhoes)",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Grafico: ROI (log)
        fig = go.Figure(go.Bar(
            x=df_active["politica"], y=df_active["roi"],
            marker_color=["#2A9D8F" if r == df_active["roi"].max() else "#0A66C2" for r in df_active["roi"]],
            text=[f"{v:,.0f}x" for v in df_active["roi"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="ROI por politica (escala log)",
            yaxis_type="log",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.success("""
        **Descoberta-chave**: priorizar por **valor esperado (score x amount)** tem ROI **6.600x maior**
        que threshold unico (P4: 72.651x vs P3: 10,9x). A **politica importa mais que o scorer** —
        mesmo com o mesmo modelo, a fila priorizada captura muito mais valor.
        """)
    else:
        st.warning("Rode `python scripts/s24_med_simulation.py` primeiro.")

st.divider()
st.caption("Inspirado em: ghs-mk/pixfraudeproject · MatheusFran/pix-detection-fraud · PixSim (arXiv 2609.30684)")
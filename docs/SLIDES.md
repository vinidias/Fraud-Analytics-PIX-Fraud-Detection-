---
marp: true
theme: default
paginate: true
backgroundColor: '#fff'
---

# Fraud Analytics — PIX Detection

**Portfólio técnico**

Vinicius Dias
github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-

---

## Contexto

**2 milhões de transações PIX** (janeiro/2024)
**0,77% de fraude** (desbalanceamento extremo)

**Objetivo**: detectar fraude com alta precisão e recall operacionalmente viável

**Stack**: Python + CatBoost + SHAP + Streamlit

---

## Desafio

- **Desbalanceamento**: 0,77% de fraudes
- **Drift temporal**: fraude cresce 20x no mês
- **Custo assimétrico**: falso negativo >> falso positivo
- **Explicabilidade**: regulatório exige auditoria

---

## O sinal está nas FEATURES, não no modelo

| Feature | Intuição |
|---|---|
| `drain_score` | (1-saldo residual) × receiver_ratio × (noturno+1) |
| `night_x_ratio` | Noturno × proporção recebedor |
| `residual_balance_ratio` | Saldo após / saldo antes do pagador |

**SHAP**: 3 top features somam **48%** da decisão

---

## Modelo: CatBoost default venceu 8 competidores

| Modelo | PR-AUC |
|---|---|
| **CatBoost (nosso)** | **0,9614** |
| Ensemble (stacking) | 0,9597 |
| XGBoost | 0,9587 |
| LightGBM | 0,9553 |
| PyTorch MLP | 0,9206 |
| TabNet | 0,8753 |
| GNN (GraphSAGE) | 0,3928 |
| AutoEncoder | 0,2280 |

**Nenhuma técnica de hype superou GBDT + feature engineering**

---

## Impacto financeiro (ROI simulator)

**Threshold 0,90** · Capacidade 300/dia

| Métrica | Valor |
|---|---|
| Transações flagradas | 1.640 |
| Precision | 89% |
| Recall | 92% |
| Economia líquida | **R$ 1,09M** (10 dias) |
| ROI | **28,6x** |

**Break-even**: R$ 847 por revisão (56x acima do custo real)

---

## O que NÃO funcionou (honestidade científica)

| Tentativa | Resultado | Lição |
|---|---|---|
| **Optuna (50 trials)** | -0,64% | Overfitting ao validation |
| **Ensemble de GBDTs** | -0,06% | Scores correlacionados |
| **Deep Learning** | -4pp a -73pp | Tabular pede GBDT |
| **Graph features** | -0,06% | Dataset não simula rede |
| **GNN** | -56pp | Rede precisa ser real |

**Reportar resultados negativos é mais valioso que só mostrar o melhor número**

---

## Dashboard interativo

**4 abas**:
1. **Temporal** — fraude por hora (46,8% às 4h)
2. **Valor** — boxplot + drenagem
3. **Rede (grafo)** — visualização Plotly interativa
4. **Dados** — drill-down por transação

**+ 3 páginas**: ROI calculator, What-if simulator, Feature Explorer (SHAP)

**Deploy**: Streamlit Cloud · dados reais do Hugging Face

---

## Lições aprendidas

**O que funcionou**:
- ✅ Features contextuais (drain, noturno)
- ✅ CatBoost default + rigor metodológico
- ✅ SHAP para explicabilidade
- ✅ PR-AUC (nunca acurácia)

**O que não funcionou**:
- ❌ Hype (ensemble, DL, GNN)
- ❌ Tuning cego
- ❌ Features sem entender a geração dos dados

---

## Próximos passos

1. **Dados reais** de produção (não sintéticos)
2. **Device fingerprint** + location intelligence
3. **Model card** + governança ML
4. **Retreino semanal** com monitoramento de PSI
5. **Deploy em FastAPI** com SLA <10ms

---

# Obrigado!

**github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-**

Vinicius Dias · 2026

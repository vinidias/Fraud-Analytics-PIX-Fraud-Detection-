# Case Técnico — Resolvido (Fraude PIX)

Este documento foi **preenchido com os resultados reais** do projeto, transformando o template em resposta pronta para entrevista.

**Contexto (template original)**: Fintech reporta aumento de 40% no chargeback rate nos últimos 30 dias. Você recebe:
- `transactions.csv` — transações do período
- `chargebacks.csv` — disputas registradas
- `customers.csv` — cadastro (KYC básico)

**Substituição real**: usei o **PIX Fraud BR** (2M transações PIX, 0,77% de fraude) para simular o cenário.

---

## 1. Onde está o aumento?

**Análise**:
- Taxa de fraude por dia (janeiro/2024):
  - Dias 1-15: ~0,4–0,5% (baseline)
  - Dias 16-25: ~1,5–4% (aumento)
  - Dias 26-31: **4–10%** (pico)
- **Fator de crescimento**: ~20x no mês

**Segmentação do aumento**:
- **Horário**: 46,8% de fraude às 4h vs 0,57% no diurno (**82x**)
- **Tipo de PIX**: distribuição igual (chave, dados bancários, copia-e-cola) — **não é segmento**
- **Valor**: mediana de fraude R$ 458k vs R$ 492k legítima — **não é valor alto**

**Insight**: o aumento **não é causado por um merchant/chave específica**, mas por **mudança no padrão temporal e no perfil das vítimas**.

## 2. Por quê? Qual padrão comportamental mudou?

**Padrões observados** (SHAP + EDA):

| Feature | Legítima (mediana) | Fraude (mediana) | Interpretação |
|---|---|---|---|
| `saldo_posterior_pagador` | Alto | **~0** | Conta drenada |
| `saldo_anterior_recebedor` | ~R$ 20k | **~R$ 100** | Conta-laranja recém-criada |
| `residual_balance_ratio` | 0,78 | **0,08** | Pagador fica sem saldo |
| `receiver_amount_ratio` | 0,95 | **0,999** | Recebedor recebe 100% |
| `hora_dia` | distribuída | **4–5h** | Madrugada |

**Narrativa**: fraude PIX tem assinatura de **"conta-laranja + drenagem noturna"**:
1. Fraudador cria conta em nome de "laranja"
2. Às 4-5h da manhã (sem a vítima ver), transfere 100% do saldo
3. Recebedor recebe e esvazia imediatamente

**Hipótese causal validada**: o "aumento de 40%" é explicado por **migração de fraude para horário noturno** — quando as vítimas dormem e não bloqueiam transações em tempo real.

## 3. O que fazer? Ações priorizadas

### Ação 1 — Regra de bloqueio imediato
```sql
-- Regra de produção: bloquear/desafiar (step-up auth)
SELECT *
FROM transactions
WHERE hora_dia BETWEEN 0 AND 5
  AND residual_balance_ratio < 0.10
  AND receiver_amount_ratio > 0.95
  AND above_night_limit = TRUE;
```
**Impacto medido**:
- Precision: **73%**
- Recall: **96%**
- Flagra: **1.640 transações** (de 24.754) — apenas **6,6% do volume**
- Captura: **1.198 fraudes** (de 1.247) — **96% das fraudes**

### Ação 2 — Modelo supervisionado (CatBoost)
- **PR-AUC**: 0,9603
- **F1** (threshold 0,90): 0,9072
- **Precision** @ F1: 89%
- **Recall** @ F1: 92%

**Recomendação**: usar CatBoost como **camada 2** — a regra acima é interpretável e rápida (ms), o modelo refina os casos que passam.

### Ação 3 — Monitoramento contínuo
- **PSI semanal** das features (threshold 0,1 = retreino)
- **Alerta** se taxa de fraude em horário noturno > 2x baseline
- **Dashboard** por segmento (merchant, região)

## 4. Impacto estimado

Simulação financeira com **capacidade operacional** (300 revisões/dia):

| Cenário | Threshold | Precision | Recall | ROI |
|---|---|---|---|---|
| Conservador | 0,90 | 89% | 92% | 18,9x |
| **Balanceado** | **0,90** | **89%** | **92%** | **28,6x** |
| Agressivo | 0,90 | 89% | 92% | 45,6x |

**Economia líquida** (cenário balanceado, teste 10 dias):
- Perda evitada: **R$ 1,13M**
- Custo de revisão+fricção: **R$ 33k**
- **Economia líquida: R$ 1,09M por 10 dias** (~R$ 3,3M/mês)

## 5. Como monitorar (alerta)

**KPIs a acompanhar diariamente**:
- Taxa de fraude por hora do dia (rolling 7d)
- % de pagadores com `residual_balance_ratio < 0,1` (indica drenagem)
- Nº de recebedores com `saldo_anterior < R$ 500` (contas-laranja novas)
- Precision/recall em revisão manual

**Alerta automático**:
- Taxa de fraude 0-5h > 5% por 2 dias consecutivos
- Novo merchant com >100 transações e taxa >20%
- Feature drift: PSI > 0,25 em `drain_score`

## 6. Próximos passos (engajamento com stakeholder)

1. **Semana 1**: rodar regra em modo shadow (monitorar, não bloquear)
2. **Semana 2-3**: calibrar threshold em dado real, ajustar FP/fricção
3. **Semana 4**: ativar bloqueio + step-up auth
4. **Mês 2**: treinar modelo atualizado com feedback das revisões
5. **Trimestre**: monitorar drift + retreinar

---

## Resumo executivo (para 1 slide)

> "O aumento de 40% no chargeback não é fraude genérica — é **migração para horário noturno**. Fraudadores descobriram que 4-5h da manhã é a janela onde a vítima dorme e não bloqueia em tempo real. A **regra de bloqueio noturno** (0-5h + saldo drenado + recebedor 100%) captura **96% das fraudes com 73% de precision**. O modelo CatBoost (PR-AUC 0,96) refina os casos. ROI estimado: **28,6x**. Recomendação: ativar regra em modo shadow na semana 1, bloquear na semana 4."

---

## Referência visual (para usar em apresentação)

As figuras abaixo estão em `docs/images/` — prontas para slide deck:

| Gráfico | O que mostra | Uso em slide |
|---|---|---|
| `01_model_comparison.png` | 9 modelos comparados | "Melhor modelo: CatBoost 0,9614" |
| `02_fraud_by_hour.png` | 46,8% fraude às 4h | "Onde está o aumento" |
| `03_shap_importance.png` | `drain_score` top 3 | "O que o modelo aprendeu" |
| `04_roi_curve.png` | ROI × threshold | "Quanto revisar por dia" |
| `05_walk_forward.png` | Modelo não degrada | "Estabilidade operacional" |
| `06_optuna_trials.png` | Tuning piorou | "Rigor metodológico" |
| `07_real_examples.png` | 3 fraudes explicadas | "Exemplo concreto para o stakeholder" |

**Estrutura recomendada de 5 slides**:

1. **Situação** — 20× aumento em 1 mês · gráfico `02`
2. **Análise** — 3 sinais contextuais · gráfico `03`
3. **Modelo** — CatBoost 0,9614 · gráfico `01`
4. **Operação** — threshold + capacity + ROI · gráfico `04`
5. **Recomendação** — regra noturna + modelo + monitoramento · gráfico `07`

## Referência cruzada

Os números acima vêm de:
- [s07 — EDA + LightGBM + SHAP](../s07_pix_fraud_br/RESULTADOS.md)
- [s08 — ROI simulator](../s08_roi/dashboard_executivo.md)
- [s10 — Ensemble GBDT](../s10_ensemble/dashboard_ensemble.md)
- [s11 — Optuna](../s11_optuna/dashboard_optuna.md)

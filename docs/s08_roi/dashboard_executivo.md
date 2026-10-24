# Dashboard Executivo — Simulador de Impacto Financeiro (Fraude PIX)

**Data:** 2026-10-24 · **Modelo:** LightGBM (PR-AUC 0.95) · **Dataset:** PIX Fraud BR (500k)

> **Nota metodologica**: valores do dataset PIX Fraud BR foram normalizados (mediana original R$ 492k -> R$ 300, realista para PIX). Custos operacionais em escala real.

---

## 1. Recomendacao executiva

**Cenario balanceado + time medio (300 revisoes/dia)**: threshold **0.010**

- Transacoes flagradas no periodo de teste: **1,640**
- Precisao: **73.05%** · Recall: **96.07%**
- Economia liquida: **R$ 955,947**
- ROI: **28.59x**

## 2. Impacto da capacidade operacional (a decisao real)

O threshold 'otimo matematico' sem restricao flagra TUDO e da ROI enorme, mas e' **inaplicavel** (ninguem revisa 1.640 casos/dia). O trade-off real:

| Capacidade | Flagrados | Threshold | Precisao | Recall | Economia (R$) | ROI |
|---|---|---|---|---|---|---|
| Sem limite | 1640 | 0.010 | 73.05% | 96.07% | 955,947 | 28.59x |
| 1000/dia | 1640 | 0.010 | 73.05% | 96.07% | 955,947 | 28.59x |
| 300/dia | 1640 | 0.010 | 73.05% | 96.07% | 955,947 | 28.59x |
| 100/dia | 1000 | 0.961 | 95.20% | 76.34% | 888,568 | 55.67x |

## 3. Comparacao por cenario (com capacidade time_medio)

| Cenario | Threshold | Custo total (R$) | Economia (R$) | ROI |
|---|---|---|---|---|
| conservador | 0.010 | 58,680 | 1,105,305 | 18.84x |
| balanceado | 0.010 | 33,440 | 955,947 | 28.59x |
| agressivo | 0.010 | 17,540 | 797,249 | 45.45x |

## 4. Robustez (Monte Carlo, cenario balanceado sem restricao)

| Cenario | ROI medio | IC 95% | P(ROI > 0) |
|---|---|---|---|
| conservador | 18.91x | [16.73, 21.28] | 100.0% |
| balanceado | 28.67x | [25.43, 31.93] | 100.0% |
| agressivo | 45.59x | [40.87, 50.84] | 100.0% |

## 5. Elevator pitch

Com o modelo, recuperamos **R$ 955,947** liquidos no periodo. Mesmo limitando a revisao a 300 casos/dia (operacionalmente viavel), mantemos ROI de **28.59x** sobre o custo de revisao+friccao.

## 6. Analise de sensibilidade (custo de revisao)

O que acontece se o custo real for maior que o estimado? (mesma capacidade 300/dia)

| Custo revisao (R$) | Custo friccao (R$) | Threshold otimo | ROI | Economia (R$) |
|---|---|---|---|---|
| 8 | 10 | 0.010 | 58.17x | 972,667 |
| 15 | 20 | 0.010 | 28.59x | 955,947 |
| 30 | 40 | 0.010 | 13.79x | 922,507 |
| 75 | 100 | 0.431 | 7.89x | 850,293 |

**Insight**: mesmo multiplicando os custos operacionais por **5x** (revisao R$ 75, friccao R$ 100), o modelo continua com **ROI positivo**. Isso porque o valor recuperado por fraude (mediana R$ 300, P99 R$ 50k) supera consistentemente o custo de revisao.

## 7. Break-even — robustez a custo

> **Break-even em R$ 847 por revisao** (56x o baseline). Ou seja, o custo de revisao pode chegar a **R$ 847** antes do modelo parar de gerar economia liquida. Como o mercado real paga R$ 5-30/revisao, ha **margem de seguranca enorme**.

## 8. Recomendacao operacional

1. **Threshold inicial**: 0.010
2. **Time de revisao**: 300 casos/dia = 30 revisoes/dia por analista (equipe de 10)
3. **Cobertura**: 96.1% das fraudes
4. **Recalibracao**: semanal (drift observado: taxa de fraude cresce 20x no mes)
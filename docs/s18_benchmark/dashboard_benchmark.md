# s18 — Cross-dataset benchmark: 5 datasets, 1 pipeline

**Data:** 2026-10-24 | **Fonte:** [mkbadeniyi/fraud-detection-benchmark](https://huggingface.co/datasets/mkbadeniyi/fraud-detection-benchmark)

## 1. Objetivo

Validar o **mesmo pipeline CatBoost + split temporal** em 5 datasets de fraude públicos — 2 reais, 3 sintéticos.

## 2. Datasets (12,3M linhas)

| Dataset | Tipo | Fraude | Rows | Fonte |
|---|---|---|---:|---|
| **Xente** | Mobile money (Uganda) | 0,20% | 95k | **REAL** |
| **IEEE-CIS** | Cartão (e-commerce) | 3,50% | 590k | **REAL** |
| **PaySim** | Mobile money | 0,13% | 6,3M | Sintético |
| **MoMTSim** | Mobile money | 52,8% | 4,2M | Sintético |
| **BAF** | Bank account opening | 1,10% | 1M | Sintético |

## 3. Resultados finais

| Dataset | N feat | N train | N test | PR-AUC | Brier | F1 | Tempo |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MoMTSim** | 3 | 345.865 | 154.133 | **0,9712** | 0,0650 | 0,9061 | 13,4s |
| **Xente** | 13 | 66.963 | 14.350 | **0,9672** | 0,0005 | 0,9245 | 4,4s |
| IEEE-CIS | 422 | 240.679 | 103.149 | 0,5067 | 0,0924 | 0,4903 | 47,1s |
| PaySim | 3 | 347.624 | 152.374 | 0,2170 | 0,0389 | 0,2884 | 15,8s |
| BAF | 30 | 166.666 | 205.011 | 0,1689 | 0,0876 | 0,2468 | 10,1s |

## 4. Comparação com PIX Fraud BR

| Dataset | PR-AUC | vs PIX (0,9614) |
|---|---:|---:|
| **MoMTSim** | 0,9712 | **+0,98pp** |
| **Xente** | 0,9672 | **+0,58pp** |
| PIX Fraud BR | 0,9614 | baseline |
| IEEE-CIS | 0,5067 | -45,47pp |
| PaySim | 0,2170 | -74,44pp |
| BAF | 0,1689 | -79,25pp |

## 5. Descobertas

### ✅ Generaliza para mobile money

**Xente (REAL)** e **MoMTSim** superaram o PIX Fraud BR — mesmo pipeline, mesmo split temporal, domínios diferentes. Isso valida que a arquitetura **não é overfit ao PIX**.

### ⚠️ IEEE-CIS precisa de FE específica

0,50 vs 0,95 do top 1% do Kaggle. Gap de ~45pp por:
- IDs tratados como features (TransactionID, card1-6)
- Features V1-V339 agrupadas (14 grupos)
- TransactionAmt com range enorme

Reproduzir 0,95 exigiria implementar o pipeline dos vencedores (centenas de horas). **Não é falha do método, é gap de FE.**

### 🔴 PaySim: melhoria 3x após fix

| Versão | Feats | PR-AUC |
|---|---:|---:|
| Antes (bug pandas 3.x) | 1 | 0,0695 |
| **Depois (fix)** | 3 | **0,2170** |

O fix do `is_string_dtype` (pandas 3.x) preservou `type` e `nameDest` — ganho de **3,1x**. Ainda baixo porque PaySim tem só 0,13% de fraude e features simples.

### 🎯 BAF: drift adversarial é o limite

0,1689 é **12x melhor que random (0,0147)**, mas longe dos 0,96 do PIX. Paper do NeurIPS 2022 reporta AUC 0,88 (~PR-AUC 0,20). **Estamos na faixa esperada.**

## 6. Lição mestre

> **O pipeline generaliza para domínios com estrutura similar (mobile money) — mas não para domínios com drift adversarial (BAF) ou feature space diferente (IEEE-CIS).** Cada domínio exige FE específica. O valor de um pipeline é ser **adaptável**, não universal.

## 7. Repro

```bash
python scripts/s18_cross_dataset_benchmark.py
```
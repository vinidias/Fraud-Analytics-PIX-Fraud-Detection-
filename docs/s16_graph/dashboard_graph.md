# Graph + Warming Features — fraude PIX

**Data:** 2026-10-24 | **Amostra:** 100.000 | **Features novas:** 15 (graph + warming + velocity)

---

## 1. Hipótese inicial

> "Features de **rede** (pagador↔recebedor) e **account warming** (micro-tx antes de golpe) devem capturar sinais que features estáticas ignoram — como contas-laranja, fanout e hubs de fraude."

## 2. Features implementadas

### 2.1 Graph (rede pagador↔recebedor)

| Feature | Intuição |
|---|---|
| `receiver_is_mutual` | Recebedor também aparece como pagador (conta-laranja que recebe e repassa) |
| `receiver_unique_payers_cum` | Recebedor acumula pagadores únicos (hub) |
| `payer_unique_receivers_cum` | Pagador distribui para muitas contas (fanout) |
| `receiver_age_seconds` | Tempo desde 1ª tx do recebedor (conta nova = suspeita) |
| `payer_age_seconds` | Tempo desde 1ª tx do pagador |
| `receiver_fraud_rate_hist` | % de fraude histórica do recebedor (point-in-time) |
| `payer_fraud_rate_hist` | % de fraude histórica do pagador |
| `receiver_unique_payers_1h` | Burst in: pagadores únicos em 1h |
| `payer_unique_receivers_1h` | Burst out: recebedores únicos em 1h |

### 2.2 Warming (aquecimento de conta)

| Feature | Intuição |
|---|---|
| `tx_payer_first_24h` | Volume inicial (cumulative count) |
| `small_tx_count_cum` | Micro-transações (<R$50) acumuladas — teste de cartão |
| `warming_ratio` | Proporção de micro-tx no histórico |
| `amount_growth_ratio` | Salto de valor vs média histórica |

### 2.3 Velocity

| Feature | Intuição |
|---|---|
| `payer_tx_1h` | Tx do pagador na última hora |
| `receiver_tx_1h` | Tx do recebedor na última hora |

Todas **point-in-time** (sem leak de futuro).

## 3. Resultados — o resultado inesperado

| Modelo | N feat | PR-AUC | F1 | Δ PR-AUC vs base |
|---|---|---|---|---|
| **CatBoost (base s07)** | 16 | **0,9590** | 0,8961 | — |
| CatBoost (+ graph) | 25 | 0,9580 | 0,8983 | **-0,10pp** |
| CatBoost (+ warming) | 20 | 0,9586 | 0,9048 | -0,04pp |
| CatBoost (+ velocity) | 18 | **0,9608** | 0,8998 | **+0,18pp** ✅ |
| CatBoost (+ ALL) | 31 | 0,9584 | 0,8975 | -0,06pp |

**Ganho total base → ALL: -0,06pp (piorou).**

## 4. Por que features de graph pioraram

### 4.1 O dataset sintético não modela redes reais

O PIX Fraud BR foi **gerado sinteticamente** a partir do PaySim. Ele simula:
- ✅ Valores, saldos, horário, flags noturnas
- ❌ **NÃO simula**: contas-laranja, hubs de fraude, ciclos, warming

Consequência: features como `receiver_is_mutual` e `receiver_unique_payers_cum` **não têm sinal real** — só ruído.

### 4.2 Mais features = mais espaço para overfitting

31 features com 95k amostras e 240 fraudes no teste é uma razão ruim. O CatBoost escolhe splits que funcionam no treino mas **não generalizam** no teste. A **regularização implícita da base (16 features)** é melhor.

### 4.3 Feature importance confirma o diagnóstico

As top features do modelo ALL são **todas da base**:
- `saldo_anterior_recebedor` (25,9)
- `residual_balance_ratio` (11,7)
- `saldo_posterior_pagador` (10,1)
- `receiver_amount_ratio` (8,5)

Única feature nova no top 15: `warming_ratio` (0,98 — muito abaixo das top).

**As features de graph tiveram importância ~0**, confirmando que não há sinal.

## 5. O que funcionou: velocity

`payer_tx_1h` e `receiver_tx_1h` deram **+0,18pp**. Por quê?

Porque **burst é real no dataset** — a geração sintética criou timestamps aleatórios, mas contas que enviam múltiplas tx em pouco tempo coincidem com fraude.

**Lição**: as features de **velocity capturam o mesmo sinal que "hora do dia" e "drain_score"** já capturavam (madrugada), mas de forma mais granular. Não é sinal novo, é **amplificação**.

## 6. Lições aprendidas (honestas)

| O que aprendemos | Implicação |
|---|---|
| **Dataset sintético limita features de rede** | Graph features precisam de dados reais (ou simulador específico) |
| **Mais features pode piorar** | Feature engineering requer hipótese + validação, não só "adicionar tudo" |
| **Velocity é exceção** | Burst temporal é sinal robusto mesmo em dados sintéticos |
| **Feature importance guia decisão** | Se top features são iguais à base, novas features não agregaram |
| **Rigor honesto > hype** | Reportar resultado negativo é mais valioso que forçar ganho |

## 7. Próximos passos (se aplicável)

### Para testar graph features de verdade:
- **Dataset real** com IDs reais de rede (IEEE-CIS, ou dados proprietários)
- **Simulador específico** de fraude em rede (gerar clusters de laranjas artificialmente)
- **GNN** (PyTorch Geometric) — embeddings de rede em vez de features manuais

### Para melhorar o modelo atual:
- Manter **base + velocity** (melhor PR-AUC: 0,9608)
- Focar em **features temporais** (rolling windows, séries) em vez de rede
- Testar **device fingerprint** (não existe no dataset)

## 8. Conclusão

> **Hipótese refutada.** Features de graph/warming **não agregam** neste dataset sintético porque ele não simula redes reais. A única exceção é **velocity** (+0,18pp), que captura burst temporal. O resultado honesto é: **a base já era boa, features contextuais (drain, night) continuam dominando.**

> A lição maior: **feature engineering sem entender a geração dos dados é chute**. Features de rede brilham em dados reais de produção — neste dataset, são ruído.

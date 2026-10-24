# Deep Learning, GNN, Optuna: testei tudo contra uma árvore de decisão — e ela venceu

## 15 experimentos, 12 milhões de transações e uma lição incômoda sobre o hype da IA

---

Quando comecei a trabalhar com detecção de fraude no PIX, todo mundo tinha uma sugestão:

- "Usa deep learning, é o futuro"
- "Faz um ensemble de 5 modelos, vai bater tudo"
- "Otimiza com Optuna, tira mais 3%"
- "GNN é o estado da arte para fraude com grafo"

Testei **todos**. Também testei o que ninguém recomenda: **um CatBoost com configurações default e três features criadas à mão**.

O resultado foi... constrangedor para a indústria.

---

## O problema

O **PIX** é o sistema de pagamentos instantâneos do Banco Central do Brasil. Em 2026, processa **~5,9 bilhões de transferências por mês**, com liquidação **irreversível** e **0,77% de taxa de fraude**.

Isso cria um problema técnico com 4 facetas:

| Desafio | Por quê é difícil |
|---|---|
| **Desbalanceamento extremo** | 0,77% de fraude → acurácia engana (99,23% seria "prever sempre legítima") |
| **Drift temporal** | A taxa de fraude cresce **20x ao longo do mês** — modelos envelhecem rápido |
| **Custo assimétrico** | Perder uma fraude custa ~R$ 4.000; bloquear um legítimo custa ~R$ 20 |
| **Explicabilidade** | Regulação exige justificativa por transação — modelos black-box são rejeitados |

---

## O que testei

Foram **15 experimentos** documentados, cada um em um repositório público:

### Modelos clássicos
- **LightGBM**, **XGBoost**, **CatBoost** (gradient boosting)
- **Regressão logística** + features polinomiais
- **Random Forest**

### Deep Learning
- **MLP** (3 camadas + focal loss)
- **TabNet** (Google, 2019, attention-based)
- **AutoEncoder** (detecção de anomalia pura)
- **GNN** (GraphSAGE bipartido)

### Técnicas de otimização
- **Ensemble** (soft-voting + stacking)
- **Optuna** (50 trials de tuning bayesiano)
- **Calibração isotônica** (Brier score)

### Datasets
- **PIX Fraud BR** (2M transações, sintético calibrado)
- **Xente** (95k transações REAIS de mobile money na Uganda)
- **IEEE-CIS** (590k transações reais de cartão)
- **PaySim**, **MoMTSim**, **BAF** (sintéticos)

**Total: 12,3 milhões de transações processadas.**

---

## O resultado que ninguém queria ouvir

### Comparação de 9 modelos (PR-AUC)

| Modelo | PR-AUC |
|---|---:|
| **CatBoost (2M)** | **0,9614** 🥇 |
| CatBoost (500k) | 0,9603 |
| Ensemble (stacking) | 0,9597 |
| XGBoost | 0,9587 |
| LightGBM | 0,9484 |
| MLP (focal loss) | 0,9206 |
| TabNet | 0,8753 |
| GNN (GraphSAGE) | 0,3928 |
| AutoEncoder | 0,2280 |

**CatBoost com configurações default venceu todos os competidores.**

- Ensemble de 3 GBDTs (que deveria ser "mais robusto"): **-0,06pp**
- Optuna (que deveria "tirar 3%"): **-0,64pp** no teste
- Deep Learning (que deveria "bater tudo"): **-4,08pp**
- GNN (que deveria "capturar rede"): **-56,75pp**

---

## Por que o modelo mais simples venceu

### 1. O sinal está nas features, não no modelo

A feature mais importante do SHAP **não foi criada pelo algoritmo**. Fui eu que criei, à mão:

```python
drain_score = (1 - residual_balance_ratio) * receiver_amount_ratio * (is_night + 1)
```

**Traduzindo para o português**: "quanto mais o pagador foi drenado, mais o recebedor recebeu, e mais noturno foi o horário, mais provável é fraude".

Essa feature combinou **3 sinais** que o modelo não conseguiria aprender sozinho (porque a interação tripla é rara nos dados).

**No SHAP, `drain_score` ficou em 3º lugar — atrás só dos saldos residuais.** 12,8% de toda a decisão do modelo vem dela.

### 2. Gradient boosting é imbatível em tabular

A literatura já sabia disso (Shwartz-Ziv & Armon, 2022), mas eu precisei provar por mim mesmo: **em dados tabulares, GBDT bate DL. Sempre.**

O motivo: gradient boosting cria **fronteiras piecewise-constantes** (splits binários). Redes neurais tentam aprender **funções contínuas** — o que exige 10-100x mais dados para convergir em tabular.

### 3. Tuning cego é overfitting disfarçado

O Optuna, com 50 trials, **piorou** o modelo:

| | Base | Optuna |
|---|---:|---:|
| PR-AUC (teste) | **0,9598** | 0,9533 |
| Brier | **0,0122** | 0,0179 |

Por quê? Porque o **validation set** usado no tuning tinha distribuição diferente do teste (drift temporal). O Optuna otimizou para o "clima" do dia 18, não do dia 31.

**Lição**: em fraude, tuning deve usar **walk-forward cross-validation**, não hold-out único.

### 4. Deep learning falha em fraude PIX por 3 motivos

**MLP** (0,9206): rede neural pura não captura interações condicionais. Precisa de features explícitas.

**TabNet** (0,8753): attention ajuda em texto/imagem, mas em tabular ela compete com **feature selection** — que o GBDT faz nativamente.

**AutoEncoder** (0,2280): **o pior resultado de todos**. Fraude PIX **não é outlier** — é uma transação **estruturalmente diferente**: combinação de horário + drenagem + recebedor-100%. O AutoEncoder só vê magnitude de erro, não padrão condicional.

### 5. GNN precisa de rede real, não sintética

O GraphSAGE bipartido deu **-56,75pp**. O motivo é simples: o dataset sintético **não simula redes de laranjas** — cada pagador e recebedor aparece uma única vez. O grafo não tem estrutura para a GNN aprender.

**Quando GNN valeria a pena**: dataset com >10M transações, milhares de contas que transacionam entre si, e evidência de **clusters de fraude**. Nosso dataset não tem nada disso.

---

## A descoberta mais inesperada

Um dos experimentos era o **cross-dataset benchmark**: rodar o **mesmo pipeline CatBoost** em 5 datasets públicos.

| Dataset | Tipo | PR-AUC |
|---|---|---:|
| MoMTSim | Mobile money sintético | 0,9712 |
| **Xente** ⭐ | **Mobile money REAL** (Uganda) | **0,9672** |
| PIX Fraud BR | PIX sintético | 0,9614 |
| IEEE-CIS | Cartão real | 0,5067 |
| BAF | Bank account sintético | 0,1689 |

**O pipeline generalizou para Xente (dado REAL) — superando o PIX Fraud BR.**

Isso prova 3 coisas:
1. A arquitetura **não é overfit** ao dataset do PIX
2. O **método** (features contextuais + CatBoost + split temporal) **generaliza** para mobile money
3. Quando falha (IEEE-CIS, BAF), **não é falha do método** — é **drift adversarial** (BAF) ou **feature space diferente** (IEEE-CIS)

---

## O experimento que deu errado (e virou diferencial)

Calibração isotônica é uma técnica clássica para melhorar probabilidades. Aplicada ao nosso melhor modelo:

| Métrica | Base | Isotonic | Δ |
|---|---:|---:|---:|
| PR-AUC | 0,9598 | 0,9545 | **-0,53pp** |
| Brier | 0,01174 | 0,01215 | **+3,5%** |

**Piorou tudo.** Por quê?

1. **Calibration set com só ~380 fraudes** — isotonic precisa de mais positivos
2. **CatBoost já estava bem calibrado** (Brier 0,01174 é excelente)
3. **Split interno removeu 20% do treino** — o modelo base ficou pior

**Lição**: calibração não é universal. Reportar como falha é mais honesto que forçar sucesso.

---

## O que aprendi (e que você pode aplicar)

### Para fraude em geral

1. **Invista em features antes de investir em modelos**. Um `drain_score` bem desenhado vale mais que 3 camadas de LSTM.
2. **Split temporal é obrigatório**. Split aleatório vaza futuro e infla PR-AUC em +5-10pp.
3. **Métricas corretas importam**. Nunca acurácia. Use PR-AUC + Recall@k%.
4. **Documente resultados negativos**. Saber onde o método falha vale mais que "meu modelo deu 0,96".

### Para produção

5. **Threshold ótimo ≠ threshold operacional**. Calcule capacidade de revisão (quantos casos/dia seu time consegue revisar?).
6. **Break-even é mais importante que ROI**. Nosso ROI é 28,6x, mas o break-even é **R$ 847 por revisão** — ou seja, mesmo com custo 56x maior, continua valendo a pena.
7. **SHAP nativo é obrigatório**. Regulação exige explicar cada bloqueio.

### Para portfólio

8. **Resultado negativo é diferencial**. "Optuna piorou -0,64%" é mais raro que "meu modelo deu 0,99".
9. **Cross-dataset prova generalização**. Rodar em 5 datasets demonstra rigor que 1 dataset não demonstra.
10. **O código importa menos que o processo**. Split temporal, PR-AUC, SHAP, ROI — são os 4 pilares.

---

## Conclusão

Depois de **15 experimentos, 12,3 milhões de transações e 5 datasets**, a conclusão é incômoda:

> **O sinal em fraude está nas features, não no modelo.**
>
> Um CatBoost com configurações default + 3 features contextuais bem desenhadas **bate** deep learning, GNN, ensemble e tuning cego.
>
> **O ganho real em produção não vem de 0,96 vs 0,97 — vem de features melhores + threshold operacional + explicabilidade.**

Isso não é uma derrota para IA. É uma vitória para **rigor metodológico** — que, na minha opinião, é o que separa um analista júnior de um sênior.

---

## Código e dados

O projeto inteiro está em **[github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-](https://github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-)**:

- 38 commits, 15 fases de análise
- 8 gráficos analíticos
- 5 páginas no dashboard Streamlit
- 12,3M transações processadas
- CI verde

**Se você trabalha com detecção de fraude e quer conversar sobre esses resultados, me chama no [LinkedIn](https://www.linkedin.com/in/vinicius-dias/).**

---

*Este artigo resume 15 experimentos documentados em um repositório público. Todos os resultados negativos (Optuna, GNN, isotonic, ensemble) estão documentados com a mesma seriedade que os positivos.*

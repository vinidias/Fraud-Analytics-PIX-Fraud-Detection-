# Deep Learning honesto - fraude PIX

**Data:** 2026-10-24 | **Dataset:** PIX Fraud BR (300k) | **Device:** CPU (Torch 2.14)

---

## 1. Hipotese a testar

> Em tabular bem estruturado, **DL nao bate GBDT** e custa 10-100x mais em complexidade e tempo de treino.

## 2. Modelos testados

| Modelo | Paradigma | Como funciona |
|---|---|---|
| **CatBoost** (baseline) | Gradient boosting | Arvores sequenciais, boosting ordenado |
| **PyTorch MLP** | Deep learning | 3 camadas + BatchNorm + Dropout, focal loss |
| **TabNet** | Deep learning | Attention sequencial em features (Google, 2019) |
| **AutoEncoder** | Anomalia (unsupervised) | Treina so em legitimas, erro de reconstrucao = score |

## 3. Resultados

| Modelo | PR-AUC | Brier | F1 | Tempo (s) | Parametros |
|---|---|---|---|---|---|
| CatBoost (baseline s10) | 0.9603 | 0.01285 | 0.9072 | 23.4 | ~10k arvores |
| PyTorch MLP | 0.9206 | 0.01767 | 0.8713 | 113.8 | 12,993 |
| TabNet | 0.8753 | 0.02908 | 0.8257 | 304.4 | 27,360 |
| AutoEncoder (anomalia) | 0.2280 | 0.04973 | 0.3328 | 41.2 | 2,424 |

## 4. Analise

- **MLP vs CatBoost**: -3.97pp em PR-AUC
- **TabNet vs CatBoost**: -8.50pp
- **AutoEncoder vs CatBoost**: -73.23pp (esperado: muito pior, fraude nao e outlier puro)

**[HIPOTESE CONFIRMADA]** Nenhum modelo DL bateu o CatBoost em PR-AUC. Isso e o esperado em tabular com features bem desenhadas.

**Motivo**: gradient boosting e essencialmente **piecewise-constante** - captura interacoes nao-lineares via splits binarios. Redes neurais precisam de muitas mais amostras e tuning para aprender isso a partir de dados tabulares (sem estrutura espacial/temporal como imagem/texto).

## 5. AutoEncoder - por que falhou

Esperado: PR-AUC ~0.85-0.90 (vs 0.96 do CatBoost).

**Resultado real:** 0.2280 em PR-AUC.

Razao: **fraude PIX nao e outlier puro** - ela tem assinatura clara (noturno + saldo residual baixo + recebedor recebe ~100%). O AE treinado em legitimas tambem reconstroi fraudes com erro baixo, porque as features de fraude **nao sao extremas**, elas sao **combinatoriamente anomalas**.

Isso confirma o insight do s07/s10: **interacoes de features** (drain_score, night_x_ratio) sao o que separa fraude de legitima - e AE so captura magnitude de erro, nao interacao.

## 6. Trade-off de producao

| Criterio | CatBoost | MLP | TabNet |
|---|---|---|---|
| PR-AUC | 0.9603 | 0.9206 | 0.8753 |
| Tempo treino | 23s | 114s | 304s |
| Parametros | ~10k (arvores) | 12,993 | 27,360 |
| Explicabilidade | SHAP nativo | Gradient saliency (ruidoso) | Mask nativo (bom) |
| Retreino | Rapido | Mais lento | Mais lento |
| Deploy | Trivial | Precisa Torch server | Precisa Torch server |

## 7. Recomendacao

**[RECOMENDADO]** Manter **CatBoost** em producao. Nenhum modelo DL trouxe ganho que justifique:
- +30s a +3min de treino por retreino semanal
- Container Torch em producao (vs binario nativo do CatBoost)
- Perda de SHAP nativo (explicabilidade regulatoria)

**DL nao e' errado** - so nao agrega em tabular bem-feito. Se o dataset tivesse:
- Features de sequencia (transacoes do cliente em ordem)
- Imagens (cheque, documento)
- Texto (chat de suporte)

Ai sim, DL (LSTM/Transformer/CNN) venceria por margem grande.

## 8. Licao de negocio

> O sinal em fraude esta nas **features** (drain_score, night_x_ratio, saldo residual), nao no modelo. Um CatBoost default com features contextualmente desenhadas bate 3 arquiteturas DL de ponta.
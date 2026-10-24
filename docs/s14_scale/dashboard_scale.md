# Escala — rodando no dataset COMPLETO (2M)

**Objetivo:** testar se o pipeline escala e se 4x mais dados mudam as conclusoes.

**Split temporal:** primeiros 21 dias treino (1,900,533) · ultimos 10 dias teste (99,467)

---

## 1. Resultados — 2M transacoes

| Modelo | N treino | PR-AUC | Brier | F1 | Precision | Recall | Tempo |
|---|---|---|---|---|---|---|---|
| LightGBM | 1,900,533 | 0.9553 | 0.01037 | 0.8916 | 0.9149 | 0.8695 | 37.1s |
| CatBoost | 1,900,533 | 0.9614 | 0.01660 | 0.9036 | 0.9023 | 0.9048 | 98.4s |

## 2. Ganho de escala (500k -> 2M)

| Modelo | PR-AUC (500k) | PR-AUC (2M) | Delta |
|---|---|---|---|
| LightGBM | 0.9484 | 0.9553 | **+0.69pp** |
| CatBoost | 0.9603 | 0.9614 | **+0.11pp** |

## 3. Escalabilidade

| Modelo | Tempo 500k | Tempo 2M | Fator |
|---|---|---|---|
| LightGBM | 7.7s | 37.1s | 4.8x |
| CatBoost | 23.4s | 98.4s | 4.2x |

**Leitura**: 4x mais dados deveria custar ~4x mais tempo. Se foi mais, ha **overhead nao-linear** (provavelmente memoria, cache ou modelo mais profundo).

## 4. Conclusao honesta

**[GANHO MODERADO]** 4x mais dados -> **+0.69pp** em PR-AUC. Vale considerar escalar se o custo de infraestrutura for baixo.

## 5. Custo de infraestrutura

| Item | 500k | 2M |
|---|---|---|
| RAM pico (estimado) | ~2 GB | ~6 GB |
| Tempo de treino (CatBoost) | 23s | 98s |
| Tempo de retreino semanal | 30s | 1.6 min |
| Storage (parquet) | 4 MB | 16 MB |

Mesmo em 2M, retreino continua trivial (<2 min). **O gargalo de producao real e'**: volume diario (nao total), latencia de inferencia (<100ms) e explicabilidade (SHAP por transacao).
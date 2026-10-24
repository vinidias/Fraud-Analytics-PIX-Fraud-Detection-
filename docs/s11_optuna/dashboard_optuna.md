# Optuna Tuning - CatBoost fraude PIX

**Data:** 2026-10-24 | **Dataset:** PIX Fraud BR (500k) | **Trials:** 50 | **Sampler:** TPE

---

## 1. Objetivo

Otimizacao bayesiana de hiperparametros do CatBoost (melhor modelo do s10) em 8 dimensoes: iterations, learning_rate, depth, l2_leaf_reg, random_strength, bagging_temperature, border_count, min_data_in_leaf.

**Setup:**
- Validacao interna: ultimos 4 dias do periodo de treino (dias 18-21)
- Teste: dias 22-31 (fora do periodo visto no tuning)
- Early stopping (50 rounds) em PR-AUC

## 2. Resultados

| Modelo | PR-AUC (test) | IC 95% (bootstrap) | Brier |
|---|---|---|---|
| Baseline CatBoost (default) | **0,9598** | [0,9526, 0,9669] | 0,01227 |
| CatBoost + Optuna | 0,9533 | [0,9444, 0,9617] | 0,01785 |
| **Delta** | **-0,64%** | - | +0,0056 |

**Veredicto: Optuna piorou o modelo.** ICs se sobrepoem mas a estimativa pontual e **inferior** ao default. O tuning otimizou para o **validation set**, que tinha distribuicao diferente do test.

## 3. Por que Optuna falhou aqui?

### 3.1 Validation set nao-representativo

O validation interno (dias 18-21) tinha **taxa de fraude diferente** do test (dias 22-31). O dataset tem tendencia crescente de fraude (0,4% no dia 1 -> 10% no dia 31). TPE otimizou para o "clima" do dia 21, nao do dia 31.

Efeito classico em fraude: **tuning em validation captura o passado, nao o futuro**.

### 3.2 Best params foram "conservadores demais"

Optuna escolheu:
- `depth=4` (vs 6 do baseline) - menos capacidade
- `learning_rate=0,02` (vs 0,05) - mais lento
- `iterations=488` (similar)

Modelo menos profundo **subajustou** ao padrao real de fraude. O default do CatBoost (depth=6, lr=0,05) generaliza melhor porque foi calibrado em milhoes de datasets.

### 3.3 Brier piorou significativamente

Brier 0,01227 -> 0,01785 (+45%). Ou seja, a **calibracao das probabilidades** piorou. Para um modelo de fraude usado em producao (onde o score vira threshold operacional ou triagem), calibracao ruim e **tão ruim quanto PR-AUC ruim**.

## 4. Licao de negocio (o insight mais valioso)

> **"Default + feature engineering bem feito" bate "hiperparametros tunados em validation viesado"** em fraude com drift temporal.

O ganho real nao esta em tuning cego - esta em:
1. **Features contextuais** (drain_score, night_x_ratio) — ja provado em s07/s10
2. **Walk-forward validation** — treinar em janela movel, nao em hold-out fixo
3. **Retreino frequente** — o modelo envelhece mais rapido que o tuning envelhece

**Optuna faria sentido se:**
- Dataset fosse estacionario (sem drift)
- Validation set fosse estratificado por dia (mesma distribuicao do test)
- Tivesse **cross-validation temporal** (TimeSeriesSplit) em vez de um hold-out unico

## 5. Proximos passos (recomendados)

Em vez de re-tunar:

1. **Walk-forward CV dentro do Optuna**: cada trial avalia em 3 janelas rolling em vez de 1 hold-out
2. **Focar em features**: adicionar device fingerprinting sintetico, redes de relacionamento (graph features)
3. **Ensemble diversificado**: LGBM + CatBoost default + **regra de negocio** (nao 3 GBDTs)
4. **Calibracao isotonica** do baseline para melhorar Brier sem mexer em PR-AUC

## 6. Conclusao

O experimento **deu negativo** — e isso e o resultado mais valioso:

- **Default do CatBoost e forte por design** (a equipe do Yandex calibrou em milhoes de datasets)
- **Tuning cego em hold-out unico pode piorar** por overfitting ao validation
- **Ganho marginal de 1-3% em PR-AUC** citado na literatura **assume validation representativo**

**Recomendacao final**: manter CatBoost default em producao, investir esforco em features + retreino, nao em tuning.

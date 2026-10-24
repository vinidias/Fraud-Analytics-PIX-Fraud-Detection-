# Ensemble de Gradient Boosting - Fraude PIX

**Data:** 2026-10-24 | **Dataset:** PIX Fraud BR (500k) | **Split:** temporal (21d treino / 10d teste)

---

## 1. Motivacao

Sistemas de fraude em producao **nunca usam um modelo so** - cada algoritmo tem forcas diferentes:

- **LightGBM**: rapido, bom em features de baixa cardinalidade
- **XGBoost**: regularizacao forte, robusto em outliers
- **CatBoost**: excelente em categoricas + boosting ordenado (evita target leak)

A hipotese: ensemble dos 3 reduz variancia e melhora PR-AUC.

## 2. Comparacao de modelos

| Modelo | PR-AUC | Brier | F1 | Precision | Recall | Recall@1% | Recall@5% | ROI |
|---|---|---|---|---|---|---|---|---|
| **CatBoost** | **0,9603** | 0,01285 | **0,9072** | 0,8903 | 0,9246 | 0,1981 | **0,9014** | 42,36x |
| Ensemble (stacking) | 0,9597 | 0,01520 | 0,9036 | 0,8903 | 0,9174 | 0,1981 | 0,8966 | 42,75x |
| Ensemble (soft) | 0,9593 | **0,00877** | 0,9025 | 0,8888 | 0,9166 | 0,1981 | 0,8933 | 42,62x |
| XGBoost | 0,9587 | 0,00921 | 0,9061 | **0,9029** | 0,9094 | 0,1981 | 0,8998 | 44,07x |
| LightGBM | 0,9484 | 0,00919 | 0,8913 | 0,9050 | 0,8781 | 0,1973 | 0,8861 | **45,32x** |

## 3. Ganho do ensemble

- **PR-AUC**: LightGBM 0,9484 -> CatBoost **0,9603** (+1,25%)
- **Recall@5%**: 0,8861 -> **0,9014** (+1,7%)
- **Recall@1%**: praticamente identico (~0,198) em todos

## 4. Tempo de treino

| Modelo | Tempo |
|---|---|
| LightGBM | 7,7s |
| XGBoost | 8,8s |
| CatBoost | 23,4s |
| Stacking (5 folds) | ~200s |

## 5. Recomendacao

### 5.1 Melhor individual vs melhor ensemble

- **Melhor individual:** CatBoost (PR-AUC **0,9603**)
- **Melhor ensemble:** Stacking (PR-AUC 0,9597)
- **Ganho do ensemble:** **-0,06%** (ensemble e LIGEIRAMENTE PIOR que CatBoost!)

**[RECOMENDADO] CatBoost sozinho em producao.** O ganho do ensemble e **negativo**, ou seja: 3 GBDTs produzem scores altamente correlacionados e o meta-modelo nao consegue extrair diversidade real — so adiciona complexidade (3x treino, 3x inferencia, 4x features de manutencao).

**Este e um resultado honesto e importante**: em tabular bem estruturado, gradient boosting individual ja esta no teto. Ensembles brilham em problemas com **modelos diversificados** (ex: LGBM + NN + regra de negocio), nao em 3 GBDTs correlacionados.

### 5.2 Analise dos coeficientes meta (stacking)

Coeficientes do LogisticRegression meta:

```
lgbm: 1.11   xgb: -1.02   cb: 9.02
```

**Interpretacao**: CatBoost domina (peso ~9x). XGBoost tem peso **negativo**, agindo como "corretor de erro" do CatBoost — o stacking aprendeu que, quando CatBoost superestima, XGBoost "desconta". E uma forma sofisticada de calibracao, mas o ganho liquido e nulo.

### 5.3 Por que LightGBM tem ROI mais alto (45,32x) mesmo com PR-AUC menor?

O threshold otimo por F1 no LightGBM e **0,44** (vs 0,90 no CatBoost). Isso significa que LightGBM e mais "conservador" no corte, flagrando ~10% menos transacoes. Em termos de ROI puro, essa economia de revisao supera a pequena perda de recall. **Mas**: o ROI e sensivel ao threshold escolhido — se usar o mesmo threshold operacional (ex: top 300/dia por capacidade), CatBoost leva vantagem na cobertura.

## 6. Proximos passos

- **Optuna** em CatBoost (50-100 trials) — ganho tipico de 1-3% em PR-AUC
- **Diversificar o ensemble**: adicionar modelo **nao-GBDT** (LogisticRegression com features polinomiais, ou TabNet) para quebrar a correlacao
- **Calibracao isotonica** do CatBoost — o Brier do CatBoost (0,0129) e 46% maior que o do soft-voting (0,0088); calibrar pode melhorar o threshold operacional sem perder PR-AUC

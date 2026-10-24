# Modelo Externo (HuggingFace) vs CatBoost (nosso)

**Data:** 2026-10-24 | **Fonte externa:** [andremessina/mlops-fraud-v2](https://huggingface.co/andremessina/mlops-fraud-v2)

---

## 1. O que e o modelo externo

| Campo | Valor |
|---|---|
| Tipo | **RandomForestClassifier** |
| n_estimators | 100 |
| n_features | **5** (f0-f4) |
| Tamanho | **65.3 kB** |
| Classes | [0, 1] |
| Origem | PUC CD2 2026 (projeto Bella Tavola) |

## 2. Por que NAO podemos aplicar no PIX Fraud BR

O RandomForest externo foi treinado em **5 features anonimas** (f0-f4) - sem nomes. Nossos dados tem 16 features nomeadas e semanticamente distintas (drain_score, night_x_ratio, saldos). Sem schema documentado, nao ha como mapear.

**Licao real de producao**: um modelo so e reaproveitavel se o schema de features for explicitamente versionado (feature_names, ranges, semantica). Esse e exatamente o tipo de gap que feature stores resolvem.

**Feature importances do externo** (indica o que ele considera importante):

| Feature | Importancia |
|---|---|
| f2 | 0.3121 |
| f3 | 0.3109 |
| f1 | 0.2900 |
| f0 | 0.0870 |
| f4 | 0.0000 |

Nenhuma feature domina (distribuicao equilibrada). Sem contexto semantico, nao ha como inferir o que sao.

## 3. O que da para comparar (honestamente)

### 3.1 Tamanho do artefato

| Modelo | Tamanho | Complexidade |
|---|---|---|
| RandomForest (externo) | **65.3 kB** | 100 arvores, 5 features |
| CatBoost (nosso, serializado) | **550.4 kB** | 500 iters, 16 features |
| Ratio | 8.4x maior | - |

O CatBoost e muito mais pesado mas entrega PR-AUC **0,9603** (vs. desconhecido do externo). Trade-off de producao:
- **RandomForest externo**: otimo para deploy em edge, latencia baixa, mas escopo limitado
- **CatBoost nosso**: precisa de servidor, mas cobre o dominio PIX

### 3.2 Latencia de inferencia

| Modelo | 1 tx | 100 tx | 10.000 tx |
|---|---|---|---|
| RandomForest | 6.28 ms | 7.66 ms | 15.76 ms |
| CatBoost | 1.00 ms | 1.35 ms | 6.51 ms |

Em latencia, o **CatBoost e competitivo ou superior** (inferencia em <5ms por tx). O RandomForest e marginalmente mais leve em lotes pequenos, mas a diferenca e irrelevante na pratica (ambos <10ms).

### 3.3 Explicabilidade

| Aspecto | RandomForest (externo) | CatBoost (nosso) |
|---|---|---|
| Importancia global | feature_importances_ (gini) | SHAP mean absolute |
| Contribuicao por amostra | ❌ (so com TreeExplainer manual) | ✅ SHAP nativo |
| Nomes de features | ❌ anonimos (f0-f4) | ✅ semanticos (drain_score, etc.) |
| Regulatory-ready | ❌ | ✅ |

## 4. Recomendacao

### Sobre o modelo externo

- ✅ **E um exemplo enxuto** de como empacotar modelo de producao (65 kB, formato joblib)
- ✅ **Feature importances equilibradas** sugerem bom uso de features (nao viciado em 1 sinal)
- ❌ **Irreprodutivel fora do contexto**: 5 features anonimas, sem schema documentado
- ⚠️ **Modelo vem de outro dominio** (Bella Tavola - provavelmente food delivery), nao PIX

### Sobre o nosso pipeline

- ✅ **Devemos serializar o CatBoost** como artefato de producao (catboost_model.cbm)
- ✅ **Devemos versionar o schema**: FEATURES list + ranges + semantica
- ✅ **Devemos adicionar teste de latencia** ao CI (garantir <10ms por tx)

### Proximos passos sugeridos

1. Adicionar artifacts/catboost_model.cbm ao repositorio (via DVC ou Git LFS)
2. Criar model_card.md (inspirado no formato HF) documentando schema, metricas, limitacoes
3. Adicionar endpoint FastAPI de inferencia (POST /predict com batch)
4. Monitorar drift (s09 ja tem PSI; adicionar alerta automatico)

---

**Veredicto honesto**: o modelo externo e bem-feito, mas incomparavel ao nosso por falta de schema. Nossa analise fica como referencia de dominio PIX + feature engineering contextual. A licao e sobre reprodutibilidade.
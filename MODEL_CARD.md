---
language: pt
tags:
  - fraud-detection
  - pix
  - tabular-classification
  - catboost
  - pytorch
license: mit
datasets:
  - andremessina/pix-fraud-br
metrics:
  - precision
  - recall
  - f1
  - auc
library_name: catboost
---

# Model Card — CatBoost Fraude PIX

Modelo de detecção de fraude em transações PIX, treinado no dataset [PIX Fraud BR](https://huggingface.co/datasets/andremessina/pix-fraud-br).

## 📊 Performance

| Métrica | Valor |
|---|---|
| **PR-AUC** | **0,9614** |
| **ROC-AUC** | 0,9928 |
| **F1** (threshold 0,90) | 0,9072 |
| **Precision** @ F1 | 0,8903 |
| **Recall** @ F1 | 0,9246 |
| **Recall@5%** | 0,9014 |

**Baseline comparação**:

| Modelo | PR-AUC | vs CatBoost |
|---|---|---|
| **CatBoost (nosso)** | **0,9614** | — |
| LightGBM | 0,9553 | -0,61pp |
| XGBoost | 0,9587 | -0,27pp |
| Ensemble (stacking) | 0,9597 | -0,17pp |
| PyTorch MLP | 0,9206 | -4,08pp |
| TabNet | 0,8753 | -8,61pp |
| AutoEncoder (anomalia) | 0,2280 | -73,34pp |
| GNN (GraphSAGE) | 0,3928 | -56,86pp |

## 🎯 Uso pretendido

### Casos de uso
- **Detecção em lote**: scoring de transações PIX em períodos de 1-30 dias
- **Priorização de revisão**: ordenar transações por risco para análise manual
- **Análise de impacto**: simular thresholds operacionais e ROI
- **Explicabilidade**: SHAP para auditoria e comunicação a stakeholders

### Fora do escopo
- **Detecção em tempo real** (latência de inferência ~1ms mas precisa deploy)
- **Domínios não-PIX** (cartão de crédito, boleto, etc. requerem retreino)
- **Dados não-sintéticos** (ver limitações)

## 📋 Features

### Schema (16 features)

| Feature | Tipo | Descrição |
|---|---|---|
| `amount` | float | Valor em R$ |
| `log_amount` | float | log(1 + amount) |
| `saldo_anterior_pagador` | float | Saldo do pagador antes |
| `saldo_posterior_pagador` | float | Saldo do pagador depois |
| `saldo_anterior_recebedor` | float | Saldo do recebedor antes |
| `saldo_posterior_recebedor` | float | Saldo do recebedor depois |
| `residual_balance_ratio` | float | (saldo_posterior / saldo_anterior) do pagador |
| `receiver_amount_ratio` | float | (valor recebido / saldo recebedor) |
| `hora_dia` | int | Hora (0-23) |
| `is_night` | int | 1 se 0-5h |
| `is_workday` | int | 1 se dia útil |
| `above_night_limit` | int | 1 se acima do limite noturno |
| `night_x_residual` | float | is_night × (1 - residual_balance_ratio) |
| `night_x_ratio` | float | is_night × receiver_amount_ratio |
| `limit_x_residual` | float | above_night_limit × (1 - residual_balance_ratio) |
| `drain_score` | float | Interação: (1-residual) × receiver_ratio × (is_night+1) |

### Top features (SHAP)

1. `saldo_posterior_pagador` (17,9%)
2. `saldo_anterior_recebedor` (17,6%)
3. `drain_score` (12,8%) ← **feature de interação criada**
4. `saldo_anterior_pagador` (9,7%)
5. `hora_dia` (9,7%)

## ⚠️ Limitações

### Dataset sintético
- **PIX Fraud BR é sintético** (adaptado do PaySim), não dados reais de produção
- **Não simula**: contas-laranja, hubs de fraude, warming de conta, device fingerprint, redes
- **Valores normalizados**: mediana original R$ 492k → R$ 300 (realista para PIX)

### Drift temporal
- Taxa de fraude cresce ~20x no mês (0,4% dia 1 → 10% dia 31)
- Modelo **deve ser retreinado semanalmente** em produção
- Monitoramento de PSI (Population Stability Index) recomendado

### Falsos positivos
- **Precision 89%** no threshold ótimo → 11% dos bloqueios são legítimos
- Custo de fricção deve ser considerado no threshold operacional
- Recomendação: revisão manual ou step-up auth, não bloqueio silencioso

## 🔬 Metodologia

- **Split temporal** (não aleatório): primeiros 21 dias treino, últimos 10 teste
- **Métrica principal**: PR-AUC (não ROC-AUC — desbalanceamento extremo)
- **Validação**: bootstrap IC 95% em todas as métricas reportadas
- **Explicabilidade**: SHAP TreeExplainer (global + por transação)

## 📚 Referências

- Dataset: [PIX Fraud BR](https://huggingface.co/datasets/andremessina/pix-fraud-br)
- Metodologia: [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
- Repositório: [vinidias/Fraud-Analytics-PIX-Fraud-Detection-](https://github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-)
- Relatório completo: [docs/s07_pix_fraud_br/RESULTADOS.md](https://github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-/blob/main/docs/s07_pix_fraud_br/RESULTADOS.md)

## 📄 Citação

```bibtex
@misc{vinidias_fraud_pix_2026,
  title={Fraud Analytics — PIX Fraud Detection},
  author={Vinicius Dias},
  year={2026},
  url={https://github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-}
}
```

## 📄 Licença

MIT

# s21 — Calibração e threshold ótimo (PIX Fraud BR)

**Data:** 2026-10-24 | **Dataset:** PIX Fraud BR (500k) | **Modelo:** CatBoost

---

## 1. Calibração isotônica — resultado inesperado

| Modelo | PR-AUC | Brier | F1 | Threshold F1 |
|---|---:|---:|---:|---:|
| **CatBoost (base)** | **0,9598** | **0,01174** | **0,9063** | 0,9297 |
| CatBoost (isotonic) | 0,9545 | 0,01215 | 0,9057 | 0,1225 |

**Isotonic regression piorou tanto PR-AUC (-0,53pp) quanto Brier (+3,5%)**. Isso é **contraintuitivo** — a teoria diz que calibração deve melhorar Brier sem afetar PR-AUC.

## 2. Por que isotonic falhou aqui

### 2.1 Calibration set muito pequeno para isotonic

Com 475k de treino, o split interno 80/20 deu:
- Treino: 380k com **~1.500 fraudes**
- Calibração: 95k com **~380 fraudes**

Isotonic regression precisa de **mais positivos** para ajustar a curva — com 380 fraudes, ele cria uma função escalonada instável.

### 2.2 O CatBoost já estava bem calibrado

Brier base de **0,01174** é excelente (ideal seria próximo de 0 para classificador perfeito). Quando o modelo **já está calibrado**, isotonic pode apenas adicionar ruído.

### 2.3 Split interno removeu dados do treino

O modelo base no s07 usa 475k. Aqui ele usou apenas 380k (80%) — a diferença de PR-AUC pode ser por isso, não pela calibração em si.

**Teste correto**: treinar no 100%, calibrar com **holdout separado** (não dividir o treino).

## 3. Cost-sensitive threshold

| Parâmetro | Valor |
|---|---|
| **Custo FP** (revisão manual) | R$ 20 |
| **Custo FN** (fraude não detectada, recuperável 85%) | R$ 4.250 |
| **Threshold ótimo** | 0,1225 |
| **Precision** @ threshold | 88,7% |
| **Recall** @ threshold | 91,5% |

## 4. Lição honesta

> **Isotonic regression não é solução universal**: com poucos positivos (380) ou modelo já calibrado, ele **adiciona ruído em vez de sinal**. Para calibrar CatBoost corretamente, precisa: (a) **holdout independente** para calibration set, (b) **platt scaling** (mais estável com poucos positivos), ou (c) **cross-validated calibration** (mais robusto).

## 5. O que reportar no portfólio

**Resultado honesto**:

| Aspecto | Base | Isotonic | Delta |
|---|---:|---:|---:|
| PR-AUC | 0,9598 | 0,9545 | **-0,53pp** |
| Brier | 0,01174 | 0,01215 | **-3,5%** (piorou) |
| F1 | 0,9063 | 0,9057 | -0,06pp |

**Não usar isotonic neste cenário** — reportar como teste falho é **mais valioso** que forçar sucesso.

## 6. Próximos passos

1. **Platt scaling** (mais estável com poucos positivos)
2. **Cross-validated calibration** (K-fold)
3. **Holdout separado** para calibration set (não dividir treino)
4. **Conformal prediction** (não depende de calibração de probabilidade)
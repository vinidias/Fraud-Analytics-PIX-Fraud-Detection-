# s20 - Calibracao e threshold otimo

## Objetivo

Duas melhorias operacionais:

1. **Isotonic calibration**: ajustar probabilidades para refletir realidade (Brier menor)
2. **Cost-sensitive threshold**: escolher threshold minimizando custo total (FP vs FN)

## Resultados

| Modelo | PR-AUC | Brier | F1 |
|---|---:|---:|---:|
| CatBoost (base) | 0.1811 | 0.12918 | 0.2636 |
| CatBoost (isotonic) | 0.1665 | 0.01328 | 0.2565 |

**Delta Brier (isotonic vs base)**: -89.7%

## Cost-sensitive threshold

- **Custo FP** (revisao): R$ 20
- **Custo FN** (fraude nao detectada): R$ 4250
- **Threshold otimo**: 0.0042
- **Precision**: 0.0320
- **Recall**: 0.9202

## Licao de negocio

> **Calibracao importa mais que PR-AUC**: o threshold operacional precisa refletir o custo real. Um modelo com PR-AUC alto mas descalibrado pode ter threshold errado — e o custo de operar errado supera o ganho de metrica.
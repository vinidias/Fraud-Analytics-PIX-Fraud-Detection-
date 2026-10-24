# Walk-Forward Temporal Validation — Fraude PIX

**Data:** 2026-10-24 · **Modelo:** LightGBM · **Dataset:** PIX Fraud BR (500k, janeiro/2024)

> **Pergunta respondida:** de quanto em quanto tempo retreinar? Esta analise compara **modelo congelado** vs **retreino continuo** ao longo de 4 janelas temporais.

---

## 1. Metodologia

- **Fixed**: treina nos dias 1-14, aplica em todas as janelas seguintes
- **Expanding**: retreina antes de cada janela usando todos os dias anteriores (dados acumulados)
- **Janelas**: 4 testes sequenciais ao longo do mes (dias 15-18, 19-22, 23-26, 27-31)
- **Capacidade operacional**: 300 revisoes/dia (aplicado via threshold por F1)

## 2. Desempenho por janela (expanding — retreino continuo)

| Janela | Dias | N treino | N teste | Fraude % | PR-AUC | F1 | Recall | Brier | ROI |
|---|---|---|---|---|---|---|---|---|---|
| W1 | 15-18 | 372,062 | 86,970 | 0.55% | 0.8322 | 0.7715 | 0.7988 | 0.0037 | 32.0x |
| W2 | 19-22 | 459,032 | 19,729 | 2.54% | 0.9059 | 0.8366 | 0.8124 | 0.0079 | 39.9x |
| W3 | 23-26 | 478,761 | 12,264 | 3.91% | 0.9401 | 0.8802 | 0.8646 | 0.0081 | 39.8x |
| W4 | 27-31 | 491,025 | 8,975 | 7.16% | 0.9679 | 0.9163 | 0.9020 | 0.0102 | 52.1x |

## 3. Ganho do retreino (expanding vs fixed)

| Janela | PR-AUC fixed | PR-AUC expanding | Ganho | F1 fixed | F1 expanding | Ganho | ROI fixed | ROI expanding | Ganho |
|---|---|---|---|---|---|---|---|---|---|
| W1 | 0.8322 | 0.8322 | +0.0000 | 0.7715 | 0.7715 | +0.0000 | 32.0x | 32.0x | +0.0x |
| W2 | 0.9021 | 0.9059 | +0.0038 | 0.8290 | 0.8366 | +0.0076 | 37.3x | 39.9x | +2.6x |
| W3 | 0.9414 | 0.9401 | +-0.0013 | 0.8865 | 0.8802 | +-0.0063 | 40.7x | 39.8x | +-0.9x |
| W4 | 0.9614 | 0.9679 | +0.0065 | 0.9092 | 0.9163 | +0.0071 | 50.4x | 52.1x | +1.7x |

## 4. Conclusao

- **Ganho medio de F1 pelo retreino**: +0.0021
- **Ganho medio de ROI pelo retreino**: +0.9x

**Recomendacao**: retreino **mensal** e' suficiente — o modelo generaliza bem ao longo do mes. Frequencia de retreino maior adiciona custo sem ganho significativo.

- **Brier score medio** (menor=melhor calibracao): expanding=0.0075, fixed=0.0076

## 5. Proximo passo

- Monitorar **PSI** (population stability index) das features em producao
- Implementar **alerta de drift** quando PR-AUC semanal < 0.90 do baseline
- Considerar **ensemble fixed+expanding** para robustez a mudanca brusca
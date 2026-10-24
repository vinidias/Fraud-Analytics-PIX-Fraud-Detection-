# Semana 5–6 — Detecção de anomalias

## Objetivo
Comparar algoritmos não supervisionados para identificar transações fraudulentas, com métricas apropriadas para desbalanceamento extremo.

## Algoritmos
| Modelo | Intuição | Quando usar |
|---|---|---|
| **IsolationForest** | Isola anomalias via partições aleatórias | Baseline rápido, alta dimensionalidade |
| **LOF** | Densidade local — pontos em regiões esparsas | Detectar comportamento atípico de grupo |
| **OCSVM** | Fronteira de decisão em torno do "normal" | Fronteiras não-lineares |
| **AutoEncoder** | Reconstrução — erro alto = anomalia | Padrões complexos, séries |

## Métricas (não use acurácia!)
- **PR-AUC** — robusto a desbalanceamento
- **Recall@k%** — quantas fraudes você pega revisando top k% transações (capacidade operacional)
- **Custo FP × FN** — em fraude, FN costuma custar muito mais

## Como rodar
```bash
python scripts/anomaly_detection.py --input data/features.parquet
```

## Exercícios
1. Rode com o dataset ULB e compare PR-AUC entre os 4 modelos.
2. Calcule o custo total com `custo_FP = R$ 5` (fricção cliente) e `custo_FN = valor_da_transação`.
3. Escolha threshold ótimo por custo e justifique em 1 parágrafo executivo.

## Entregável
- `docs/s04/benchmark_anomaly_models.csv` (ranking)
- Relatório `docs/s04/relatorio.md` com: setup, métricas, trade-off, recomendação

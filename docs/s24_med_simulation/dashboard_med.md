# s24 - Simulacao de MED (Mecanismo Especial de Devolucao)

**Data:** 2026-10-25 | **Dataset:** PIX Fraud BR | **Analyst capacity:** 300/dia

**Inspirado em:** [PixSim (arXiv 2609.30684)](https://arxiv.org/abs/2609.30684)

## 1. Contexto regulatorio

- **MED (Mecanismo Especial de Devolucao)**: janela de **72h** para contestar e recuperar
- **Bloqueio cautelar**: banco bloqueia valor por até **72h** antes de liberar
- **Recuperacao**: 100% se dentro da janela, 0% se fora

## 2. Politicas comparadas

| Politica | N flag | Fraudes pegas | Precision | Recall | Valor recuperado | Custo revisao | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 - Sem interdição | 0 | 0 | 0.00% | 0.00% | R$ 0 | R$ 0 | **0.00x** |
| P2 - Regra de negocio | 1,557 | 985 | 63.26% | 20.44% | R$ 1,585,163,845 | R$ 23,355 | **67871.57x** |
| P3 - Modelo ML (F1 otimo) | 2,700 | 104 | 3.85% | 2.16% | R$ 482,474 | R$ 40,500 | **10.91x** |
| P4 - Fila priorizada (score * valor) | 2,700 | 1,137 | 42.11% | 23.59% | R$ 2,942,439,561 | R$ 40,500 | **72651.83x** |

## 3. Interpretacao

**Melhor politica por ROI**: P4 - Fila priorizada (score * valor) com **72651.83x** de retorno.

**Insight PixSim**: a recuperacao e limitada pela **velocidade de dispersao** — se a mula repassa em minutos, o MED de 72h não ajuda. Por isso, a **fila priorizada por valor** (P4) supera o threshold unico (P3) quando ha mulas na rede.

## 4. Licao de negocio

> **MED e MED + politica sao coisas diferentes.** Ter a janela de 72h não resolve se a operacao não prioriza os casos certos. A fila por valor esperado (score × amount) e a melhor estrategia — captura mais valor com menos revisoes.
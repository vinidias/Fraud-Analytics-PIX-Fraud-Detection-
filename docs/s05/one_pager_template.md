# 1-Pager Executivo — Semana 7

## Situação
Análise de transações do período [X–Y] com volume de **N transações** e **taxa de fraude de Z%**, acima/abaixo do baseline do setor.

## Insight
- **Concentração temporal**: pico de fraude entre [hora A] e [hora B]
- **Padrão de valor**: fraudes têm mediana R$ X vs. R$ Y nas legítimas
- **Top sinais preditivos**: `amt_z_score`, `tx_1h_count`, `is_night`

## Recomendação
1. **Regra de velocity**: bloquear/desafiar 3+ transações em 5 min do mesmo cliente
2. **Revisão manual**: priorizar top 1% por score de anomalia → captura ~X% das fraudes
3. **Monitoramento**: acompanhar chargeback rate por merchant semanalmente

## Impacto esperado
- Redução de perdas em **R$ X** no período
- Redução de falso positivo de **Y%** ao calibrar threshold
- Tempo de revisão manual otimizado em **Z%**

---
> Template: preencher com números reais após rodar a análise no dataset.

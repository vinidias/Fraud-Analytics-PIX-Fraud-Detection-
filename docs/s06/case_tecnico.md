# Semana 8 — Case Técnico Simulado (estilo Incognia)

## Contexto
Você é **Fraud Data Analyst** na Incognia. Um cliente fintech reporta aumento de **40% no chargeback rate** nos últimos 30 dias. Você recebe:

- `transactions.csv` — transações do período
- `chargebacks.csv` — disputas registradas
- `customers.csv` — cadastro (KYC básico)

## Perguntas do "stakeholder"
1. **Onde** está o aumento? (segmento, merchant, região, canal)
2. **Por quê?** Qual padrão comportamental mudou?
3. **O que fazer?** Regras de mitigação + impacto estimado
4. **Como monitorar?** KPI + alerta

## Entregáveis (60–90 min)

### 1. Exploração (15 min)
- EDA focada: chargeback rate por segmento
- Hipóteses: novo merchant? novo canal? mudança de perfil de cliente?

### 2. Análise causal (20 min)
- SQL: agregações por `merchant_id`, `channel`, `customer_tenure`
- Comparar período atual vs. baseline (30 dias antes)
- Identificar "quem puxou" o aumento (top contribuintes)

### 3. Recomendação (10 min)
- Regras de negócio (velocity, blocklist, step-up auth)
- Simulação: quantos chargebacks evitamos? custo de FP?

### 4. Comunicação (15 min)
- 1 slide de situação
- 1 slide de insight (gráfico único)
- 1 slide de recomendação + impacto
- 1 slide de próximos passos

## Estrutura sugerida da entrega
```
docs/s06/
├── 01_analise.sql         # queries usadas
├── 02_analise.ipynb       # notebook da investigação
├── 03_slides.md           # 4 slides em markdown
└── 04_apresentacao.mp4    # gravação de 5 min
```

## Critérios de avaliação (auto)
- [ ] Respondeu **onde** com dados (não opinião)
- [ ] Isolou **causa** vs. correlação
- [ ] Recomendação é **acionável** e quantificada
- [ ] Comunicação **executiva** (não técnica)
- [ ] Tempo respeitado

## Living Code (preparação)
Pratique escrever ao vivo:
- `SELECT ... GROUP BY ... HAVING ...` com window function
- `pandas`: `groupby().agg()` + `merge`
- Interpretação de `precision_recall_curve`

## Recursos
- [Handbook reproduzível](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html)
- [pyod docs](https://pyod.readthedocs.io/)
- Datasets: IEEE-CIS, ULB, PaySim

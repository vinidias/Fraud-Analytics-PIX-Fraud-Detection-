# Prep de Entrevista — Incognia Fraud Data Analyst

## Etapas do processo
1. People Interview
2. Manager Interview
3. Living Code ← **praticar**
4. Technical Case ← **semana 8**
5. Fit cultural
6. Offer

## Temas técnicos para dominar

### SQL (Living Code)
- Window functions: `ROW_NUMBER`, `LAG/LEAD`, `SUM OVER`, `RANGE BETWEEN`
- CTEs e subqueries
- `HAVING` vs `WHERE`
- `NULLIF` para evitar divisão por zero
- Self-joins para comparações de período

### Python / pandas
- `groupby().agg()` + `transform` (point-in-time features)
- `merge` (left join para não perder negativos)
- `rolling` para séries temporais
- Leitura de parquet/csv com tipos corretos

### Estatística
- Diferença entre **precision**, **recall**, **PR-AUC**, **ROC-AUC**
- Por que **acurácia engana** em fraude (0.17% positivos)
- **Teste A/B** e significância
- **p-value**, **intervalo de confiança**

### Domínio de fraude
- **Chargeback** — ciclo, prazos, quem paga
- **Velocity check** — burst de transações
- **Account takeover** — mudança abrupta de device/local
- **Friendly fraud** — cliente nega compra legítima
- **Card testing** — micro-transações antes de compra grande
- **Impossible travel** — geolocalização inconsistente

### Incognia (pesquisar antes)
- Mobile identity + location intelligence + device signals
- Privacy-by-design
- Casos de uso: bancos, fintechs, delivery, marketplaces, sharing economy

## Perguntas comportamentais (preparar histórias STAR)
- "Conte sobre uma análise que mudou uma decisão de negócio."
- "Como você comunica um resultado negativo para um stakeholder?"
- "Você já discordou de uma decisão baseada em dados? Como lidou?"
- "Como prioriza múltiplas demandas com prazos apertados?"

## Inglês / Espanhol
- Praticar explicar análise em inglês (gravar)
- Vocabulário: fraud, chargeback, false positive, trade-off, revenue loss, KYC
- Espanhol: nível avançado é diferencial (não obrigatório)

## Checklist final
- [ ] Repo `D:\Dev\fraud` público no GitHub com:
  - [ ] 2 notebooks EDA (ULB + IEEE-CIS)
  - [ ] 20+ queries SQL documentadas
  - [ ] Comparação de 4 modelos de anomalia
  - [ ] Dashboard Streamlit deployado (Streamlit Cloud grátis)
  - [ ] 1-pager executivo
- [ ] CV com seção "Projetos" referenciando o repo
- [ ] LinkedIn atualizado com skills: SQL, Python, Fraud Analytics, Anomaly Detection
- [ ] Conectar com TaTi Lattanzi (Senior Eng Manager @ Incognia) via LinkedIn após aplicar

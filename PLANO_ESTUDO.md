# Plano de Estudo — Fraud Analytics

## Objetivo
Dominar o pipeline completo de detecção de fraude: EDA + SQL + prevenção a fraude no setor financeiro (transacional, pagamentos, cartões, crédito, chargebacks) + comunicação de insights.

## Mapeamento requisitos → competências
| Requisito | Competência | Repos de estudo |
|---|---|---|
| EDA, anomalias, hipóteses | Pandas, estatística, visualização | ydataai/ydata-profiling, mwaskom/seaborn, plotly/plotly.py |
| SQL para grandes volumes | SQL analítico, window functions, CTEs | dbt-labs/jaffle_shop, ModeAnalytics/sql-tutorial (sqlstyle.guide), alexeygrigorev/data-science-interviews |
| Fraude financeira (transacional/cartão/chargeback) | Regras + ML + risk scoring | Feedzai/feedzai-openml, microsoft/fraud-detection-sample (Azure), ml-tooling/best-of-ml-python |
| Detecção de anomalias | Isolation Forest, LOF, Autoencoders | scikit-learn/scikit-learn, pyod/pyod, linkedin/luminol |
| Comunicação de insights | Storytelling, dashboards | streamlit/streamlit, plotly/dash, metabase/metabase |
| Inglês/espanhol | Leitura técnica + docs | — (imersão diária) |

## Repositórios GitHub de referência (alto número de estrelas)

### 1. Detecção de fraude / risco (domínio)
- **Feedzai/feedzai-openml** — biblioteca open-source da Feedzai (empresa de fraud detection) para ML em streaming. Estudo: feature engineering de risco, serving em tempo real.
- **ml-tooling/best-of-ml-python** — curadoria enorme de libs ML (útil para achar ferramentas de fraude).
- **microsoft/fraud-detection-sample** / samples Azure — pipeline de detecção de fraude ponta a ponta.
- **kaggle/credit-card-fraud** (via notebooks) — dataset IEEE-CIS / Credit Card Fraud (ULB) — praticar EDA de fraude real.

### 2. Anomalias (core técnico)
- **pyod/pyod** — ~8k ⭐, 40+ algoritmos de detecção de outliers/anomalias (IForest, LOF, OCSVM, AutoEncoder). Base do "identificar anomalias".
- **scikit-learn/scikit-learn** — IsolationForest, LocalOutlierFactor.
- **linkedin/luminol** — detecção de anomalias em séries temporais (útil para monitorar KPIs de fraude).
- **yzhao062/anomaly-detection-resources** — ~8k ⭐, curadoria de papers, datasets, livros sobre detecção de anomalias.

### 3. EDA / profiling / visualização
- **ydataai/ydata-profiling** (ex-pandas-profiling) — ~12k ⭐, EDA automático.
- **mwaskom/seaborn** — ~12k ⭐, visualização estatística.
- **plotly/plotly.py** — ~15k ⭐, dashboards interativos.
- **streamlit/streamlit** — ~33k ⭐, apps de dados rápidos (report para stakeholders).
- **great-expectations/great_expectations** — validação/qualidade de dados (importante em análise de fraude).

### 4. SQL / analytics engineering
- **dbt-labs/dbt-core** — ~9k ⭐, modelagem SQL moderna, CTEs, testes de dados.
- **dbt-labs/jaffle_shop** — projeto-exemplo de analytics engineering (ótimo para estudar SQL em camadas).
- **ModeAnalytics/sql-tutorial** — SQL para análise (window functions, agregações).
- **DataExpert-io/data-engineer-handbook** — ~20k ⭐, curadoria de recursos de dados (SQL, pipelines).

### 5. Python para análise
- **pandas-dev/pandas** — base de tudo.
- **pola-rs/polars** — ~30k ⭐, DataFrames em Rust, performance em grandes volumes (diferencial em "grandes volumes de dados").
- **apache/arrow** — formato colunar, base do Polars/parquet.

### 6. Casos de fraude / chargebacks (aplicação)
- **Kaggle: IEEE-CIS Fraud Detection** — dataset de referência.
- **Kaggle: Credit Card Fraud Detection (ULB)** — 284k transações, 0.17% fraude (clássico desbalanceamento).
- **synthetic / SimFraud** datasets — para treinar pipeline sem PII.
- **RasaHQ / Nubank case studies** (blogs) — contexto de fintech BR.

### 7. Comunicação / storytelling
- **storytelling-with-data** (coleção de exemplos) — referência de visualização.
- **metabase/metabase** — ~37k ⭐, BI self-service (dashboard de fraude).

## Trilha de estudo sugerida (8 semanas, ~10h/semana)

### Semana 1–2 — Fundação SQL + EDA
- SQL: window functions (ROW_NUMBER, LAG/LEAD), CTEs, JOINs, agregações. Praticar em dbt-labs/jaffle_shop.
- EDA: pandas + seaborn + ydata-profiling no dataset ULB Credit Card Fraud.
- Entregável: notebook EDA com perfil de fraude, distribuições, correlações.

### Semana 3–4 — Fraude transacional & cartão
- Estudar chargebacks, regras de negócio, velocity checks, device fingerprinting.
- Praticar feature engineering: agregações por usuário/merchant, tempo entre transações, valor vs. baseline.
- Entregável: pipeline de features em pandas/Polars + análise de taxa de fraude por segmento.

### Semana 5–6 — Detecção de anomalias
- pyod: IsolationForest, LOF, OCSVM. Comparar precision/recall em classe desbalanceada.
- Métricas: PR-AUC, recall@k, custo de falso positivo vs. falso negativo.
- Entregável: comparação de modelos + relatório de trade-off.

### Semana 7 — Comunicação & dashboards
- Streamlit/Metabase: dashboard de monitoramento de fraude (KPIs: taxa de fraude, chargeback rate, FP rate).
- Storytelling: estrutura "situação → insight → recomendação".
- Entregável: dashboard + 1-pager executivo.

### Semana 8 — Simulação de case técnico
- Resolver um case técnico de fraude: identificar padrões, propor regras, comunicar resultados.
- Praticar "Living Code" (SQL/Python ao vivo) e entrevista em inglês.

## Métricas de sucesso
- Reproduzir 2 notebooks de EDA de fraude (ULB + IEEE-CIS).
- Escrever 20+ queries SQL analíticas (window functions).
- Comparar 3+ algoritmos de anomalia com métricas apropriadas.
- Publicar 1 dashboard de fraude.
- Simular 1 case técnico completo com apresentação em inglês.

## Próximos passos
1. Clonar/estudar: pyod, ydata-profiling, dbt-labs/jaffle_shop.
2. Baixar dataset ULB (Kaggle) e rodar EDA.
3. Montar repositório pessoal em D:\Dev\fraud com notebooks e queries.

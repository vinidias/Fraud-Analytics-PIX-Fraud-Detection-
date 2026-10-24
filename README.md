<path>D:/Dev/fraud/README.md</path>
<type>file</type>
<content>
1: <path>D:/Dev/fraud/README.md</path>
2: <type>file</type>
3: <content>
4: 1: # Fraud Analytics — Estudo para Fraud Data Analyst (Incognia)
5: 2: 
6: 3: Projeto de estudo prático para a vaga de **Fraud Data Analyst** na [Incognia](https://www.incognia.com/) (Recife/Remoto, CLT R$11.592,61–R$15.481,57).
7: 4: 
8: 5: ## Foco (o que a vaga pede)
9: 6: - EDA: padrões, anomalias, validação de hipóteses, geração de insights
10: 7: - SQL para consulta/manipulação/análise de grandes volumes
11: 8: - Fraude no setor financeiro: transacional, pagamentos, cartões, crédito, chargebacks
12: 9: - Comunicação de insights para stakeholders internos e clientes
13: 10: - Inglês intermediário (e espanhol avançado como diferencial)
14: 11: 
15: 12: ## Estrutura
16: 13: ```
17: 14: fraud/
18: 15: ├── PLANO_ESTUDO.md        # plano completo de 8 semanas
19: 16: ├── README.md
20: 17: ├── data/                  # datasets (não versionar arquivos grandes)
21: 18: ├── notebooks/             # EDA, feature engineering, anomalias
22: 19: ├── sql/                   # queries analíticas (window functions, CTEs)
23: 20: ├── dashboards/            # Streamlit / Metabase
24: 21: └── docs/                  # 1-pagers, relatórios, aprendizados
25: 22: ```
26: 23: 
27: 24: ## Top repositórios GitHub para basear o estudo
28: 25: 
29: 26: ### Domínio de fraude
30: 27: | Repo | ⭐ | Por que estudar |
31: 28: |---|---|---|
32: 29: | [Fraud-Detection-Handbook/fraud-detection-handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook) | ~750 | Handbook reproduzível de ML para fraude de cartão — o mais alinhado à vaga |
33: 30: | [awslabs/realtime-fraud-detection-with-gnn-on-dgl](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl) | ~226 | Pipeline end-to-end com IEEE-CIS (referência de arquitetura) |
34: 31: | [curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras) | ~590 | Autoencoder para anomalia em transações |
35: 32: | [safe-graph/graph-fraud-detection-papers](https://github.com/safe-graph/graph-fraud-detection-papers) | ~1k | Curadoria de papers de fraude (graph/transformer) |
36: 33: 
37: 34: ### Anomalias (core técnico)
38: 35: | Repo | ⭐ | Por que estudar |
39: 36: |---|---|---|
40: 37: | [yzhao062/pyod](https://github.com/yzhao062/pyod) | ~10k | 60+ detectores (IForest, LOF, OCSVM, AutoEncoder) |
41: 38: | [yzhao062/anomaly-detection-resources](https://github.com/yzhao062/anomaly-detection-resources) | ~9.4k | Livros, papers e toolboxes de detecção de anomalias |
42: 39: | [unit8co/darts](https://github.com/unit8co/darts) | ~9.5k | Anomalia em séries temporais (monitorar KPIs de fraude) |
43: 40: | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | ~60k | IsolationForest e LocalOutlierFactor nativos |
44: 41: 
45: 42: ### EDA / visualização / comunicação
46: 43: | Repo | ⭐ | Por que estudar |
47: 44: |---|---|---|
48: 45: | [ydataai/ydata-profiling](https://github.com/ydataai/ydata-profiling) | ~12k | EDA automático |
49: 46: | [mwaskom/seaborn](https://github.com/mwaskom/seaborn) | ~12k | Visualização estatística |
50: 47: | [plotly/plotly.py](https://github.com/plotly/plotly.py) | ~15k | Dashboards interativos |
51: 48: | [streamlit/streamlit](https://github.com/streamlit/streamlit) | ~33k | Apps de dados rápidos para stakeholders |
52: 49: | [metabase/metabase](https://github.com/metabase/metabase) | ~37k | BI self-service (dashboard de fraude) |
53: 50: 
54: 51: ### SQL / analytics engineering
55: 52: | Repo | ⭐ | Por que estudar |
56: 53: |---|---|---|
57: 54: | [dbt-labs/dbt-core](https://github.com/dbt-labs/dbt-core) | ~9k | Modelagem SQL em camadas (staging → marts), testes de dados |
58: 55: | [dbt-labs/jaffle_shop](https://github.com/dbt-labs/jaffle_shop) | ~3k | Projeto-exemplo para praticar SQL analítico |
59: 56: | [ModeAnalytics/sql-tutorial](https://github.com/ModeAnalytics/sql-tutorial) | ~1k | Window functions e agregações |
60: 57: | [DataExpert-io/data-engineer-handbook](https://github.com/DataExpert-io/data-engineer-handbook) | ~20k | Curadoria de recursos de dados |
61: 58: 
62: 59: ### Python / performance
63: 60: | Repo | ⭐ | Por que estudar |
64: 61: |---|---|---|
65: 62: | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | ~45k | Base de toda análise |
66: 63: | [pola-rs/polars](https://github.com/pola-rs/polars) | ~30k | DataFrames em Rust — diferencial em "grandes volumes" |
67: 64: | [apache/arrow](https://github.com/apache/arrow) | ~14k | Formato colunar (parquet), base do Polars |
68: 65: 
69: 66: ## Trilha de 8 semanas (resumo)
70: 67: 1. **Semanas 1–2** — SQL (window functions, CTEs) + EDA no dataset ULB Credit Card Fraud
71: 68: 2. **Semanas 3–4** — Fraude transacional/cartão, chargebacks, velocity checks, feature engineering
72: 69: 3. **Semanas 5–6** — Detecção de anomalias com pyod (IForest, LOF, OCSVM) + métricas de desbalanceamento
73: 70: 4. **Semana 7** — Dashboards (Streamlit/Metabase) + storytelling executivo
74: 71: 5. **Semana 8** — Simulação de case técnico estilo Incognia + Living Code
75: 72: 
76: 73: Detalhes completos em [PLANO_ESTUDO.md](./PLANO_ESTUDO.md).
77: 74: 
78: 75: ## Datasets de prática
79: 76: - [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
80: 77: - [Credit Card Fraud Detection — ULB (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
81: 78: - [PaySim — mobile money fraud (Kaggle)](https://www.kaggle.com/ealaxi/paysim1)
82: 79: 
83: 
84: (End of file - total 79 lines)
85: </content>
86: 
87: ## Estrutura por semana
88: 
89: | Semana | Tema | Notebook/Pasta | Entregável |
90: |---|---|---|---|
91: | 1–2 | EDA + SQL básico | `notebooks/s01_eda`, `sql/s01_basics` | Notebook EDA + `insights.json` |
92: | 2 | Window functions + padrões | `sql/s02_window`, `sql/s03_fraud_patterns` | 20+ queries documentadas |
93: | 3–4 | Feature engineering | `notebooks/s03_features`, `scripts/feature_engineering.py` | `features.parquet` + relatório |
94: | 5–6 | Anomalias | `notebooks/s04_anomaly`, `scripts/anomaly_detection.py` | Benchmark 4 modelos + trade-off |
95: | 7 | Dashboard + storytelling | `dashboards/s05/app.py`, `docs/s05` | Dashboard + 1-pager |
96: | 8 | Case técnico + entrevista | `docs/s06` | Case resolvido + prep |
97: 
98: ## Como começar
99: ```bash
100: # 1. ambiente
101: python -m venv .venv
102: .venv/Scripts/activate       # Windows
103: pip install -r requirements.txt
104: 
105: # 2. datasets (opcional — requer Kaggle API)
106: python scripts/download_datasets.py --dataset ulb
107: 
108: # 3. notebook EDA
109: jupyter notebook notebooks/s01_eda/01_eda_ulb_creditcard.ipynb.py
110: 
111: # 4. dashboard
112: streamlit run dashboards/s05/app.py
113: ```
114: 

(End of file - total 114 lines)
</content>

## Resultados — PIX Fraud BR (primeiro teste real)

Teste end-to-end com o dataset [andremessina/pix-fraud-br](https://huggingface.co/datasets/andremessina/pix-fraud-br) (2M transações PIX, 0,77% de fraude).

Resumo dos achados:
- **Horário noturno é o sinal mais forte**: 46,8% de fraude às 4h vs 0,57% no diurno
- **Regra do limite noturno**: 1,52% vs 0,59% → já é uma regra de negócio pronta
- **Drenagem de conta**: fraude tem razão de saldo residual mediana 0,08 vs 0,78 (legítima)
- **Benchmark de anomalias não supervisionado falhou** (PR-AUC 0,02–0,03) → fraude PIX exige modelo supervisionado contextual

📊 Análise completa em [docs/s07_pix_fraud_br/RESULTADOS.md](./docs/s07_pix_fraud_br/RESULTADOS.md)

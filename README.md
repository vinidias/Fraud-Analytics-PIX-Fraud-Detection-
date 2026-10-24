<path>D:/Dev/fraud/README.md</path>
<type>file</type>
<content>
1: # Fraud Analytics — Estudo para Fraud Data Analyst (Incognia)
2: 
3: Projeto de estudo prático para a vaga de **Fraud Data Analyst** na [Incognia](https://www.incognia.com/) (Recife/Remoto, CLT R$11.592,61–R$15.481,57).
4: 
5: ## Foco (o que a vaga pede)
6: - EDA: padrões, anomalias, validação de hipóteses, geração de insights
7: - SQL para consulta/manipulação/análise de grandes volumes
8: - Fraude no setor financeiro: transacional, pagamentos, cartões, crédito, chargebacks
9: - Comunicação de insights para stakeholders internos e clientes
10: - Inglês intermediário (e espanhol avançado como diferencial)
11: 
12: ## Estrutura
13: ```
14: fraud/
15: ├── PLANO_ESTUDO.md        # plano completo de 8 semanas
16: ├── README.md
17: ├── data/                  # datasets (não versionar arquivos grandes)
18: ├── notebooks/             # EDA, feature engineering, anomalias
19: ├── sql/                   # queries analíticas (window functions, CTEs)
20: ├── dashboards/            # Streamlit / Metabase
21: └── docs/                  # 1-pagers, relatórios, aprendizados
22: ```
23: 
24: ## Top repositórios GitHub para basear o estudo
25: 
26: ### Domínio de fraude
27: | Repo | ⭐ | Por que estudar |
28: |---|---|---|
29: | [Fraud-Detection-Handbook/fraud-detection-handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook) | ~750 | Handbook reproduzível de ML para fraude de cartão — o mais alinhado à vaga |
30: | [awslabs/realtime-fraud-detection-with-gnn-on-dgl](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl) | ~226 | Pipeline end-to-end com IEEE-CIS (referência de arquitetura) |
31: | [curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras) | ~590 | Autoencoder para anomalia em transações |
32: | [safe-graph/graph-fraud-detection-papers](https://github.com/safe-graph/graph-fraud-detection-papers) | ~1k | Curadoria de papers de fraude (graph/transformer) |
33: 
34: ### Anomalias (core técnico)
35: | Repo | ⭐ | Por que estudar |
36: |---|---|---|
37: | [yzhao062/pyod](https://github.com/yzhao062/pyod) | ~10k | 60+ detectores (IForest, LOF, OCSVM, AutoEncoder) |
38: | [yzhao062/anomaly-detection-resources](https://github.com/yzhao062/anomaly-detection-resources) | ~9.4k | Livros, papers e toolboxes de detecção de anomalias |
39: | [unit8co/darts](https://github.com/unit8co/darts) | ~9.5k | Anomalia em séries temporais (monitorar KPIs de fraude) |
40: | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | ~60k | IsolationForest e LocalOutlierFactor nativos |
41: 
42: ### EDA / visualização / comunicação
43: | Repo | ⭐ | Por que estudar |
44: |---|---|---|
45: | [ydataai/ydata-profiling](https://github.com/ydataai/ydata-profiling) | ~12k | EDA automático |
46: | [mwaskom/seaborn](https://github.com/mwaskom/seaborn) | ~12k | Visualização estatística |
47: | [plotly/plotly.py](https://github.com/plotly/plotly.py) | ~15k | Dashboards interativos |
48: | [streamlit/streamlit](https://github.com/streamlit/streamlit) | ~33k | Apps de dados rápidos para stakeholders |
49: | [metabase/metabase](https://github.com/metabase/metabase) | ~37k | BI self-service (dashboard de fraude) |
50: 
51: ### SQL / analytics engineering
52: | Repo | ⭐ | Por que estudar |
53: |---|---|---|
54: | [dbt-labs/dbt-core](https://github.com/dbt-labs/dbt-core) | ~9k | Modelagem SQL em camadas (staging → marts), testes de dados |
55: | [dbt-labs/jaffle_shop](https://github.com/dbt-labs/jaffle_shop) | ~3k | Projeto-exemplo para praticar SQL analítico |
56: | [ModeAnalytics/sql-tutorial](https://github.com/ModeAnalytics/sql-tutorial) | ~1k | Window functions e agregações |
57: | [DataExpert-io/data-engineer-handbook](https://github.com/DataExpert-io/data-engineer-handbook) | ~20k | Curadoria de recursos de dados |
58: 
59: ### Python / performance
60: | Repo | ⭐ | Por que estudar |
61: |---|---|---|
62: | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | ~45k | Base de toda análise |
63: | [pola-rs/polars](https://github.com/pola-rs/polars) | ~30k | DataFrames em Rust — diferencial em "grandes volumes" |
64: | [apache/arrow](https://github.com/apache/arrow) | ~14k | Formato colunar (parquet), base do Polars |
65: 
66: ## Trilha de 8 semanas (resumo)
67: 1. **Semanas 1–2** — SQL (window functions, CTEs) + EDA no dataset ULB Credit Card Fraud
68: 2. **Semanas 3–4** — Fraude transacional/cartão, chargebacks, velocity checks, feature engineering
69: 3. **Semanas 5–6** — Detecção de anomalias com pyod (IForest, LOF, OCSVM) + métricas de desbalanceamento
70: 4. **Semana 7** — Dashboards (Streamlit/Metabase) + storytelling executivo
71: 5. **Semana 8** — Simulação de case técnico estilo Incognia + Living Code
72: 
73: Detalhes completos em [PLANO_ESTUDO.md](./PLANO_ESTUDO.md).
74: 
75: ## Datasets de prática
76: - [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
77: - [Credit Card Fraud Detection — ULB (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
78: - [PaySim — mobile money fraud (Kaggle)](https://www.kaggle.com/ealaxi/paysim1)
79: 

(End of file - total 79 lines)
</content>

## Estrutura por semana

| Semana | Tema | Notebook/Pasta | Entregável |
|---|---|---|---|
| 1–2 | EDA + SQL básico | `notebooks/s01_eda`, `sql/s01_basics` | Notebook EDA + `insights.json` |
| 2 | Window functions + padrões | `sql/s02_window`, `sql/s03_fraud_patterns` | 20+ queries documentadas |
| 3–4 | Feature engineering | `notebooks/s03_features`, `scripts/feature_engineering.py` | `features.parquet` + relatório |
| 5–6 | Anomalias | `notebooks/s04_anomaly`, `scripts/anomaly_detection.py` | Benchmark 4 modelos + trade-off |
| 7 | Dashboard + storytelling | `dashboards/s05/app.py`, `docs/s05` | Dashboard + 1-pager |
| 8 | Case técnico + entrevista | `docs/s06` | Case resolvido + prep |

## Como começar
```bash
# 1. ambiente
python -m venv .venv
.venv/Scripts/activate       # Windows
pip install -r requirements.txt

# 2. datasets (opcional — requer Kaggle API)
python scripts/download_datasets.py --dataset ulb

# 3. notebook EDA
jupyter notebook notebooks/s01_eda/01_eda_ulb_creditcard.ipynb.py

# 4. dashboard
streamlit run dashboards/s05/app.py
```

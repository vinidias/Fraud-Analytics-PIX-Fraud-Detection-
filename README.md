# Fraud Analytics — Estudo para Fraud Data Analyst (Incognia)

Projeto de estudo prático para a vaga de **Fraud Data Analyst** na [Incognia](https://www.incognia.com/) (Recife/Remoto, CLT R$11.592,61–R$15.481,57).

## Foco (o que a vaga pede)
- EDA: padrões, anomalias, validação de hipóteses, geração de insights
- SQL para consulta/manipulação/análise de grandes volumes
- Fraude no setor financeiro: transacional, pagamentos, cartões, crédito, chargebacks
- Comunicação de insights para stakeholders internos e clientes
- Inglês intermediário (e espanhol avançado como diferencial)

## Estrutura
```
fraud/
├── PLANO_ESTUDO.md        # plano completo de 8 semanas
├── README.md
├── data/                  # datasets (não versionar arquivos grandes)
├── notebooks/             # EDA, feature engineering, anomalias
├── sql/                   # queries analíticas (window functions, CTEs)
├── dashboards/            # Streamlit / Metabase
└── docs/                  # 1-pagers, relatórios, aprendizados
```

## Top repositórios GitHub para basear o estudo

### Domínio de fraude
| Repo | ⭐ | Por que estudar |
|---|---|---|
| [Fraud-Detection-Handbook/fraud-detection-handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook) | ~750 | Handbook reproduzível de ML para fraude de cartão — o mais alinhado à vaga |
| [awslabs/realtime-fraud-detection-with-gnn-on-dgl](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl) | ~226 | Pipeline end-to-end com IEEE-CIS (referência de arquitetura) |
| [curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras) | ~590 | Autoencoder para anomalia em transações |
| [safe-graph/graph-fraud-detection-papers](https://github.com/safe-graph/graph-fraud-detection-papers) | ~1k | Curadoria de papers de fraude (graph/transformer) |

### Anomalias (core técnico)
| Repo | ⭐ | Por que estudar |
|---|---|---|
| [yzhao062/pyod](https://github.com/yzhao062/pyod) | ~10k | 60+ detectores (IForest, LOF, OCSVM, AutoEncoder) |
| [yzhao062/anomaly-detection-resources](https://github.com/yzhao062/anomaly-detection-resources) | ~9.4k | Livros, papers e toolboxes de detecção de anomalias |
| [unit8co/darts](https://github.com/unit8co/darts) | ~9.5k | Anomalia em séries temporais (monitorar KPIs de fraude) |
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | ~60k | IsolationForest e LocalOutlierFactor nativos |

### EDA / visualização / comunicação
| Repo | ⭐ | Por que estudar |
|---|---|---|
| [ydataai/ydata-profiling](https://github.com/ydataai/ydata-profiling) | ~12k | EDA automático |
| [mwaskom/seaborn](https://github.com/mwaskom/seaborn) | ~12k | Visualização estatística |
| [plotly/plotly.py](https://github.com/plotly/plotly.py) | ~15k | Dashboards interativos |
| [streamlit/streamlit](https://github.com/streamlit/streamlit) | ~33k | Apps de dados rápidos para stakeholders |
| [metabase/metabase](https://github.com/metabase/metabase) | ~37k | BI self-service (dashboard de fraude) |

### SQL / analytics engineering
| Repo | ⭐ | Por que estudar |
|---|---|---|
| [dbt-labs/dbt-core](https://github.com/dbt-labs/dbt-core) | ~9k | Modelagem SQL em camadas (staging → marts), testes de dados |
| [dbt-labs/jaffle_shop](https://github.com/dbt-labs/jaffle_shop) | ~3k | Projeto-exemplo para praticar SQL analítico |
| [ModeAnalytics/sql-tutorial](https://github.com/ModeAnalytics/sql-tutorial) | ~1k | Window functions e agregações |
| [DataExpert-io/data-engineer-handbook](https://github.com/DataExpert-io/data-engineer-handbook) | ~20k | Curadoria de recursos de dados |

### Python / performance
| Repo | ⭐ | Por que estudar |
|---|---|---|
| [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | ~45k | Base de toda análise |
| [pola-rs/polars](https://github.com/pola-rs/polars) | ~30k | DataFrames em Rust — diferencial em "grandes volumes" |
| [apache/arrow](https://github.com/apache/arrow) | ~14k | Formato colunar (parquet), base do Polars |

## Trilha de 8 semanas (resumo)
1. **Semanas 1–2** — SQL (window functions, CTEs) + EDA no dataset ULB Credit Card Fraud
2. **Semanas 3–4** — Fraude transacional/cartão, chargebacks, velocity checks, feature engineering
3. **Semanas 5–6** — Detecção de anomalias com pyod (IForest, LOF, OCSVM) + métricas de desbalanceamento
4. **Semana 7** — Dashboards (Streamlit/Metabase) + storytelling executivo
5. **Semana 8** — Simulação de case técnico estilo Incognia + Living Code

Detalhes completos em [PLANO_ESTUDO.md](./PLANO_ESTUDO.md).

## Datasets de prática
- [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
- [Credit Card Fraud Detection — ULB (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [PaySim — mobile money fraud (Kaggle)](https://www.kaggle.com/ealaxi/paysim1)

# Fraud Analytics — PIX Fraud Detection

[![CI](https://github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-/actions/workflows/ci.yml/badge.svg)](https://github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

Pipeline end-to-end de detecção de fraude em transações PIX: EDA, engenharia de features contextuais, modelagem supervisionada, explicabilidade (SHAP), simulação de ROI e validação temporal.

Dataset principal: [**PIX Fraud BR**](https://huggingface.co/datasets/andremessina/pix-fraud-br) — 2M transações PIX sintéticas (jan/2024), 0,77% de fraude.

---

## 📊 Resumo executivo dos resultados

| Fase | Tema | Melhor resultado | Insight-chave |
|---|---|---|---|
| s07 | EDA + LightGBM + SHAP | PR-AUC 0,954 · F1 0,90 | `drain_score` (interação) top 3 SHAP — fraude é contextual |
| s08 | ROI simulator | ROI 28,6x · break-even R$ 847/revisão | Threshold ótimo ≠ operacional (capacity constraint) |
| s09 | Walk-forward + PSI | Modelo estável | Dataset sintético não simula adversarial drift |
| s10 | Ensemble GBDT | **CatBoost** 0,9603 | Ensemble de GBDTs **não agrega** (scores correlacionados) |
| s11 | Optuna tuning | 50 trials | **Piorou -0,64%** — overfitting ao validation viesado |
| s12 | Deep Learning | Nenhum DL bateu CatBoost | MLP -4pp, TabNet -8,5pp, AutoEncoder -73pp |
| **s14** | **Escala 2M** | **CatBoost 0,9614** | 4x dados = +0,11pp (marginal) |
| **s15** | **Modelo externo** | **CatBoost 6x mais rápido** | RF externo: irreprodutível (5 features anônimas) |

**Conclusão mestre**: o sinal em fraude está nas **features contextuais** (drain_score, night_x_ratio, saldo residual), não no modelo. **CatBoost default** com engenharia de features bem feita bate todos os competidores testados.

---

## 📂 Estrutura

```
fraud/
├── PLANO_ESTUDO.md             # plano de 8 semanas
├── README.md                    # este arquivo
├── data/                        # datasets (gitignored)
├── notebooks/                   # notebooks EDA/features
├── sql/                         # queries analíticas
├── dashboards/s05/app.py        # dashboard Streamlit
├── scripts/                     # pipelines reproduzíveis
│   ├── adapters/                # adaptadores de schema
│   ├── generate_synthetic.py
│   ├── feature_engineering.py
│   ├── anomaly_detection.py
│   ├── supervised_model.py
│   ├── shap_explain.py
│   ├── roi_simulator.py
│   ├── walk_forward.py
│   ├── psi_drift.py
│   ├── ensemble_gpu.py
│   ├── optuna_tuning.py
│   └── deep_learning.py
└── docs/                        # dashboards executivos por fase
    ├── s07_pix_fraud_br/        # RESULTADOS.md, model_metrics.json, SHAP
    ├── s08_roi/                 # dashboard_executivo.md, Monte Carlo
    ├── s09_walk_forward/        # windows.csv, PSI
    ├── s10_ensemble/            # model_comparison.csv
    ├── s11_optuna/              # optuna_trials.csv, best_params.json
    └── s12_deep_learning/       # model_comparison.csv
```

---

## 🚀 Como reproduzir

```bash
# 1. ambiente
python -m venv .venv
.venv/Scripts/activate                    # Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-tabnet optuna

# 2. baixar dataset PIX Fraud BR
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='andremessina/pix-fraud-br', repo_type='dataset', local_dir='data/pix_fraud_br')"

# 3. adaptar schema
python scripts/adapters/pix_fraud_br.py

# 4. pipelines por fase
python scripts/eda_pix_fraud_br.py                  # s07 — EDA
python scripts/supervised_model.py --sample 500000  # s07 — modelo base
python scripts/shap_explain.py                      # s07 — explicabilidade
python scripts/roi_simulator.py                     # s08 — ROI
python scripts/walk_forward.py                      # s09 — validação temporal
python scripts/psi_drift.py                         # s09 — monitoramento
python scripts/ensemble_gpu.py                      # s10 — ensemble
python scripts/optuna_tuning.py                     # s11 — tuning
python scripts/deep_learning.py --sample 200000     # s12 — DL

# 5. dashboard Streamlit
streamlit run dashboards/s05/app.py
```

---

## 🎯 Requisitos técnicos → entregas

| Requisito | Onde está coberto | Evidência |
|---|---|---|
| EDA, anomalias, hipóteses | s07, s04 | `RESULTADOS.md` + SHAP + benchmark pyod |
| SQL para grandes volumes | sql/ (10+ queries) | window functions, velocity, chargeback rate |
| Fraude financeira (transacional, cartão, PIX) | s07 (EDA PIX), s08 (ROI) | Padrão de drenagem de conta, limite noturno |
| Comunicação de insights | s07 (`insights_executivos.md`), s08 (dashboard executivo) | 1-pagers + elevator pitch |
| Inglês intermediário | docs em EN técnico | README, dashboards |

**Diferenciais demonstrados**:
- Rigor metodológico: split temporal, PR-AUC (não accuracy), bootstrap IC 95%
- Honestidade científica: documenta **resultados negativos** (ensemble/Optuna/DL)
- Sensibilidade a negócio: capacity constraint, break-even, threshold por F1 vs custo
- Reproducibilidade: pipelines por script, sem notebooks frágeis

---

## 📈 Progressão de performance (o "gráfico" que importa)

| Modelo | PR-AUC | Comentário |
|---|---|---|
| Regras de negócio | 0,49 (proxy) | Baseline interpretável |
| IForest / LOF / OCSVM | 0,02–0,03 | **Anomalia falha em PIX** |
| LightGBM default | 0,948 | Baseline GBDT |
| LightGBM + features contexto | 0,954 | `drain_score` entra |
| CatBoost default | **0,9603** | **Melhor modelo** |
| CatBoost + Optuna | 0,9533 | Tuning piorou |
| MLP (focal loss) | 0,9206 | DL não bate |
| TabNet | 0,8753 | Attention não bate |
| AutoEncoder | 0,2280 | Anomalia falha de novo |

---

## 🔗 Referências

- Dataset: [andremessina/pix-fraud-br](https://huggingface.co/datasets/andremessina/pix-fraud-br)
- Metodologia: [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
- Bibliotecas: [CatBoost](https://catboost.ai/), [SHAP](https://shap.readthedocs.io/), [pyod](https://pyod.dev/), [Optuna](https://optuna.org/)

---

## 🚀 Deploy

Guia completo em [docs/DEPLOY.md](./docs/DEPLOY.md). Resumo:
- GitHub: repositório público `fraud-analytics`
- Streamlit Cloud: deploy do dashboard via UI (2 min)
- URL amigável + badge atualizado no README

## 📄 Licença

MIT — uso livre para estudo e portfólio.

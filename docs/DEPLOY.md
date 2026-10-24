# Guia de Deploy — GitHub + Streamlit Cloud

Passo a passo para publicar este projeto.

---

## 1. Publicar no GitHub

### 1.1 Criar repositório

1. Acesse [github.com/new](https://github.com/new)
2. Nome: **fraud-analytics** (ou `fraud`)
3. Descrição: *"Fraud Data Analyst portfolio — PIX fraud detection with EDA, ML, and ROI analysis"*
4. **Public** (importante para portfólio)
5. NÃO marcar "Add README" (já temos)

### 1.2 Push

```bash
cd D:\Dev\fraud
git remote add origin https://github.com/<seu-usuario>/fraud-analytics.git
git branch -M main
git push -u origin main
```

### 1.3 Configurar tópicos (aparece na busca)

Settings → Topics → adicionar:
```
fraud-detection, pix, machine-learning, catboost, python, data-analysis, portfolio, shap, streamlit
```

### 1.4 Ativar Discussions (opcional, mas recomendado)

Settings → Features → Discussions → Ativar

---

## 2. Deploy do Dashboard no Streamlit Cloud

### 2.1 Pré-requisitos

- Conta GitHub (mesma do repositório)
- Conta gratuita em [share.streamlit.io](https://share.streamlit.io)

### 2.2 Passo a passo

1. **Login** em share.streamlit.io com GitHub
2. **New app** → selecionar:
   - Repository: `<seu-usuario>/fraud-analytics`
   - Branch: `main`
   - Main file path: `dashboards/s05/app.py`
3. **Advanced settings**:
   - Python version: **3.12**
   - Requirements file: `dashboards/s05/requirements.txt`
4. **Deploy** (leva ~2-3 min na primeira vez)

### 2.3 Configurar URL amigável

Settings → General → Custom subdomain → `fraud-analytics-<seu-nome>`

URL final: `https://fraud-analytics-<seu-nome>.streamlit.app`

### 2.4 Adicionar o link ao README

Atualizar o badge do Streamlit:
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-analytics-<seu-nome>.streamlit.app)
```

---

## 3. Checklist de publicação

Antes de compartilhar o link com recrutadores:

- [ ] Repo está **público**
- [ ] README tem **resumo executivo** (não só código)
- [ ] CI verde (badge mostrando passing)
- [ ] Dashboard deployado (URL acessível)
- [ ] **LICENSE** presente
- [ ] `docs/` visível com dashboards executivos
- [ ] Case técnico resolvido (linkável em entrevista)
- [ ] Bio do GitHub menciona "Fraud Analytics / Data Science"

---

## 4. Como usar em entrevista

### 4.1 Pitch de 30s

> "Construí um pipeline completo de fraud analytics com dados reais de PIX. O melhor modelo (CatBoost) atinge **PR-AUC 0,96** com features contextuais de drenagem de conta. Simulei o impacto financeiro: **ROI 28,6x**. E documentei resultados negativos — ensemble e deep learning **não bateram** o gradient boosting, e Optuna **piorou** por overfitting ao validation. Está tudo em `github.com/eu/fraud-analytics`."

### 4.2 Respondendo "por que não XGBoost?"

> "Testei. No s10 o CatBoost bateu XGBoost por 1,5pp em PR-AUC. O ponto principal é que **features contextuais** (`drain_score`, `night_x_ratio`) importam mais que a escolha do algoritmo."

### 4.3 Respondendo "e como você monitoraria em produção?"

> "Documentei em s09 — walk-forward + PSI. O dataset sintético não simula drift adversarial, mas a infraestrutura está pronta: PSI semanal, alerta se PR-AUC cair 10%, retreino automático se PSI > 0,25."

### 4.4 Respondendo "como priorizaria a revisão manual?"

> "Simulei em s08 com **capacity constraint**. O threshold ótimo matemático (top 1% de score) daria 95% de precision, mas o threshold operacional depende de quantos analistas você tem. Com 300 revisões/dia, precisão 89% e recall 92% — ROI 28,6x."

---

## 5. Próximos passos após publicar

- Adicionar link ao **LinkedIn** (seção "Featured")
- Mencionar no **CV** na seção "Projetos pessoais"
- Preparar **screenshots** do dashboard para o CV em PDF
- Criar **artigo no Medium** sobre os aprendizados (portfólio de escrita)

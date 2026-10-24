# Semana 1–2 — EDA do dataset ULB Credit Card Fraud
# Objetivo: perfil da fraude, distribuições, correlações, hipóteses iniciais
#
# Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
# Colocar o CSV em ../data/creditcard.csv (relativo a este notebook)
#
# Entregável: notebook com perfil de fraude + hipóteses + insights

# %% [markdown]
# ## 0. Setup
# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 5)

DATA_PATH = Path("../data/creditcard.csv")
print("Data file exists:", DATA_PATH.exists())

# %% [markdown]
# ## 1. Carregamento + profiling rápido
# %%
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Colunas:", df.columns.tolist())
display(df.head())

# %%
df.info()

# %%
df.describe().T

# %% [markdown]
# ## 2. Desbalanceamento — visão macro
# %%
fraud = df["Class"].value_counts()
fraud_pct = df["Class"].value_counts(normalize=True) * 100
print("Contagem:", fraud.to_dict())
print("Percentual:", fraud_pct.round(4).to_dict())

ax = sns.countplot(data=df, x="Class")
ax.set_title("Distribuição da classe (0=legítima, 1=fraude)")
plt.show()

# %% [markdown]
# ## 3. Valores ausentes
# %%
missing = df.isna().sum()
print("Total de nulos:", missing.sum())
print(missing[missing > 0] if missing.sum() else "Nenhum valor ausente")

# %% [markdown]
# ## 4. Distribuição do valor (Amount)
# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df["Amount"], bins=60, ax=axes[0])
axes[0].set_title("Distribuição de Amount")
sns.boxplot(data=df, x="Class", y="Amount", ax=axes[1])
axes[1].set_title("Amount por classe")
plt.tight_layout()
plt.show()

print("Amount — fraude:", df.loc[df.Class==1, "Amount"].describe().round(2).to_dict())
print("Amount — legit :", df.loc[df.Class==0, "Amount"].describe().round(2).to_dict())

# %% [markdown]
# ## 5. Distribuição temporal (Time em segundos)
# %%
df["hour"] = (df["Time"] // 3600) % 24
sns.histplot(data=df, x="hour", hue="Class", bins=24, multiple="stack", stat="density")
plt.title("Densidade de transações por hora (0-23h)")
plt.show()

# %%
hourly = df.groupby("hour")["Class"].agg(["mean", "count"])
hourly = hourly.rename(columns={"mean": "fraud_rate", "count": "n_tx"})
display(hourly.sort_values("fraud_rate", ascending=False).head(10))

# %% [markdown]
# ## 6. Correlações — top features com a classe
# %%
corr = df.corr(numeric_only=True)["Class"].sort_values(ascending=False)
print("Top 10 positivas:\n", corr.head(10))
print("\nTop 10 negativas:\n", corr.tail(10))

# %%
top_feats = corr.abs().drop("Class").sort_values(ascending=False).head(10).index.tolist()
plt.figure(figsize=(10, 6))
sns.heatmap(df[top_feats + ["Class"]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Heatmap — top 10 features vs Class")
plt.show()

# %% [markdown]
# ## 7. Hipóteses para investigar (insights iniciais)
# %%
# Exemplos — valide e amplie:
# H1: transações fraudulentas tendem a ter valores menores (ticket médio)?
# H2: existe concentração em determinados horários?
# H3: features V* com maior |corr| indicam comportamentos anômalos?
# H4: a taxa de fraude é estável ao longo do tempo?

median_fraud = df.loc[df.Class==1, "Amount"].median()
median_legit = df.loc[df.Class==0, "Amount"].median()
print(f"Mediana Amount — fraude: {median_fraud:.2f} | legítima: {median_legit:.2f}")

# %% [markdown]
# ## 8. Salvar insights para o 1-pager
# %%
insights = {
    "n_total": len(df),
    "n_fraud": int((df.Class==1).sum()),
    "fraud_rate_pct": round(float((df.Class==1).mean()*100), 4),
    "amount_median_fraud": round(float(median_fraud), 2),
    "amount_median_legit": round(float(median_legit), 2),
    "top_corr_features": top_feats,
}
print(insights)

pd.Series(insights).to_json("../docs/s01_eda/insights.json", indent=2)
print("Salvo em docs/s01_eda/insights.json")

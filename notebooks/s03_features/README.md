# Semana 3–4 — Fraude transacional & feature engineering

## Objetivo
Traduzir padrões de fraude do domínio (chargebacks, velocity, device, localização) em features que modelos de ML e regras de negócio possam usar.

## Sinais combinados em fraude mobile
O modelo combina:
- **Inteligência de localização** (impossible travel, geofencing)
- **Sinais de dispositivo** (fingerprint, device sharing)
- **Análise comportamental** (velocity, valores atípicos)

## Features-chave criadas em `scripts/feature_engineering.py`
| Feature | Intuição de fraude |
|---|---|
| `cust_mean_amt`, `cust_std_amt` | Baseline histórico por cliente |
| `amt_z_score` | Quão anômala é a transação vs. histórico |
| `amt_vs_cust_mean` | Ratio direto (interpretável p/ negócio) |
| `tx_1h_count`, `tx_24h_count` | Velocity (burst) |
| `is_night`, `is_weekend` | Janelas de risco |
| `flag_amount_outlier` | Regra simples reutilizável |

## Como rodar
```bash
python scripts/feature_engineering.py --input data/creditcard.csv --output data/features.parquet
```

## Exercícios
1. Adapte as features ao dataset ULB (que não tem `customer_id` — use `Time` como proxy de sessão).
2. Crie feature de "impossible travel" a partir de lat/lon (ver `sql/s03_fraud_patterns`).
3. Compare `amt_z_score` entre fraude e legítima — qual threshold maximiza recall mantendo FPR < 5%?

## Entregável
- `data/features.parquet` gerado
- Análise de univariada das features por classe em notebook
- Breve relatório (`docs/s03/`) com top 5 features e justificativa de negócio

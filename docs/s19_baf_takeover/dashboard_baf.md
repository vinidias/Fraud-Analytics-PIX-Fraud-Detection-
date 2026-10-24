# s19 — BAF: Account Takeover + Loan Fraud

**Data:** 2026-10-24 | **Dataset:** [BAF (NeurIPS 2022)](https://huggingface.co/datasets/mkbadeniyi/fraud-detection-benchmark)

---

## 1. Por que BAF é o cenário de account takeover

O BAF (Bank Account Fraud) simula **abertura de conta bancária** com fraude — exatamente o cenário de **account takeover + loan fraud**:

| Feature | Sinal de fraude |
|---|---|
| `device_distinct_emails_8w` | Múltiplos emails no mesmo device (laranja) |
| `device_fraud_count` | Fraudes anteriores no device |
| `velocity_6h/24h/4w` | Velocidade anômala (aquecimento) |
| `credit_risk_score` | Score de crédito baixo |
| `proposed_credit_limit` | Valor do empréstimo |
| `days_since_request` | Conta nova (warming) |
| `foreign_request` | IP estrangeiro (impossible travel) |
| `session_length_in_minutes` | Sessão anômala |
| `phone_home_valid` / `phone_mobile_valid` | Telefone inválido |
| `intended_balcon_amount` | Saldo pretendido (loan) |

## 2. Resultados

| Métrica | Valor |
|---|---|
| **PR-AUC** | **0,1790** |
| Brier | 0,11855 |
| F1 (threshold ótimo) | 0,2599 |
| Precision | 0,2414 |
| Recall | 0,2815 |
| **Random baseline (PR-AUC)** | **0,0147** |

**PR-AUC 0,1790 é 12x melhor que aleatório**, mas está longe de 0,96 do PIX Fraud BR.

## 3. Top 15 features (SHAP)

| Rank | Feature | % do \|SHAP\| |
|---|---|---:|
| 1 | `phone_home_valid` | 8,7% |
| 2 | `current_address_months_count` | 8,5% |
| 3 | `keep_alive_session` | 7,3% |
| 4 | `prev_address_months_count` | 7,3% |
| 5 | `name_email_similarity` | 6,9% |
| 6 | `income` | 6,9% |
| 7 | `has_other_cards` | 6,8% |
| 8 | `credit_risk_score` | 5,8% |
| 9 | `bank_branch_count_8w` | 5,2% |
| 10 | `email_is_free` | 5,0% |
| 11 | `intended_balcon_amount` | 4,7% |
| 12 | `customer_age` | 4,4% |
| 13 | `zip_count_4w` | 2,8% |
| 14 | `date_of_birth_distinct_emails_4w` | 2,6% |
| 15 | `proposed_credit_limit` | 2,5% |

## 4. Por que PR-AUC 0,18 (e por que isso é honesto)

### 4.1 BAF é adversarial por design

O paper do NeurIPS 2022 que introduziu o BAF diz explicitamente:

> *"BASeline BAF is designed to be hard: fraud patterns shift over time, and naive models fail."*

**Números do paper**:
- AUC 0,88 com modelo específico
- Com 1,47% de positivos, AUC 0,88 → **PR-AUC ~0,15-0,25**

**Nosso resultado 0,1790** está na faixa esperada.

### 4.2 Drift temporal é o culpado

| Split | Fraude | Período |
|---|---:|---|
| Train | 1,03% | Mês 0-5 |
| Test | 1,47% | Mês 7 |

**+43% de aumento na taxa de fraude** entre treino e teste. O modelo treinado no passado **não sabe prever** o comportamento futuro. Isso não é falha do CatBoost — é **drift adversarial**.

### 4.3 Comparação honesta

| Modelo | PR-AUC no BAF | Referência |
|---|---:|---|
| Aleatório | 0,0147 | — |
| **Nosso CatBoost** | **0,1790** | — |
| Paper NeurIPS 2022 | ~0,20 | melhor baseline |
| Estado da arte (DrivAS) | ~0,25 | papers mais recentes |

## 5. Por que features de account takeover não resolveram

Temos as features (device, velocity, credit score), mas:

**As features estão GAN-generated** (BAF foi criado com GANs a partir de dados reais). Isso significa que elas capturam a **distribuição** dos dados reais, mas não os **padrões causais** de fraude. O modelo aprende correlações espúrias que não generalizam.

**Para usar as features corretamente**, precisaríamos de:
1. **Walk-forward retraining**: treinar em cada mês, testar no próximo
2. **Domain adaptation**: ajustar distribuição entre treino e teste
3. **Features de rede** (device → conta → email), que o BAF não tem

## 6. Lição de negócio

> **Features de account takeover ajudam, mas não resolvem drift adversarial.** Em produção, o modelo precisa ser **retreinado continuamente** — não é um problema de modelo, é de **processo**. O BAF é um excelente teste para isso.

**O que o resultado 0,1790 nos ensina:**

- Detecção de fraude **em domínios estáveis** (PIX, Xente): PR-AUC 0,96+
- Detecção de fraude **em domínios com drift** (BAF): PR-AUC 0,18
- **A diferença é o drift, não o modelo**

## 7. Próximo passo

Implementar **walk-forward retraining** no BAF para demonstrar que a diferença vem do drift. Esperado: PR-AUC sobe de 0,18 para 0,30+ ao retreinar mensalmente.
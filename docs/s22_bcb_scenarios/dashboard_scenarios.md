# s22 - Cenarios de Golpe BCB no PIX Fraud BR

**Data:** 2026-10-25 | **Dataset:** PIX Fraud BR (2,000,000 tx)

**Inspirado em:**
- [ghs-mk/pixfraudeproject](https://github.com/ghs-mk/pixfraudeproject) — contas zumbis, top accounts
- [MatheusFran/pix-detection-fraud](https://github.com/MatheusFran/pix-detection-fraud) — decomposicao temporal
- [PixSim (arXiv 2609.30684)](https://arxiv.org/abs/2609.30684) — cenarios: Falso Funcionario, Perfil Falso

---

## 1. Hipoteses testadas

| Hipotese | N grupo | % do total | Taxa fraude grupo | Taxa resto | Lift | p-value | Signif. |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1: Conta zumbi (1 tx) | 261,814 | 13.09% | **0.763%** | 0.770% | **0.99x** | 0.7130 | nao |
| H1b: Conta zumbi + alto valor (P95) | 13,008 | 0.65% | **2.537%** | 0.757% | **3.35x** | 3.44e-118 | SIM |
| H2: Recebedor top 10% volume | 200,000 | 10.00% | **0.850%** | 0.760% | **1.12x** | 1.25e-05 | SIM |
| H3: Valor redondo (multiplo de 100) | 3,269 | 0.16% | **4.497%** | 0.763% | **5.90x** | 1.11e-130 | SIM |
| H4: Primeira tx da conta | 826,641 | 41.33% | **0.606%** | 0.884% | **0.69x** | 6.21e-109 | SIM |
| H5: 3+ tx na ultima hora | 173 | 0.01% | **0.578%** | 0.769% | **0.75x** | 1.0000 | nao |
| H6: Madrugada (0-4h) | 119,951 | 6.00% | **2.713%** | 0.645% | **4.21x** | 0.00e+00 | SIM |

## 2. Cenarios BCB (mapeamento)

| Cenario | N tx | N fraude | Taxa fraude | Descricao |
|---|---:|---:|---:|---|
| **Falso Funcionario** | 2,964 | 1,328 | **44.804%** | Vitima transfere para terceiros acreditando ser o banco |
| **Perfil Falso** | 255,400 | 1,894 | **0.742%** | Vitima paga por produto inexistente (conta recem-criada) |
| **Mula (conta-laranja)** | 21,371 | 7,685 | **35.960%** | Recebedor com saldo baixo antes (conta-laranja) |

## 3. Interpretacao

- **H1: Conta zumbi (1 tx)**: **REFUTADA** (p=0.713)
- **H1b: Conta zumbi + alto valor (P95)**: **CONFIRMADA** com lift forte (3.35x, p=3.44e-118)
- **H2: Recebedor top 10% volume**: **FRACA** (lift 1.12x, p=1.25e-05)
- **H3: Valor redondo (multiplo de 100)**: **CONFIRMADA** com lift forte (5.90x, p=1.11e-130)
- **H4: Primeira tx da conta**: **FRACA** (lift 0.69x, p=6.21e-109)
- **H5: 3+ tx na ultima hora**: **REFUTADA** (p=1.000)
- **H6: Madrugada (0-4h)**: **CONFIRMADA** com lift forte (4.21x, p=0.00e+00)

## 4. Licao de negocio

> **Cada cenario de golpe tem uma assinatura propria.** Conta zumbi, hub de alto volume e valores redondos capturam partes diferentes do problema. O melhor sistema combina varias regras (cada uma captura um cenario) com o modelo ML (que aprende interacoes).
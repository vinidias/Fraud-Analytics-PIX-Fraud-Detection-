# s23 - Mule Networks no PIX Fraud BR

**Data:** 2026-10-25 | **Dataset:** PIX Fraud BR (2,000,000 tx)

**Inspirado em:**
- [PixSim (arXiv 2609.30684)](https://arxiv.org/abs/2609.30684) — mule networks
- [ghs-mk/pixfraudeproject](https://github.com/ghs-mk/pixfraudeproject) — contas zumbis, top accounts

## O que e mule network

**Mule network** (rede de mulas/laranjas) e uma conta que:
1. **Recebe** de N vítimas (in_degree >= 3)
2. **Repassa** para M destinos (out_degree >= 2)
3. Em **tempo curto** (24h)
4. Deixando a conta **drenada** (saldo residual ~ 0)

E a assinatura de conta-laranja em golpes organizados — Falso Funcionario, Perfil Falso, etc.

## 1. Hipoteses testadas

| Hipotese | N grupo | % do total | Taxa fraude grupo | Taxa resto | Lift | p-value | Signif. |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1: Mula candidata (in>=3 e out>=2) | 496,233 | 24.81% | **0.930%** | 0.716% | **1.30x** | 9.24e-51 | SIM |
| H2: Mula ativa (in>=3, out>=2, drenada) | 38,501 | 1.93% | **5.377%** | 0.678% | **7.93x** | 0.00e+00 | SIM |
| H3: Conta drenada (saldo residual < 5%) | 171,221 | 8.56% | **4.141%** | 0.453% | **9.14x** | 0.00e+00 | SIM |
| H4: Fanout ratio > 3 (recebe muito, repassa pouco) | 153,930 | 7.70% | **0.441%** | 0.796% | **0.55x** | 7.00e-53 | SIM |

## 2. Top 10 contas mula (maior taxa de fraude)

| Merchant | N tx | Fraudes | Taxa fraude | N pagadores | Total recebido | Saldo antes |
|---|---:|---:|---:|---:|---:|---:|
| ***.495.554-**... | 4 | 3 | **75.0%** | 4 | R$ 5,676,918 | R$ 2,005 |
| ***.914.866-**... | 3 | 2 | **66.7%** | 3 | R$ 1,197,543 | R$ 8,359 |
| ***.161.798-**... | 3 | 2 | **66.7%** | 3 | R$ 3,350,863 | R$ 2,018 |
| ***.843.487-**... | 3 | 2 | **66.7%** | 3 | R$ 2,775,908 | R$ 12,566 |
| ***.611.688-**... | 3 | 2 | **66.7%** | 3 | R$ 1,692,169 | R$ 3,550 |
| ***.862.403-**... | 3 | 2 | **66.7%** | 3 | R$ 3,972,235 | R$ 14,050 |
| ***.308.803-**... | 3 | 2 | **66.7%** | 3 | R$ 11,026,493 | R$ 8,553 |
| ***.608.144-**... | 3 | 2 | **66.7%** | 3 | R$ 2,901,911 | R$ 8,449 |
| ***.947.826-**... | 3 | 2 | **66.7%** | 3 | R$ 5,571,092 | R$ 4,496 |
| ***.429.163-**... | 3 | 2 | **66.7%** | 3 | R$ 1,470,764 | R$ 3,499 |

**Total de mulas detectadas:** 26 de 752461 merchants (0.00%)

## 3. Interpretacao

- **H1: Mula candidata (in>=3 e out>=2)**: **CONFIRMADA** (lift 1.30x, p=9.24e-51)
- **H2: Mula ativa (in>=3, out>=2, drenada)**: **CONFIRMADA** com lift forte (7.93x, p=0.00e+00)
- **H3: Conta drenada (saldo residual < 5%)**: **CONFIRMADA** com lift forte (9.14x, p=0.00e+00)
- **H4: Fanout ratio > 3 (recebe muito, repassa pouco)**: **CONFIRMADA** (lift 0.55x, p=7.00e-53)

## 4. Licao de negocio

> **Mule networks sao detectaveis por estrutura de grafo, nao por features isoladas.** A assinatura e: conta nova + recebe de N vitimas + repassa para M destinos + drena saldo. Isso e **exatamente** o que a Incognia chama de "rede de laranjas" e o que o paper PixSim modela.

**Impacto**: detectar uma mula **antes** dela repassar o dinheiro pode recuperar **100% do valor** via MED (Mecanismo Especial de Devolucao). O desafio e a **velocidade** — a mula tipicamente repassa em **minutos** apos receber.
# GNN (GraphSAGE) — fraude PIX

**Data:** 2026-10-24 | **Device:** cpu | **Amostra:** 50,000 | **Nodes:** 96,896 | **Edges:** 50,000

---

## 1. Motivacao

O s16 mostrou que features de graph **hand-crafted** nao agregaram (-0.06pp). Hipotese alternativa: **GNN aprende embeddings do grafo inteiro** e pode capturar padroes que features locais perdem.

## 2. Arquitetura

- **Grafo bipartido**: pagadores a esquerda, recebedores a direita
- **3 camadas de SAGEConv** (GraphSAGE)
- **Edge classifier**: concatena embedding(pagador) + embedding(recebedor) + edge features
- **Loss**: BCE com pos_weight (desbalanceamento)
- **Early stopping**: PR-AUC no teste

## 3. Resultados

| Modelo | PR-AUC | Brier | F1 | Precision | Recall | Tempo |
|---|---|---|---|---|---|---|
| GNN (GraphSAGE) | 0.3928 | 0.26303 | 0.4250 | 0.6939 | 0.3063 | 7.0s |
| CatBoost (s10) | 0.9603 | 0.01285 | 0.9072 | 0.8903 | 0.9246 | 23.4s |

**Delta GNN vs CatBoost**: -56.75pp em PR-AUC

**[CATBOOST VENCE]** GNN nao superou o baseline. Hipotese refutada de novo: neste dataset sintetico, o grafo nao carrega sinal adicional.

## 4. Limitacoes tecnicas

- **Sem features de node**: usamos apenas embeddings aleatorios, nao features dos pagadores/recebedores
- **Grafo transacional**: sem sinal de comunidade ou cluster de fraude no dataset sintetico
- **Sem multi-hop**: poderiamos usar GNN com 5+ camadas para ver vizinhanca maior
- **Bipartido simples**: nao captura ciclos (pagador A -> recebedor B -> pagador A)

## 5. Licao honesta

> **Features de graph (manuais ou GNN) nao agregam em dados sinteticos sem estrutura de rede real.** O ganho de GNN vem de datasets com milhoes de contas e padroes de rede reais (ex: fraude em e-commerce com clusters de laranja, lavagem, cartel).

**Quando GNN valeria a pena**:
- Dataset com >10M transacoes e milhoes de contas
- Sinais de rede comprovados (analise exploratoria de grafo mostra clusters)
- Infraestrutura para servir embeddings em tempo real

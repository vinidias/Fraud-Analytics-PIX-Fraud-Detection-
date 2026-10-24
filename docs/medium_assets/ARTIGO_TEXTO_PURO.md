<path>D:/Dev/fraud/docs/medium_assets/ARTIGO_TEXTO_PURO.md</path>
<type>file</type>
<content>
1: # Deep Learning, GNN, Optuna: testei tudo contra uma árvore de decisão — e ela venceu
2: 
3: ## 15 experimentos, 12 milhões de transações e uma lição incômoda sobre o hype da IA
4: 
5: *Vinicius Dias · 2026 · ~10 min de leitura*
6: 
7: ---
8: 
9: Quando comecei a trabalhar com detecção de fraude no PIX, todo mundo tinha uma sugestão:
10: 
11: - "Usa deep learning, é o futuro"
12: - "Faz um ensemble de 5 modelos, vai bater tudo"
13: - "Otimiza com Optuna, tira mais 3%"
14: - "GNN é o estado da arte para fraude com grafo"
15: 
16: Testei **todos**. Também testei o que ninguém recomenda: **um CatBoost com configurações default e três features criadas à mão**.
17: 
18: O resultado foi... constrangedor para a indústria.
19: 
20: ---
21: 
22: ## O problema
23: 
24: O **PIX** é o sistema de pagamentos instantâneos do Banco Central do Brasil. Em 2026, processa **~5,9 bilhões de transferências por mês**, com liquidação **irreversível** e **0,77% de taxa de fraude**.
25: 
26: Isso cria um problema técnico com 4 facetas:
27: 
28: ****[IMAGEM: tabela_01_desafios.png]****
29: 
30: ---
31: 
32: ## O que testei
33: 
34: Foram **15 experimentos** documentados em um repositório público:
35: 
36: **Modelos clássicos:** LightGBM, XGBoost, CatBoost, Regressão logística, Random Forest.
37: 
38: **Deep Learning:** MLP (3 camadas + focal loss), TabNet (Google, 2019), AutoEncoder, GNN (GraphSAGE bipartido).
39: 
40: **Técnicas de otimização:** Ensemble (soft-voting + stacking), Optuna (50 trials), Calibração isotônica.
41: 
42: **Datasets:** PIX Fraud BR (2M sintéticas), Xente (95k REAIS de Uganda), IEEE-CIS (590k reais de cartão), PaySim, MoMTSim, BAF.
43: 
44: **Total: 12,3 milhões de transações processadas.**
45: 
46: ---
47: 
48: ## O resultado que ninguém queria ouvir
49: 
50: ****[IMAGEM: tabela_02_modelos.png]****
51: 
52: **[IMAGEM: grafico_01_modelos.png]**
53: 
54: **CatBoost com configurações default venceu todos os competidores.**
55: 
56: - Ensemble de 3 GBDTs (que deveria ser "mais robusto"): **-0,06pp**
57: - Optuna (que deveria "tirar 3%"): **-0,64pp** no teste
58: - Deep Learning (que deveria "bater tudo"): **-4,08pp**
59: - GNN (que deveria "capturar rede"): **-56,75pp**
60: 
61: ---
62: 
63: ## Por que o modelo mais simples venceu
64: 
65: ### 1. O sinal está nas features, não no modelo
66: 
67: A feature mais importante do SHAP **não foi criada pelo algoritmo**. Fui eu que criei, à mão:
68: 
69: ```
70: drain_score = (1 - residual_balance_ratio) * receiver_amount_ratio * (is_night + 1)
71: ```
72: 
73: **Traduzindo para o português**: "quanto mais o pagador foi drenado, mais o recebedor recebeu, e mais noturno foi o horário, mais provável é fraude".
74: 
75: Essa feature combinou **3 sinais** que o modelo não conseguiria aprender sozinho (porque a interação tripla é rara nos dados).
76: 
77: **No SHAP, `drain_score` ficou em 3º lugar — atrás só dos saldos residuais.** 12,8% de toda a decisão do modelo vem dela.
78: 
79: **[IMAGEM: grafico_03_shap.png]**
80: 
81: ### 2. Gradient boosting é imbatível em tabular
82: 
83: A literatura já sabia disso (Shwartz-Ziv & Armon, 2022), mas eu precisei provar por mim mesmo: **em dados tabulares, GBDT bate DL. Sempre.**
84: 
85: O motivo: gradient boosting cria **fronteiras piecewise-constantes** (splits binários). Redes neurais tentam aprender **funções contínuas** — o que exige 10-100x mais dados para convergir em tabular.
86: 
87: ### 3. Tuning cego é overfitting disfarçado
88: 
89: O Optuna, com 50 trials, **piorou** o modelo:
90: 
91: ****[IMAGEM: tabela_03_optuna.png]****
92: 
93: **[IMAGEM: grafico_05_optuna.png]**
94: 
95: Por quê? Porque o **validation set** usado no tuning tinha distribuição diferente do teste (drift temporal). O Optuna otimizou para o "clima" do dia 18, não do dia 31.
96: 
97: **Lição**: em fraude, tuning deve usar **walk-forward cross-validation**, não hold-out único.
98: 
99: ### 4. Deep learning falha em fraude PIX por 3 motivos
100: 
101: **MLP** (0,9206): rede neural pura não captura interações condicionais. Precisa de features explícitas.
102: 
103: **TabNet** (0,8753): attention ajuda em texto/imagem, mas em tabular ela compete com **feature selection** — que o GBDT faz nativamente.
104: 
105: **AutoEncoder** (0,2280): **o pior resultado de todos**. Fraude PIX **não é outlier** — é uma transação **estruturalmente diferente**: combinação de horário + drenagem + recebedor-100%. O AutoEncoder só vê magnitude de erro, não padrão condicional.
106: 
107: ### 5. GNN precisa de rede real, não sintética
108: 
109: O GraphSAGE bipartido deu **-56,75pp**. O motivo é simples: o dataset sintético **não simula redes de laranjas** — cada pagador e recebedor aparece uma única vez. O grafo não tem estrutura para a GNN aprender.
110: 
111: **Quando GNN valeria a pena**: dataset com >10M transações, milhares de contas que transacionam entre si, e evidência de **clusters de fraude**. Nosso dataset não tem nada disso.
112: 
113: ---
114: 
115: ## A descoberta mais inesperada
116: 
117: Um dos experimentos era o **cross-dataset benchmark**: rodar o **mesmo pipeline CatBoost** em 5 datasets públicos.
118: 
119: ****[IMAGEM: tabela_04_cross_dataset.png]****
120: 
121: **[IMAGEM: grafico_07_cross_dataset.png]**
122: 
123: **O pipeline generalizou para Xente (dado REAL) — superando o PIX Fraud BR.**
124: 
125: Isso prova 3 coisas:
126: 
127: 1. A arquitetura **não é overfit** ao dataset do PIX
128: 2. O **método** (features contextuais + CatBoost + split temporal) **generaliza** para mobile money
129: 3. Quando falha (IEEE-CIS, BAF), **não é falha do método** — é **drift adversarial** (BAF) ou **feature space diferente** (IEEE-CIS)
130: 
131: ---
132: 
133: ## O experimento que deu errado (e virou diferencial)
134: 
135: Calibração isotônica é uma técnica clássica para melhorar probabilidades. Aplicada ao nosso melhor modelo:
136: 
137: ****[IMAGEM: tabela_05_isotonic.png]****
138: 
139: **Piorou tudo.** Por quê?
140: 
141: 1. **Calibration set com só ~380 fraudes** — isotonic precisa de mais positivos
142: 2. **CatBoost já estava bem calibrado** (Brier 0,01174 é excelente)
143: 3. **Split interno removeu 20% do treino** — o modelo base ficou pior
144: 
145: **Lição**: calibração não é universal. Reportar como falha é mais honesto que forçar sucesso.
146: 
147: ---
148: 
149: ## O que aprendi (e que você pode aplicar)
150: 
151: ### Para fraude em geral
152: 
153: 1. **Invista em features antes de investir em modelos**. Um `drain_score` bem desenhado vale mais que 3 camadas de LSTM.
154: 2. **Split temporal é obrigatório**. Split aleatório vaza futuro e infla PR-AUC em +5-10pp.
155: 3. **Métricas corretas importam**. Nunca acurácia. Use PR-AUC + Recall@k%.
156: 4. **Documente resultados negativos**. Saber onde o método falha vale mais que "meu modelo deu 0,96".
157: 
158: ### Para produção
159: 
160: 5. **Threshold ótimo ≠ threshold operacional**. Calcule capacidade de revisão (quantos casos/dia seu time consegue revisar?).
161: 6. **Break-even é mais importante que ROI**. Nosso ROI é 28,6x, mas o break-even é **R$ 847 por revisão** — ou seja, mesmo com custo 56x maior, continua valendo a pena.
162: 7. **SHAP nativo é obrigatório**. Regulação exige explicar cada bloqueio.
163: 
164: ### Para portfólio
165: 
166: 8. **Resultado negativo é diferencial**. "Optuna piorou -0,64%" é mais raro que "meu modelo deu 0,99".
167: 9. **Cross-dataset prova generalização**. Rodar em 5 datasets demonstra rigor que 1 dataset não demonstra.
168: 10. **O código importa menos que o processo**. Split temporal, PR-AUC, SHAP, ROI — são os 4 pilares.
169: 
170: ---
171: 
172: ## Conclusão
173: 
174: Depois de **15 experimentos, 12,3 milhões de transações e 5 datasets**, a conclusão é incômoda:
175: 
176: > **O sinal em fraude está nas features, não no modelo.**
177: >
178: > Um CatBoost com configurações default + 3 features contextuais bem desenhadas **bate** deep learning, GNN, ensemble e tuning cego.
179: >
180: > **O ganho real em produção não vem de 0,96 vs 0,97 — vem de features melhores + threshold operacional + explicabilidade.**
181: 
182: Isso não é uma derrota para IA. É uma vitória para **rigor metodológico** — que, na minha opinião, é o que separa um analista júnior de um sênior.
183: 
184: ---
185: 
186: ## Código e dados
187: 
188: O projeto inteiro está em **[github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-](https://github.com/vinidias/Fraud-Analytics-PIX-Fraud-Detection-)**:
189: 
190: - 41 commits, 15 fases de análise
191: - 8 gráficos analíticos
192: - 5 páginas no dashboard Streamlit
193: - 12,3M transações processadas
194: - CI verde
195: 
196: **Se você trabalha com detecção de fraude e quer conversar sobre esses resultados, me chama no [LinkedIn](https://www.linkedin.com/in/vinicius-dias/).**
197: 
198: ---
199: 
200: *Este artigo resume 15 experimentos documentados em um repositório público. Todos os resultados negativos (Optuna, GNN, isotonic, ensemble) estão documentados com a mesma seriedade que os positivos.*
201: 

(End of file - total 201 lines)
</content>
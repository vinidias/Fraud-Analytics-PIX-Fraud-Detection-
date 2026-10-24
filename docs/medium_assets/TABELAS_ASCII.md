# Tabelas ASCII para Medium

Cole cada bloco dentro de um **code block** (bloco de código) no Medium.

---

## tabela_01_desafios

```
### Os 4 desafios técnicos do PIX

+--------------------------+---------------------------------------------+
| Desafio                  | Por que é difícil                           |
+==========================+=============================================+
| Desbalanceamento extremo | 0,77% de fraude -> acurácia engana          |
| Drift temporal           | Taxa cresce 20x ao longo do mês             |
| Custo assimétrico        | FN custa ~R$ 4.000; FP custa ~R$ 20         |
| Explicabilidade          | Regulação exige justificativa por transação |
+--------------------------+---------------------------------------------+
```

---

## tabela_02_modelos

```
### Comparação de 9 modelos -- PR-AUC

+-------------------------+--------+
| Modelo                  | PR-AUC |
+=========================+========+
| CatBoost (2M) -- MELHOR | 0,9614 |
| CatBoost (500k)         | 0,9603 |
| Ensemble (stacking)     | 0,9597 |
| XGBoost                 | 0,9587 |
| LightGBM                | 0,9484 |
| MLP (focal loss)        | 0,9206 |
| TabNet                  | 0,8753 |
| GNN (GraphSAGE)         | 0,3928 |
| AutoEncoder             | 0,2280 |
+-------------------------+--------+
```

---

## tabela_03_optuna

```
### Optuna: tuning piorou o modelo

+----------------+--------+--------+---------+
| Métrica        | Base   | Optuna | Delta   |
+================+========+========+=========+
| PR-AUC (teste) | 0,9598 | 0,9533 | -0,64pp |
| Brier          | 0,0122 | 0,0179 | +46%    |
+----------------+--------+--------+---------+
```

---

## tabela_04_cross_dataset

```
### Cross-dataset benchmark -- 5 datasets + PIX

+--------------+------------------------+--------+----------+
| Dataset      | Tipo                   | PR-AUC | vs PIX   |
+==============+========================+========+==========+
| MoMTSim      | Mobile money sintético | 0,9712 | +0,98pp  |
| Xente (REAL) | Mobile money REAL      | 0,9672 | +0,58pp  |
| PIX Fraud BR | PIX sintético          | 0,9614 | baseline |
| IEEE-CIS     | Cartão real            | 0,5067 | -45,47pp |
| BAF          | Bank account sintético | 0,1689 | -79,25pp |
+--------------+------------------------+--------+----------+
```

---

## tabela_05_isotonic

```
### Calibração isotônica: piorou tudo

+---------+---------+----------+---------+
| Métrica | Base    | Isotonic | Delta   |
+=========+=========+==========+=========+
| PR-AUC  | 0,9598  | 0,9545   | -0,53pp |
| Brier   | 0,01174 | 0,01215  | +3,5%   |
+---------+---------+----------+---------+
```

---

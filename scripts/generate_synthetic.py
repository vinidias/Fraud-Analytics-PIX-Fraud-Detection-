<path>D:/Dev/fraud/scripts/generate_synthetic.py</path>
<type>file</type>
<content>
1: """
2: Gera dataset sintético de transações com padrões de fraude para testes offline.
3: Útil enquanto não se baixa o dataset real do Kaggle.
4: 
5: Uso:
6:     python scripts/generate_synthetic.py --n 50000 --output data/synthetic/transactions.csv
7: """
8: from __future__ import annotations
9: 
10: import argparse
11: from datetime import datetime, timedelta
12: from pathlib import Path
13: 
14: import numpy as np
15: import pandas as pd
16: 
17: 
18: def generate(n: int = 50000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
19:     rng = np.random.default_rng(seed)
20:     n_fraud = int(n * fraud_rate)
21:     n_legit = n - n_fraud
22: 
23:     start = datetime(2025, 1, 1)
24: 
25:     def _ts(k):
26:         # timestamps ao longo de 90 dias
27:         offsets = rng.integers(0, 90 * 24 * 3600, size=k)
28:         return [start + timedelta(seconds=int(s)) for s in offsets]
29: 
30:     # legítimas: valores log-normais, distribuição uniforme em horas
31:     legit = pd.DataFrame({
32:         "tx_id": rng.integers(1, 10**9, size=n_legit),
33:         "customer_id": rng.integers(1, 5000, size=n_legit),
34:         "merchant_id": rng.integers(1, 300, size=n_legit),
35:         "device_id": [f"dev_{i}" for i in rng.integers(1, 4000, size=n_legit)],
36:         "amount": np.round(rng.lognormal(mean=4.0, sigma=0.8, size=n_legit), 2),
37:         "event_time": _ts(n_legit),
38:         "lat": np.round(rng.uniform(-30, -3, size=n_legit), 4),
39:         "lon": np.round(rng.uniform(-60, -35, size=n_legit), 4),
40:         "is_fraud": 0,
41:         "is_chargeback": 0,
42:     })
43: 
44:     # fraudulentas: valores extremos, madrugada, velocity alto, device compartilhado
45:     fraud = pd.DataFrame({
46:         "tx_id": rng.integers(1, 10**9, size=n_fraud),
47:         "customer_id": rng.integers(1, 5000, size=n_fraud),
48:         "merchant_id": rng.integers(1, 300, size=n_fraud),
49:         "device_id": [f"dev_{i}" for i in rng.integers(1, 50, size=n_fraud)],
50:         "amount": np.round(rng.lognormal(mean=5.5, sigma=1.4, size=n_fraud), 2),
51:         "event_time": [
52:             (start + timedelta(days=int(d), hours=int(rng.choice([0, 1, 2, 3, 4, 23]))))
53:             for d in rng.integers(0, 90, size=n_fraud)
54:         ],
55:         "lat": np.round(rng.uniform(-30, -3, size=n_fraud), 4),
56:         "lon": np.round(rng.uniform(-60, -35, size=n_fraud), 4),
57:         "is_fraud": 1,
58:         "is_chargeback": 1,
59:     })
60: 
61:     df = pd.concat([legit, fraud], ignore_index=True)
62:     df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
63:     df["event_time"] = pd.to_datetime(df["event_time"])
64:     # target compatível com a vaga (Class = 1 fraude)
65:     df["Class"] = df["is_fraud"]
66:     return df
67: 
68: 
69: def main() -> None:
70:     p = argparse.ArgumentParser()
71:     p.add_argument("--n", type=int, default=50000)
72:     p.add_argument("--fraud-rate", type=float, default=0.02)
73:     p.add_argument("--output", required=True)
74:     args = p.parse_args()
75: 
76:     out = Path(args.output)
77:     out.parent.mkdir(parents=True, exist_ok=True)
78:     df = generate(args.n, args.fraud_rate)
79:     df.to_csv(out, index=False)
80:     print(f"[OK] {len(df):,} transações salvas em {out}")
81:     print(f"  fraude: {df['Class'].sum():,} ({100*df['Class'].mean():.2f}%)")
82:     print(f"  período: {df['event_time'].min()} → {df['event_time'].max()}")
83: 
84: 
85: if __name__ == "__main__":
86:     main()
87: 

(End of file - total 87 lines)
</content>
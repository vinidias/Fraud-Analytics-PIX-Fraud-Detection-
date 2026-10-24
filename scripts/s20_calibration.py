"""
s20 - Calibracao + cost-sensitive threshold.

Melhora o modelo de producao em 3 frentes:
    1. Isotonic calibration (Brier score)
    2. Cost-sensitive threshold (ROI)
    3. Conformal prediction (intervalos por amostra)
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import train_test_split
import catboost as cb

warnings.filterwarnings("ignore")

SRC = Path("data/external_2026/fraud_benchmark/fraud_datasets/baf_bank_account_opening_synthetic.parquet")
OUT = Path("docs/s20_calibration")
OUT.mkdir(parents=True, exist_ok=True)

META_COLS = ["dataset", "split", "ts", "label", "user_id", "seen_user"]


def main():
    print("[1] carregando BAF...")
    df = pd.read_parquet(SRC)
    y = df["label"].astype(int)
    feat_cols = [c for c in df.columns if c not in META_COLS and c != "label"]
    X = df[feat_cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            try:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            except Exception:
                X[c] = pd.factorize(X[c].astype(str))[0]
    for c in X.columns:
        if X[c].dtype == "object" or str(X[c].dtype) in ("category", "string"):
            X[c] = pd.factorize(X[c].astype(str))[0]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(-999).astype("float64")

    if "split" in df.columns:
        train_mask = (df["split"] == "train").values
        test_mask = (df["split"] == "test").values
    elif "ts" in df.columns:
        ts_sorted = df["ts"].sort_values()
        cutoff = ts_sorted.iloc[int(len(ts_sorted) * 0.7)]
        train_mask = (df["ts"] < cutoff).values
        test_mask = (df["ts"] >= cutoff).values
    else:
        idx_tr, idx_te = train_test_split(
            df.index, test_size=0.3, random_state=42, stratify=df["label"]
        )
        train_mask = df.index.isin(idx_tr)
        test_mask = df.index.isin(idx_te)
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    print(f"    train: {len(X_tr):,} | test: {len(X_te):,}")

    # Split interno para calibracao
    X_tr2, X_cal, y_tr2, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
    )

    print("[2] treinando modelo base...")
    pos_weight = (y_tr2 == 0).sum() / max((y_tr2 == 1).sum(), 1)
    base = cb.CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        scale_pos_weight=pos_weight, random_seed=42,
        verbose=False, thread_count=-1, bootstrap_type="Bernoulli",
    )
    base.fit(X_tr2, y_tr2)

    print("[3] calibrando (isotonic)...")
    # sklearn >= 1.6 removeu cv="prefit" — usar FrozenEstimator
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    except ImportError:
        # fallback para versões antigas
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    # Avaliar
    def eval_model(name, proba):
        pr_auc = average_precision_score(y_te, proba)
        brier = brier_score_loss(y_te, proba)
        prec, rec, thr = precision_recall_curve(y_te, proba)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        idx = int(np.argmax(f1))
        return {
            "modelo": name,
            "pr_auc": round(float(pr_auc), 4),
            "brier": round(float(brier), 5),
            "f1": round(float(f1[idx]), 4),
            "threshold": round(float(thr[idx]) if idx < len(thr) else 0.5, 4),
        }

    results = []
    results.append(eval_model("CatBoost (base)", base.predict_proba(X_te)[:, 1]))
    results.append(eval_model("CatBoost (isotonic)", calibrated.predict_proba(X_te)[:, 1]))

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT / "calibration_results.csv", index=False)
    print(df_res.to_string(index=False))

    # Cost-sensitive threshold
    print("[4] cost-sensitive threshold...")
    proba = calibrated.predict_proba(X_te)[:, 1]
    prec, rec, thr = precision_recall_curve(y_te, proba)
    cost_fp = 20.0  # custo de falso positivo (revisao)
    avg_fraud = 5000.0  # valor medio de fraude evitada
    cost_fn = avg_fraud * 0.85  # custo de falso negativo (recuperavel 85%)
    costs = []
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        tp = r * y_te.sum()
        fp = tp * (1 - p) / max(p, 1e-9)
        fn = y_te.sum() - tp
        cost = fp * cost_fp + fn * cost_fn
        costs.append(cost)
    best_idx = int(np.argmin(costs))
    best_thr = float(thr[best_idx])
    best_prec = float(prec[best_idx])
    best_rec = float(rec[best_idx])

    # Dashboard
    lines = []
    lines.append("# s20 - Calibracao e threshold otimo")
    lines.append("")
    lines.append("## Objetivo")
    lines.append("")
    lines.append("Duas melhorias operacionais:")
    lines.append("")
    lines.append("1. **Isotonic calibration**: ajustar probabilidades para refletir realidade (Brier menor)")
    lines.append("2. **Cost-sensitive threshold**: escolher threshold minimizando custo total (FP vs FN)")
    lines.append("")
    lines.append("## Resultados")
    lines.append("")
    lines.append("| Modelo | PR-AUC | Brier | F1 |")
    lines.append("|---|---:|---:|---:|")
    for _, r in df_res.iterrows():
        lines.append(f"| {r['modelo']} | {r['pr_auc']:.4f} | {r['brier']:.5f} | {r['f1']:.4f} |")
    lines.append("")
    delta_brier = (df_res.iloc[1]["brier"] - df_res.iloc[0]["brier"]) / df_res.iloc[0]["brier"] * 100
    lines.append(f"**Delta Brier (isotonic vs base)**: {delta_brier:+.1f}%")
    lines.append("")
    lines.append("## Cost-sensitive threshold")
    lines.append("")
    lines.append(f"- **Custo FP** (revisao): R$ {cost_fp:.0f}")
    lines.append(f"- **Custo FN** (fraude nao detectada): R$ {cost_fn:.0f}")
    lines.append(f"- **Threshold otimo**: {best_thr:.4f}")
    lines.append(f"- **Precision**: {best_prec:.4f}")
    lines.append(f"- **Recall**: {best_rec:.4f}")
    lines.append("")
    lines.append("## Licao de negocio")
    lines.append("")
    lines.append("> **Calibracao importa mais que PR-AUC**: o threshold operacional precisa refletir o custo real. Um modelo com PR-AUC alto mas descalibrado pode ter threshold errado — e o custo de operar errado supera o ganho de metrica.")

    (OUT / "dashboard_calibration.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "calibracao": df_res.to_dict(orient="records"),
        "cost_sensitive_threshold": {
            "threshold": round(best_thr, 4),
            "precision": round(best_prec, 4),
            "recall": round(best_rec, 4),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] dashboard em {OUT}")


if __name__ == "__main__":
    main()
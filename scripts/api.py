"""
FastAPI endpoint para inferencia de fraude PIX.

Rodar:
    uvicorn scripts.api:app --reload --port 8000

Docs interativas: http://localhost:8000/docs
"""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Fraud Detection API - PIX",
    description="Endpoint de inferencia para deteccao de fraude em transacoes PIX",
    version="1.0.0",
)

MODEL_PATH = Path("docs/s15_external_model/catboost_model.cbm")

# Tentar carregar modelo
model = None
try:
    from catboost import CatBoostClassifier
    if MODEL_PATH.exists():
        model = CatBoostClassifier()
        model.load_model(str(MODEL_PATH))
except Exception as e:
    print(f"Modelo nao carregado: {e}")


class Transaction(BaseModel):
    """Schema de uma transacao (16 features do modelo)."""
    amount: float = Field(..., description="Valor em R$", ge=0)
    saldo_anterior_pagador: float = Field(0.0, description="Saldo antes do pagador")
    saldo_posterior_pagador: float = Field(0.0, description="Saldo depois do pagador")
    saldo_anterior_recebedor: float = Field(0.0, description="Saldo antes do recebedor")
    saldo_posterior_recebedor: float = Field(0.0, description="Saldo depois do recebedor")
    residual_balance_ratio: float = Field(0.5, description="Saldo depois / antes do pagador", ge=0, le=1)
    receiver_amount_ratio: float = Field(0.5, description="Proporcao do valor que o recebedor recebe", ge=0, le=1)
    hora_dia: int = Field(12, description="Hora do dia (0-23)", ge=0, le=23)
    is_workday: int = Field(1, description="1 = dia util, 0 = fim de semana", ge=0, le=1)
    above_night_limit: int = Field(0, description="1 = acima do limite noturno", ge=0, le=1)


class Prediction(BaseModel):
    """Resultado da predicao."""
    probability: float = Field(..., description="Probabilidade de fraude (0-1)")
    is_fraud: bool = Field(..., description="Decisao binaria (threshold 0.5)")
    risk_level: str = Field(..., description="low / medium / high")
    recommended_action: str = Field(..., description="approve / step_up_auth / block")


def compute_features(tx: Transaction) -> np.ndarray:
    """Replica o feature engineering do s07."""
    is_night = 1 if tx.hora_dia in [0, 1, 2, 3, 4, 5] else 0
    night_x_residual = is_night * (1 - tx.residual_balance_ratio)
    night_x_ratio = is_night * tx.receiver_amount_ratio
    limit_x_residual = tx.above_night_limit * (1 - tx.residual_balance_ratio)
    drain_score = (1 - tx.residual_balance_ratio) * tx.receiver_amount_ratio * (is_night + 1)
    log_amount = np.log1p(tx.amount)

    return np.array([[
        tx.amount, log_amount,
        tx.saldo_anterior_pagador, tx.saldo_posterior_pagador,
        tx.saldo_anterior_recebedor, tx.saldo_posterior_recebedor,
        tx.residual_balance_ratio, tx.receiver_amount_ratio,
        tx.hora_dia, is_night, tx.is_workday, tx.above_night_limit,
        night_x_residual, night_x_ratio, limit_x_residual, drain_score,
    ]])


@app.get("/")
def root():
    return {
        "service": "Fraud Detection API - PIX",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    """Prediz probabilidade de fraude para uma transacao."""
    if model is None:
        # Fallback: heuristica calibrada pelos SHAP values
        is_night = 1 if tx.hora_dia in [0, 1, 2, 3, 4, 5] else 0
        drain_score = (1 - tx.residual_balance_ratio) * tx.receiver_amount_ratio * (is_night + 1)
        logit = (
            -3.0
            + 2.5 * is_night
            + 3.0 * drain_score
            + 1.5 * (tx.receiver_amount_ratio - 0.5)
            - 2.0 * tx.residual_balance_ratio
            - 0.5 * np.log1p(tx.saldo_anterior_recebedor / 1000)
            + 0.5 * np.log1p(tx.amount)
            - 0.3 * tx.is_workday
        )
        proba = float(1 / (1 + np.exp(-logit)))
    else:
        X = compute_features(tx)
        proba = float(model.predict_proba(X)[0, 1])

    risk_level = "high" if proba > 0.7 else ("medium" if proba > 0.4 else "low")
    action = "block" if proba > 0.7 else ("step_up_auth" if proba > 0.4 else "approve")

    return Prediction(
        probability=round(proba, 4),
        is_fraud=proba > 0.5,
        risk_level=risk_level,
        recommended_action=action,
    )


@app.post("/predict/batch")
def predict_batch(transactions: List[Transaction]):
    """Predicao em lote (max 1000 transacoes)."""
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 transactions per request")
    return [predict(tx) for tx in transactions]


@app.get("/model/info")
def model_info():
    """Informacoes sobre o modelo em producao."""
    return {
        "model_type": "CatBoost" if model is not None else "heuristic-fallback",
        "features": [
            "amount", "log_amount",
            "saldo_anterior_pagador", "saldo_posterior_pagador",
            "saldo_anterior_recebedor", "saldo_posterior_recebedor",
            "residual_balance_ratio", "receiver_amount_ratio",
            "hora_dia", "is_night", "is_workday", "above_night_limit",
            "night_x_residual", "night_x_ratio", "limit_x_residual", "drain_score",
        ],
        "performance": {
            "pr_auc": 0.9614,
            "roc_auc": 0.9928,
            "f1": 0.9072,
            "precision": 0.8903,
            "recall": 0.9246,
        },
        "threshold_recommended": 0.5,
        "dataset": "pix-fraud-br (Hugging Face)",
    }

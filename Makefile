# ============================================================
# Makefile — Fraud Analytics (Incognia prep)
# Uso: make <target>   (Windows: usar Git Bash / WSL)
# ============================================================

.PHONY: help venv install setup data clean test

help:
	@echo "Targets:"
	@echo "  venv     - cria .venv"
	@echo "  install  - instala requirements.txt"
	@echo "  setup    - cria estrutura de pastas + .env"
	@echo "  data     - baixa datasets (requer Kaggle API)"
	@echo "  clean    - limpa caches e artefatos"

venv:
	python -m venv .venv

install:
	.venv/Scripts/pip install --upgrade pip
	.venv/Scripts/pip install -r requirements.txt

setup:
	mkdir -p data models notebooks/s01_eda notebooks/s02_sql notebooks/s03_features notebooks/s04_anomaly notebooks/s05_dashboard notebooks/s06_case
	cp -n .env.example .env || true

data:
	python scripts/download_datasets.py

clean:
	rm -rf .ipynb_checkpoints notebooks/**/.ipynb_checkpoints
	find . -name "__pycache__" -type d -exec rm -rf {} +

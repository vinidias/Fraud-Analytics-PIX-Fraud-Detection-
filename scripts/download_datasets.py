"""
Baixa datasets de prática via Kaggle API.

Pré-requisitos:
    pip install kaggle
    ~/.kaggle/kaggle.json  (ou variáveis KAGGLE_USERNAME/KAGGLE_KEY)

Uso:
    python scripts/download_datasets.py --dataset ulb
    python scripts/download_datasets.py --dataset ieee
    python scripts/download_datasets.py --dataset all
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DATASETS = {
    "ulb":  ("mlg-ulb/creditcardfraud",  "data/creditcard"),
    "ieee": ("c/ieee-fraud-detection",   "data/ieee"),
    "paysim": ("ealaxi/paysim1",         "data/paysim"),
}


def download(name: str) -> None:
    slug, out = DATASETS[name]
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {name} ({slug}) → {out_path}")
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_path), "--unzip"]
    subprocess.run(cmd, check=True)
    print(f"✓ {name} pronto em {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=[*DATASETS, "all"], required=True)
    args = p.parse_args()
    names = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for n in names:
        try:
            download(n)
        except subprocess.CalledProcessError as e:
            print(f"✗ {n} falhou: {e}. Configure a Kaggle API (ver README).")


if __name__ == "__main__":
    main()

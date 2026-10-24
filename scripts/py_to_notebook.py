"""
Converte um script .py (estilo jupytext # %%) em .ipynb.

Uso:
    python scripts/py_to_notebook.py notebooks/s01_eda/01_eda_ulb_creditcard.ipynb.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def split_cells(src: str):
    parts = re.split(r"^#\s*%%(.*)$", src, flags=re.MULTILINE)
    # parts[0] = preâmbulo, depois (marker, conteúdo) pares
    cells = []
    if parts[0].strip():
        cells.append(("code", parts[0]))
    for i in range(1, len(parts), 2):
        marker = parts[i].strip().lower()
        content = parts[i + 1]
        if marker.startswith("markdown"):
            # remove a marcação # do markdown
            md = "\n".join(line.lstrip("# ").rstrip() for line in content.strip().splitlines())
            cells.append(("markdown", md))
        else:
            cells.append(("code", content))
    return cells


def to_notebook(cells) -> dict:
    nb_cells = []
    for kind, content in cells:
        if kind == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": content.splitlines(keepends=True),
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": content.splitlines(keepends=True),
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src.with_suffix("")
    out = out.with_suffix(".ipynb") if out.suffix != ".ipynb" else out

    text = src.read_text(encoding="utf-8")
    cells = split_cells(text)
    nb = to_notebook(cells)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {src} -> {out} ({len(cells)} células)")


if __name__ == "__main__":
    main()

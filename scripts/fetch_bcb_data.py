"""
Coleta dados REAIS de fraude PIX do Banco Central (2026).

Endpoint: Pix_DadosAbertos / EstatisticasFraudesPix
Fonte: https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/versao/v1/odata/
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/external_2026")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/versao/v1/odata"
ENDPOINTS = {
    "fraudes": f"{BASE}/EstatisticasFraudesPix()",
    "transacoes": f"{BASE}/EstatisticasTransacoesPix()",
    "chaves": f"{BASE}/ChavesPix()",
    "usuarios_dict": f"{BASE}/PixUsuariosCadastradosDICT()",
    "cnae_recebedor": f"{BASE}/CnaePorteRecebedor()",
}


def fetch_all(url: str, page_size: int = 500) -> list:
    """Baixa todos os registros paginando."""
    rows = []
    skip = 0
    while True:
        params = {"$format": "json", "$top": page_size, "$skip": skip}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  erro em skip={skip}: {e}")
            break
        batch = data.get("value", [])
        if not batch:
            break
        rows.extend(batch)
        print(f"  baixados {len(rows)} registros...")
        if len(batch) < page_size:
            break
        skip += page_size
    return rows


def main():
    summaries = {}
    for name, url in ENDPOINTS.items():
        print(f"\n[{name}] {url}")
        rows = fetch_all(url)
        if not rows:
            print(f"  vazio ou erro")
            summaries[name] = {"status": "empty"}
            continue
        df = pd.DataFrame(rows)
        out = OUT / f"bcb_{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"  [OK] {len(df):,} linhas x {df.shape[1]} colunas -> {out}")
        summaries[name] = {"rows": len(df), "cols": df.shape[1], "path": str(out)}

    (OUT / "bcb_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[OK] coleta BCB concluida")


if __name__ == "__main__":
    main()
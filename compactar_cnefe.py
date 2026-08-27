"""
Script Utilitário: Compactador de Base CNEFE (IBGE)
Reduz o tamanho do arquivo .parquet em até 70% mantendo apenas as colunas essenciais para o geocodificador.
Uso: python compactar_cnefe.py
"""

import os
import sys
import time
import duckdb
from pathlib import Path

def compactar_parquet(origem, destino):
    if not Path(origem).exists():
        print(f"❌ Arquivo de origem '{origem}' não encontrado.")
        return

    print(f"⏳ Compactando '{origem}' -> '{destino}'...")
    t0 = time.time()
    
    con = duckdb.connect()
    query = f"""
        COPY (
            SELECT 
                CAST(COD_MUNICIPIO AS INTEGER) AS COD_MUNICIPIO,
                TRIM(COALESCE(NOM_TIPO_SEGLOGR, '') || ' ' || COALESCE(NOM_TITULO_SEGLOGR, '') || ' ' || COALESCE(NOM_SEGLOGR, '')) AS NOM_LOGRADOURO,
                CAST(NUM_ENDERECO AS INTEGER) AS NUM_ENDERECO,
                CAST(LATITUDE AS FLOAT) AS LATITUDE,
                CAST(LONGITUDE AS FLOAT) AS LONGITUDE,
                CEP
            FROM read_parquet('{origem}')
        ) TO '{destino}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);
    """
    con.execute(query)
    con.close()
    
    t_total = time.time() - t0
    tam_orig = os.path.getsize(origem) / (1024 * 1024)
    tam_dest = os.path.getsize(destino) / (1024 * 1024)
    reducao = ((tam_orig - tam_dest) / tam_orig) * 100

    print(f"✅ Concluído em {t_total:.2f}s!")
    print(f"📦 Tamanho Original: {tam_orig:.2f} MB")
    print(f"⚡ Tamanho Compactado: {tam_dest:.2f} MB")
    print(f"🎉 Redução de {reducao:.1f}%\n")

if __name__ == '__main__':
    # Compacta SP e RJ se existirem
    for uf in ['sp', 'rj']:
        orig = f"cnefe_{uf}.parquet"
        dest = f"cnefe_{uf}_compacto.parquet"
        if Path(orig).exists():
            compactar_parquet(orig, dest)

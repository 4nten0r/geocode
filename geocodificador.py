"""
Geocodificador Inteligente de Alta Performance
Integração CNEFE (IBGE 2022) via DuckDB/Parquet + IA/Heurística de Endereços + Fallback Geopy
"""

import os
import sys
import time
import json
import re
from pathlib import Path
import pandas as pd
import duckdb
from rapidfuzz import process, fuzz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderRateLimited, GeocoderTimedOut, GeocoderServiceError
import streamlit as st

# Importa o módulo de inteligência de endereços
from endereco_ia import (
    normalizar_texto,
    normalizar_municipio,
    limpar_numero,
    extrair_numero_endereco,
    preparar_endereco,
    sem_prefixo_tipo_rua,
    calcular_similaridade
)

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
ARQUIVO_IBGE_MUNICIPIOS = "ibge_municipios.json"
CACHE_GEOPY_ARQUIVO = "cache_geopy.json"
NOMINATIM_USER_AGENT = "GeocodificadorIA_IBGE_Turbo/8.0"
DEFAULT_SCORE_CUTOFF = 75

# ============================================================
# GERENCIAMENTO DE CACHE E BASES DE DADOS
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_base_ibge():
    """Carrega o mapeamento offline de municípios para códigos oficiais do IBGE."""
    if Path(ARQUIVO_IBGE_MUNICIPIOS).exists():
        try:
            with open(ARQUIVO_IBGE_MUNICIPIOS, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def carregar_cache_geopy():
    """Carrega o cache unificado de geocodificação externa."""
    cache = {}
    # Carrega cache principal e migra v2 se existir
    for arquivo in [CACHE_GEOPY_ARQUIVO, "cache_geopy_v2.json"]:
        p = Path(arquivo)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    dados = json.load(f)
                    cache.update(dados)
            except Exception:
                pass
    return cache

def salvar_cache_geopy(cache):
    """Salva o cache em disco de forma segura."""
    try:
        with open(CACHE_GEOPY_ARQUIVO, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def localizar_parquet_estado(estado):
    """Localiza o arquivo Parquet do CNEFE correspondente ao estado."""
    estado_limpo = normalizar_texto(estado).lower()[:2]
    candidatos = [
        f"cnefe_{estado_limpo}.parquet",
        f"cnefe_{estado_limpo.upper()}.parquet"
    ]
    for c in candidatos:
        if Path(c).exists():
            return c
    return None

# ============================================================
# MOTOR DE RESOLUÇÃO CNEFE EM LOTE
# ============================================================

def resolver_codigos_ibge(df, col_mun, col_uf, ibge_base):
    """Mapeia os nomes dos municípios para seus respectivos códigos numéricos do IBGE."""
    cods_ibge = []
    ibge_sp = ibge_base.get("SP", {})
    
    # Dicionário de cache local para evitar matches repetidos
    cache_mun_match = {}
    
    for _, row in df.iterrows():
        uf_raw = normalizar_texto(row.get(col_uf, "SP"))[:2] if col_uf else "SP"
        uf = uf_raw if uf_raw in ibge_base else "SP"
        mun_raw = normalizar_municipio(row.get(col_mun, ""))
        
        chave = (uf, mun_raw)
        if chave in cache_mun_match:
            cods_ibge.append(cache_mun_match[chave])
            continue
            
        uf_dict = ibge_base.get(uf, ibge_sp)
        cod = uf_dict.get(mun_raw)
        
        if not cod and mun_raw and uf_dict:
            # Match fuzzy de alta velocidade no nome da cidade
            match = process.extractOne(mun_raw, list(uf_dict.keys()), scorer=fuzz.token_sort_ratio)
            if match and match[1] >= 80:
                cod = uf_dict[match[0]]
                
        cache_mun_match[chave] = cod
        cods_ibge.append(cod)
        
    return cods_ibge

def carregar_e_indexar_cnefe(parquet_file, cods_ibge):
    """
    Executa consulta vetorial única no DuckDB e monta o índice em memória
    para busca de ruas e números prediais em tempo O(1).
    """
    cods_validos = [str(int(c)) for c in set(cods_ibge) if c is not None and not pd.isna(c)]
    if not cods_validos:
        return {}

    cods_sql = ", ".join(cods_validos)
    parquet_escaped = parquet_file.replace("'", "''")
    
    con = duckdb.connect(database=":memory:")
    query = f"""
        SELECT 
            COD_MUNICIPIO,
            TRIM(COALESCE(NOM_TIPO_SEGLOGR, '') || ' ' || COALESCE(NOM_TITULO_SEGLOGR, '') || ' ' || COALESCE(NOM_SEGLOGR, '')) AS NOM_LOGRADOURO,
            NUM_ENDERECO,
            LATITUDE,
            LONGITUDE,
            CEP
        FROM read_parquet('{parquet_escaped}')
        WHERE COD_MUNICIPIO IN ({cods_sql})
    """
    cnefe_df = con.execute(query).df()
    con.close()

    if cnefe_df.empty:
        return {}

    # Normalização em lote
    cnefe_df['RUA_NORM'] = cnefe_df['NOM_LOGRADOURO'].map(normalizar_texto)
    cnefe_df['NUM_NORM'] = cnefe_df['NUM_ENDERECO'].map(limpar_numero)

    # Indexação otimizada em dicionários
    registros = cnefe_df[['COD_MUNICIPIO', 'RUA_NORM', 'NUM_NORM', 'LATITUDE', 'LONGITUDE', 'CEP']].to_dict(orient='records')
    
    cnefe_index = {}
    for r in registros:
        cod = int(r['COD_MUNICIPIO'])
        rua = r['RUA_NORM']
        if not rua:
            continue
        if cod not in cnefe_index:
            cnefe_index[cod] = {'ruas_dict': {}, 'ruas_list': []}
            
        ruas_dict = cnefe_index[cod]['ruas_dict']
        if rua not in ruas_dict:
            ruas_dict[rua] = []
            cnefe_index[cod]['ruas_list'].append(rua)
        ruas_dict[rua].append(r)

    return cnefe_index

def buscar_endereco_no_indice(consulta, mun_cod, cnefe_index, score_cutoff=DEFAULT_SCORE_CUTOFF):
    """
    Realiza a correspondência do endereço (exata ou fuzzy) contra o índice em memória.
    """
    if mun_cod not in cnefe_index:
        return None

    mun_data = cnefe_index[mun_cod]
    ruas_list = mun_data['ruas_list']
    ruas_dict = mun_data['ruas_dict']

    if not consulta["rua"] or not ruas_list:
        return None

    # 1. Tentativa de correspondência exata O(1)
    candidatos = {}
    for v in consulta["variantes"]:
        if v in ruas_dict:
            candidatos[v] = 100.0

    # 2. Se não houver correspondência exata, executa RapidFuzz C++
    if not candidatos:
        for v in consulta["variantes"]:
            matches = process.extract(
                v, ruas_list,
                scorer=fuzz.WRatio,
                limit=6,
                score_cutoff=score_cutoff
            )
            for m_rua, m_score, _ in matches:
                candidatos[m_rua] = max(candidatos.get(m_rua, 0), m_score)

    if not candidatos:
        return None

    melhor_score = 0
    melhor_row = None
    num_busca = consulta["numero"]

    for cand_rua, score_base in candidatos.items():
        rows_cand = ruas_dict[cand_rua]
        for r in rows_cand:
            score = score_base
            num_cand = r['NUM_NORM']
            
            # Bônus para correspondência exata do número predial
            if num_busca and num_cand and num_busca == num_cand:
                score += 15

            if score > melhor_score:
                melhor_score = score
                melhor_row = r
                if score >= 115:  # Match perfeito de rua e número
                    break
        if melhor_score >= 115:
            break

    if melhor_row is not None and melhor_score >= score_cutoff:
        exato = bool(num_busca) and melhor_row['NUM_NORM'] == num_busca
        status = "✅ CNEFE Exato" if exato else "✅ CNEFE Logradouro"
        return float(melhor_row['LATITUDE']), float(melhor_row['LONGITUDE']), status

    return None

# ============================================================
# INTERFACE STREAMLIT
# ============================================================

def main():
    st.set_page_config(
        page_title="Geocodificador IA Turbo",
        page_icon="🌍",
        layout="wide"
    )

    st.title("🌍 Geocodificador IA Turbo")
    st.markdown("""
    **Motor Híbrido de Geocodificação em Lote**: 
    CNEFE IBGE 2022 (DuckDB + RapidFuzz Nativo) + IA de Tratamento de Endereços + Fallback Geopy (Nominatim).
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    limiar_score = st.sidebar.slider("Limiar de Similaridade de Logradouro (%)", min_value=60, max_value=95, value=75, step=1)
    geopy_delay = st.sidebar.slider("Delay Nominatim Fallback (segundos)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

    cache_geopy = carregar_cache_geopy()
    st.sidebar.info(f"💾 **Cache Geopy**: {len(cache_geopy)} endereços indexados.")

    ibge_base = carregar_base_ibge()
    if not ibge_base:
        st.sidebar.warning("⚠️ Base de municípios do IBGE não encontrada.")
    else:
        st.sidebar.success(f"🏛️ **Base IBGE**: {sum(len(v) for v in ibge_base.values())} municípios carregados.")

    # Upload
    arquivo_upload = st.file_uploader("📂 Envie sua planilha Excel (.xlsx)", type=["xlsx"])

    if arquivo_upload is not None:
        caminho_temp = "temp_entrada.xlsx"
        with open(caminho_temp, "wb") as f:
            f.write(arquivo_upload.getbuffer())

        df = pd.read_excel(caminho_temp)
        st.success(f"Planilha carregada com sucesso! Total de registros: **{len(df)}**")

        # Identificação de colunas
        colunas = list(df.columns)
        def detectar_col(opcoes, default_idx=0):
            for i, c in enumerate(colunas):
                for op in opcoes:
                    if op.lower() in str(c).lower():
                        return i
            return default_idx

        st.subheader("📋 Mapeamento de Colunas")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            col_rua = st.selectbox("Logradouro / Rua", colunas, index=detectar_col(["rua", "logradouro", "endereco"]))
        with c2:
            col_num = st.selectbox("Número", colunas, index=detectar_col(["numero", "num", "nº"]))
        with c3:
            col_bairro = st.selectbox("Bairro", colunas, index=detectar_col(["bairro"]))
        with c4:
            col_cep = st.selectbox("CEP", colunas, index=detectar_col(["cep"]))
        with c5:
            col_mun = st.selectbox("Município", colunas, index=detectar_col(["municipio", "cidade"]))
        with c6:
            col_uf = st.selectbox("Estado / UF", colunas, index=detectar_col(["estado", "uf"]))

        with st.expander("🔍 Pré-visualizar dados carregados"):
            st.dataframe(df[[col_rua, col_num, col_bairro, col_cep, col_mun, col_uf]].head(10))

        if st.button("🚀 Iniciar Geocodificação Turbo", type="primary"):
            t_inicio = time.time()
            total = len(df)
            lats = [""] * total
            lons = [""] * total
            status_list = [""] * total

            # Painel de métricas
            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            m_total = m_col1.metric("Total", f"{total}")
            m_cnefe_exato = m_col2.metric("CNEFE Exato", "0")
            m_cnefe_rua = m_col3.metric("CNEFE Logradouro", "0")
            m_geopy_cache = m_col4.metric("Geopy Cache", "0")
            m_geopy_live = m_col5.metric("Geopy Online", "0")
            m_falha = m_col6.metric("Não Encontrado", "0")

            barra_progresso = st.progress(0, text="🔍 Mapeando códigos de municípios pelo IBGE...")

            # Passo 1: Mapear códigos IBGE
            cods_ibge = resolver_codigos_ibge(df, col_mun, col_uf, ibge_base)
            df['__cod_ibge'] = cods_ibge

            # Agrupar por Estado para carregar parquets necessários
            ufs_presentes = set(df[col_uf].dropna().map(lambda x: normalizar_texto(x)[:2]).tolist()) if col_uf else {"SP"}
            if not ufs_presentes:
                ufs_presentes = {"SP"}

            cnefe_indices_por_uf = {}
            for uf in ufs_presentes:
                parquet_path = localizar_parquet_estado(uf)
                if parquet_path:
                    barra_progresso.progress(10, text=f"⚡ Carregando base CNEFE ({uf}) com DuckDB em memória...")
                    idx_uf = carregar_e_indexar_cnefe(parquet_path, cods_ibge)
                    cnefe_indices_por_uf[uf] = idx_uf

            barra_progresso.progress(25, text="⚡ Executando varredura rápida CNEFE com IA e RapidFuzz C++...")

            # Passo 2: Executar matching CNEFE
            cnefe_exato_count = 0
            cnefe_rua_count = 0
            indices_pendentes = []

            for i in range(total):
                row = df.iloc[i]
                cod = row['__cod_ibge']
                uf = normalizar_texto(row.get(col_uf, "SP"))[:2] if col_uf else "SP"
                if uf not in cnefe_indices_por_uf:
                    uf = "SP"

                cnefe_idx = cnefe_indices_por_uf.get(uf, {})
                consulta = preparar_endereco(row.get(col_rua, ""), row.get(col_num, ""))

                res = None
                if cod and not pd.isna(cod) and int(cod) in cnefe_idx:
                    res = buscar_endereco_no_indice(consulta, int(cod), cnefe_idx, score_cutoff=limiar_score)

                if res:
                    lats[i], lons[i], status_list[i] = res
                    if "Exato" in status_list[i]:
                        cnefe_exato_count += 1
                    else:
                        cnefe_rua_count += 1
                else:
                    indices_pendentes.append(i)

                if (i + 1) % 100 == 0 or (i + 1) == total:
                    prog = 25 + int(((i + 1) / total) * 35)  # 25% a 60%
                    barra_progresso.progress(prog, text=f"⚡ CNEFE: {i + 1}/{total} processados...")
                    m_cnefe_exato.metric("CNEFE Exato", f"{cnefe_exato_count}")
                    m_cnefe_rua.metric("CNEFE Logradouro", f"{cnefe_rua_count}")

            # Passo 3: Fallback Geopy para os não encontrados
            geopy_cache_count = 0
            geopy_live_count = 0
            falhas_count = 0
            total_pendentes = len(indices_pendentes)

            if total_pendentes > 0:
                barra_progresso.progress(60, text=f"🌍 Iniciando fallback Geopy para {total_pendentes} endereços...")
                geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=15)

                for k, idx in enumerate(indices_pendentes):
                    row = df.iloc[idx]
                    rua_prep = preparar_endereco(row.get(col_rua, ""), row.get(col_num, ""))
                    rua = rua_prep["rua"]
                    num = rua_prep["numero"]
                    mun = normalizar_municipio(row.get(col_mun, ""))
                    uf = normalizar_texto(row.get(col_uf, "SP"))[:2] if col_uf else "SP"
                    bairro = normalizar_texto(row.get(col_bairro, "")) if col_bairro else ""

                    consultas = [
                        {"street": f"{num} {rua}".strip() if num else rua, "city": mun, "state": uf, "country": "Brasil"},
                        f"{rua}, {num}, {bairro}, {mun}, {uf}, Brasil" if bairro and num else f"{rua}, {num}, {mun}, {uf}, Brasil" if num else f"{rua}, {mun}, {uf}, Brasil",
                        f"{rua}, {mun}, {uf}, Brasil"
                    ]

                    encontrou = False
                    for consulta in consultas:
                        chave = json.dumps(consulta, ensure_ascii=False, sort_keys=True)
                        if chave in cache_geopy:
                            coords = cache_geopy[chave]
                            if coords:
                                lats[idx], lons[idx] = coords[0], coords[1]
                                status_list[idx] = "✅ Geopy (Cache)"
                                encontrou = True
                                geopy_cache_count += 1
                                break
                            continue

                        # Requisição online segura com tratamento de rate limit
                        for tentativa in range(3):
                            try:
                                loc = geolocator.geocode(consulta, addressdetails=True)
                                time.sleep(geopy_delay)
                                if loc and -35 <= loc.latitude <= 6 and -75 <= loc.longitude <= -30:
                                    lats[idx], lons[idx] = loc.latitude, loc.longitude
                                    status_list[idx] = "✅ Geopy (Online)"
                                    cache_geopy[chave] = [loc.latitude, loc.longitude]
                                    encontrou = True
                                    geopy_live_count += 1
                                else:
                                    cache_geopy[chave] = None
                                break
                            except GeocoderRateLimited:
                                time.sleep(15.0)
                            except (GeocoderTimedOut, GeocoderServiceError):
                                time.sleep(geopy_delay)
                                break
                            except Exception:
                                break

                        if encontrou:
                            break

                    if not encontrou:
                        status_list[idx] = "❌ Não Encontrado"
                        falhas_count += 1

                    prog_geopy = 60 + int(((k + 1) / total_pendentes) * 40)
                    barra_progresso.progress(prog_geopy, text=f"🌍 Geopy: {k + 1}/{total_pendentes} processados...")
                    m_geopy_cache.metric("Geopy Cache", f"{geopy_cache_count}")
                    m_geopy_live.metric("Geopy Online", f"{geopy_live_count}")
                    m_falha.metric("Não Encontrado", f"{falhas_count}")

                salvar_cache_geopy(cache_geopy)

            # Finalização
            df.drop(columns=['__cod_ibge'], errors='ignore', inplace=True)
            df['Latitude'] = lats
            df['Longitude'] = lons
            df['Status_Precisao'] = status_list

            caminho_saida = "Coordenadas_Corrigidas_Final.xlsx"
            df.to_excel(caminho_saida, index=False)

            tempo_total = time.time() - t_inicio
            barra_progresso.progress(100, text=f"🎉 Concluído em {tempo_total:.2f}s!")
            st.success(f"Geocodificação concluída com sucesso em **{tempo_total:.2f} segundos**!")

            # Exibição de resultados e download
            st.subheader("📊 Resumo e Visualização")
            st.dataframe(df.head(20))

            # Mapa de pontos válidos
            df_mapa = df[(df['Latitude'] != "") & (df['Longitude'] != "")].copy()
            if not df_mapa.empty:
                df_mapa['lat'] = pd.to_numeric(df_mapa['Latitude'], errors='coerce')
                df_mapa['lon'] = pd.to_numeric(df_mapa['Longitude'], errors='coerce')
                df_mapa = df_mapa.dropna(subset=['lat', 'lon'])
                if not df_mapa.empty:
                    with st.expander("🗺️ Visualizar mapa de pontos geocodificados"):
                        st.map(df_mapa[['lat', 'lon']])

            with open(caminho_saida, "rb") as f:
                st.download_button(
                    label="📥 Baixar Planilha Corrigida (.xlsx)",
                    data=f,
                    file_name="Coordenadas_Corrigidas_Final.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == '__main__':
    main()
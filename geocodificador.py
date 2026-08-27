"""
Geocodificador Inteligente de Alta Performance
Integração CNEFE (IBGE 2022) via DuckDB/Parquet + ArcGIS World Geocoder + Google Maps API + IA de Endereços
Foco: Precisão Estrita Nível Google Maps (Número / Rua / Bairro) - Rejeição total de centróides de cidade.
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

# Garante que o diretório onde o script está localizado esteja no sys.path (essencial para Streamlit Cloud)
DIRETORIO_RAIZ = Path(__file__).resolve().parent
if str(DIRETORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_RAIZ))

import pandas as pd
import duckdb
from rapidfuzz import process, fuzz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderRateLimited, GeocoderTimedOut, GeocoderServiceError
import streamlit as st

# Importa o módulo de inteligência de endereços e motores de alta precisão
from endereco_ia import (
    normalizar_texto,
    normalizar_municipio,
    limpar_numero,
    extrair_numero_endereco,
    preparar_endereco,
    sem_prefixo_tipo_rua,
    calcular_similaridade,
    consultar_arcgis,
    consultar_google_maps,
    validar_resposta_geopy
)

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
ARQUIVO_IBGE_MUNICIPIOS = str(DIRETORIO_RAIZ / "ibge_municipios.json")
CACHE_GEOPY_ARQUIVO = str(DIRETORIO_RAIZ / "cache_geopy.json")
NOMINATIM_USER_AGENT = "GeocodificadorIA_IBGE_Turbo/8.5"
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
    """Carrega o cache de geocodificação offline."""
    if Path(CACHE_GEOPY_ARQUIVO).exists():
        try:
            with open(CACHE_GEOPY_ARQUIVO, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def salvar_cache_geopy(cache):
    """Salva o cache em disco de forma segura."""
    try:
        with open(CACHE_GEOPY_ARQUIVO, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def obter_url_secret_cnefe(estado):
    """Obtém a URL do CNEFE a partir do st.secrets ou variável de ambiente."""
    uf = normalizar_texto(estado).upper()[:2]
    chaves = [f"CNEFE_{uf}_URL", "CNEFE_URL", f"cnefe_{uf.lower()}_url", "cnefe_url"]
    
    try:
        if hasattr(st, "secrets"):
            for k in chaves:
                if k in st.secrets:
                    return str(st.secrets[k]).strip()
    except Exception:
        pass

    for k in chaves:
        val = os.getenv(k)
        if val and val.startswith("http"):
            return val.strip()
            
    return None

def obter_secret_google_maps():
    """Obtém a chave de API do Google Maps dos Secrets ou variáveis de ambiente."""
    try:
        if hasattr(st, "secrets") and "GOOGLE_MAPS_API_KEY" in st.secrets:
            return str(st.secrets["GOOGLE_MAPS_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

def localizar_parquet_estado(estado, url_remota=None):
    """Localiza o arquivo Parquet do CNEFE correspondente ao estado (local, URL do Secrets ou URL digitada)."""
    if url_remota and url_remota.startswith("http"):
        return url_remota

    url_secret = obter_url_secret_cnefe(estado)
    if url_secret and url_secret.startswith("http"):
        return url_secret

    estado_limpo = normalizar_texto(estado).lower()[:2]
    candidatos = [
        DIRETORIO_RAIZ / f"cnefe_{estado_limpo}_compacto.parquet",
        DIRETORIO_RAIZ / f"cnefe_{estado_limpo}.parquet",
        DIRETORIO_RAIZ / f"cnefe_{estado_limpo.upper()}.parquet",
        Path(f"cnefe_{estado_limpo}_compacto.parquet"),
        Path(f"cnefe_{estado_limpo}.parquet"),
        Path(f"cnefe_{estado_limpo.upper()}.parquet")
    ]
    for c in candidatos:
        if c.exists():
            return str(c)
            
    return None

# ============================================================
# MOTOR DE RESOLUÇÃO CNEFE EM LOTE
# ============================================================

def resolver_codigos_ibge(df, col_mun, col_uf, ibge_base):
    """Mapeia os nomes dos municípios para seus respectivos códigos numéricos do IBGE."""
    cods_ibge = []
    ibge_sp = ibge_base.get("SP", {})
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
        if not uf_dict:
            cods_ibge.append(None)
            cache_mun_match[chave] = None
            continue

        # 1. Match direto exato
        if mun_raw in uf_dict:
            cod = uf_dict[mun_raw]
            cods_ibge.append(cod)
            cache_mun_match[chave] = cod
            continue

        # 2. Match Fuzzy
        match = process.extractOne(
            mun_raw,
            list(uf_dict.keys()),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=78
        )
        if match:
            cod = uf_dict[match[0]]
            cods_ibge.append(cod)
            cache_mun_match[chave] = cod
        else:
            cods_ibge.append(None)
            cache_mun_match[chave] = None

    return cods_ibge

@st.cache_data(show_spinner=False)
def carregar_e_indexar_cnefe(caminho_parquet, cods_ibge_unicos):
    """Carrega dados do CNEFE filtrados para os municípios do lote e cria índice hash em memória."""
    cods_validos = [str(c) for c in cods_ibge_unicos if c and not pd.isna(c)]
    if not cods_validos:
        return {}

    in_clause = ",".join(cods_validos)
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except Exception:
        pass

    try:
        # Detecta se é o parquet compacto ou completo
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{caminho_parquet}') LIMIT 1;")
        colunas_parquet = [row[0] for row in con.fetchall()]
        
        if 'NOM_LOGRADOURO' in colunas_parquet:
            query = f"""
                SELECT 
                    CAST(COD_MUNICIPIO AS INTEGER) AS COD_MUNICIPIO,
                    NOM_LOGRADOURO,
                    CAST(NUM_ENDERECO AS VARCHAR) AS NUM_ENDERECO,
                    LATITUDE,
                    LONGITUDE,
                    CEP
                FROM read_parquet('{caminho_parquet}')
                WHERE CAST(COD_MUNICIPIO AS VARCHAR) IN ({in_clause})
            """
        else:
            query = f"""
                SELECT 
                    CAST(COD_MUNICIPIO AS INTEGER) AS COD_MUNICIPIO,
                    TRIM(COALESCE(NOM_TIPO_SEGLOGR, '') || ' ' || COALESCE(NOM_TITULO_SEGLOGR, '') || ' ' || COALESCE(NOM_SEGLOGR, '')) AS NOM_LOGRADOURO,
                    CAST(NUM_ENDERECO AS VARCHAR) AS NUM_ENDERECO,
                    LATITUDE,
                    LONGITUDE,
                    CEP
                FROM read_parquet('{caminho_parquet}')
                WHERE CAST(COD_MUNICIPIO AS VARCHAR) IN ({in_clause})
            """
        cnefe_df = con.execute(query).df()
    except Exception as e:
        st.error(f"Erro ao ler Parquet CNEFE: {e}")
        con.close()
        return {}
    finally:
        con.close()

    if cnefe_df.empty:
        return {}

    cnefe_df['RUA_NORM'] = cnefe_df['NOM_LOGRADOURO'].map(normalizar_texto)
    cnefe_df['NUM_NORM'] = cnefe_df['NUM_ENDERECO'].map(limpar_numero)

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
    """Realiza a correspondência do endereço (exata ou fuzzy) contra o índice em memória."""
    if mun_cod not in cnefe_index:
        return None

    mun_data = cnefe_index[mun_cod]
    ruas_list = mun_data['ruas_list']
    ruas_dict = mun_data['ruas_dict']

    if not consulta["rua"] or not ruas_list:
        return None

    # 1. Correspondência exata O(1)
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
            
            if num_busca and num_cand and num_busca == num_cand:
                score += 15

            if score > melhor_score:
                melhor_score = score
                melhor_row = r
                if score >= 115:
                    break
        if melhor_score >= 115:
            break

    if melhor_row is not None and melhor_score >= score_cutoff:
        exato = bool(num_busca) and melhor_row['NUM_NORM'] == num_busca
        status = "✅ CNEFE Exato (Número)" if exato else "✅ CNEFE Logradouro (Rua)"
        return float(melhor_row['LATITUDE']), float(melhor_row['LONGITUDE']), status

    return None

# ============================================================
# INTERFACE STREAMLIT
# ============================================================

def main():
    st.set_page_config(
        page_title="Geocodificador IA Turbo - Nível Google Maps",
        page_icon="🌍",
        layout="wide"
    )

    st.title("🌍 Geocodificador IA Turbo")
    st.markdown("""
    **Motor Multi-Camadas de Alta Precisão (Nível Google Maps)**:
    - 🏢 **Camada 1: CNEFE IBGE 2022**: Busca ultrarrápida vetorial em memória via DuckDB e RapidFuzz.
    - 🗺️ **Camada 2: Google Maps API**: Precisão máxima de coordenadas prediais no telhado (opcional).
    - 🌐 **Camada 3: ArcGIS World Geocoder (Esri)**: Geocodificador empresarial com localização exata de número predial e rua.
    - 🛡️ **Garantia Anti-Erro**: Rejeição estrita de centróides de cidade e validação de município de destino.
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configurações de Precisão")
    limiar_score = st.sidebar.slider("Limiar de Similaridade CNEFE (%)", min_value=60, max_value=95, value=75, step=1)
    
    st.sidebar.header("🗺️ Chave Google Maps (Opcional)")
    secret_google_key = obter_secret_google_maps()
    google_api_key = st.sidebar.text_input(
        "Chave Google Maps API (Nível Máximo):",
        value=secret_google_key,
        type="password",
        help="Se fornecida, consulta a API oficial do Google Maps com precisão ROOFTOP."
    )
    if google_api_key:
        st.sidebar.success("🔑 Google Maps API configurada!")

    st.sidebar.header("📁 Base de Dados CNEFE (IBGE)")
    tem_cnefe_local = (DIRETORIO_RAIZ / "cnefe_sp_compacto.parquet").exists() or (DIRETORIO_RAIZ / "cnefe_sp.parquet").exists() or (DIRETORIO_RAIZ / "cnefe_rj.parquet").exists() or Path("cnefe_sp.parquet").exists()
    secret_cnefe_sp = obter_url_secret_cnefe("SP")
    
    url_cnefe_custom = None
    if tem_cnefe_local:
        st.sidebar.success("✅ Base CNEFE Local (.parquet) detectada!")
    elif secret_cnefe_sp:
        st.sidebar.success("🌐 Base CNEFE Nuvem ativa via Streamlit Secrets!")
    else:
        st.sidebar.warning("⚠️ Base CNEFE não detectada localmente.")
        url_cnefe_custom = st.sidebar.text_input(
            "1. URL Remota CNEFE SP (Hugging Face / S3):",
            placeholder="https://.../cnefe_sp.parquet",
            help="O DuckDB lerá o arquivo Parquet remotamente da nuvem sem precisar baixar tudo."
        )
        
        cnefe_upload = st.sidebar.file_uploader("2. Ou envie o arquivo CNEFE (.parquet) aqui:", type=["parquet"])
        if cnefe_upload is not None:
            caminho_salvo = DIRETORIO_RAIZ / cnefe_upload.name
            with open(caminho_salvo, "wb") as f:
                f.write(cnefe_upload.getbuffer())
            st.sidebar.success(f"✅ Base {cnefe_upload.name} carregada com sucesso!")
            st.rerun()

    cache_geopy = carregar_cache_geopy()
    st.sidebar.info(f"💾 **Cache Offline**: {len(cache_geopy)} endereços indexados.")

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

        # Garante que as colunas do DataFrame sejam únicas
        df = df.loc[:, ~df.columns.duplicated()].copy()
        colunas = list(df.columns)
        opcoes_com_nenhum = ["(Não informado)"] + colunas

        def detectar_col(opcoes, default_idx=0):
            for i, c in enumerate(colunas):
                for op in opcoes:
                    if op.lower() in str(c).lower():
                        return i
            return default_idx

        def detectar_col_opcional(opcoes):
            for i, c in enumerate(colunas):
                for op in opcoes:
                    if op.lower() in str(c).lower():
                        return i + 1
            return 0

        st.subheader("📋 Mapeamento de Colunas")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            idx_rua = detectar_col(["rua", "logradouro", "endereco", "end"])
            col_rua = st.selectbox("Logradouro / Rua*", colunas, index=idx_rua)
        with c2:
            idx_num = detectar_col_opcional(["numero", "num", "nº", "n"])
            col_num_escolha = st.selectbox("Número", opcoes_com_nenhum, index=idx_num)
            col_num = None if col_num_escolha == "(Não informado)" else col_num_escolha
        with c3:
            idx_bairro = detectar_col_opcional(["bairro", "bair", "distrito"])
            col_bairro_escolha = st.selectbox("Bairro", opcoes_com_nenhum, index=idx_bairro)
            col_bairro = None if col_bairro_escolha == "(Não informado)" else col_bairro_escolha
        with c4:
            idx_cep = detectar_col_opcional(["cep", "codigo postal"])
            col_cep_escolha = st.selectbox("CEP", opcoes_com_nenhum, index=idx_cep)
            col_cep = None if col_cep_escolha == "(Não informado)" else col_cep_escolha
        with c5:
            idx_mun = detectar_col(["municipio", "cidade", "mun", "cid"])
            col_mun = st.selectbox("Município*", colunas, index=idx_mun)
        with c6:
            idx_uf = detectar_col_opcional(["uf", "estado", "est"])
            col_uf_escolha = st.selectbox("UF / Estado", opcoes_com_nenhum, index=idx_uf)
            col_uf = None if col_uf_escolha == "(Não informado)" else col_uf_escolha

        # Visualização prévia dos dados mapeados
        cols_preview = [c for c in [col_rua, col_num, col_bairro, col_cep, col_mun, col_uf] if c]
        # Remove eventuais duplicidades mantendo a ordem
        cols_preview_unicas = list(dict.fromkeys(cols_preview))
        st.dataframe(df[cols_preview_unicas].head(10))

        # Botão de Execução
        if st.button("🚀 Iniciar Geocodificação de Alta Precisão", type="primary"):
            t_inicio = time.time()
            total = len(df)
            lats = [""] * total
            lons = [""] * total
            status_list = [""] * total

            # Painel de métricas
            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            m_total = m_col1.metric("Total", f"{total}")
            m_cnefe_exato = m_col2.metric("CNEFE Exato", "0")
            m_cnefe_rua = m_col3.metric("CNEFE Rua", "0")
            m_web_exato = m_col4.metric("ArcGIS/Google Exato", "0")
            m_web_rua = m_col5.metric("ArcGIS/Google Rua", "0")
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
                parquet_path = localizar_parquet_estado(uf, url_remota=url_cnefe_custom)
                if parquet_path:
                    barra_progresso.progress(10, text=f"⚡ Carregando base CNEFE ({uf}) com DuckDB em memória...")
                    idx_uf = carregar_e_indexar_cnefe(parquet_path, cods_ibge)
                    cnefe_indices_por_uf[uf] = idx_uf

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
                num_val = row.get(col_num, "") if col_num else ""
                consulta = preparar_endereco(row.get(col_rua, ""), num_val)

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
                    prog = int(((i + 1) / total) * 35)
                    barra_progresso.progress(prog, text=f"⚡ CNEFE: {i + 1}/{total} processados...")
                    m_cnefe_exato.metric("CNEFE Exato", f"{cnefe_exato_count}")
                    m_cnefe_rua.metric("CNEFE Rua", f"{cnefe_rua_count}")

            # Passo 3: Motores de Alta Precisão (Google Maps / ArcGIS World Geocoder)
            web_exato_count = 0
            web_rua_count = 0
            falhas_count = 0
            total_pendentes = len(indices_pendentes)

            if total_pendentes > 0:
                barra_progresso.progress(40, text=f"🌍 Geocodificando {total_pendentes} endereços pendentes com precisão Google Maps / ArcGIS...")
                session = requests.Session()

                def processar_item_web(idx):
                    row = df.iloc[idx]
                    rua_val = row.get(col_rua, "")
                    num_val = row.get(col_num, "") if col_num else ""
                    bairro_val = row.get(col_bairro, "") if col_bairro else ""
                    mun_val = row.get(col_mun, "")
                    uf_val = row.get(col_uf, "SP") if col_uf else "SP"

                    # 1. Tenta Google Maps API se chave fornecida
                    if google_api_key:
                        res_google = consultar_google_maps(rua_val, num_val, bairro_val, mun_val, uf_val, google_api_key, session=session)
                        if res_google:
                            lat, lon, st_p, match_addr = res_google
                            return idx, lat, lon, st_p

                    # 2. Tenta ArcGIS World Geocoder (Alta precisão com número predial)
                    res_arc = consultar_arcgis(rua_val, num_val, bairro_val, mun_val, uf_val, session=session)
                    if res_arc:
                        lat, lon, st_p, match_addr = res_arc
                        return idx, lat, lon, st_p

                    return idx, None, None, None

                # Executa em paralelo com 6 threads
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futuros = [executor.submit(processar_item_web, idx) for idx in indices_pendentes]
                    for count, fut in enumerate(futuros, 1):
                        idx, lat_val, lon_val, st_val = fut.result()
                        if lat_val and lon_val:
                            lats[idx] = lat_val
                            lons[idx] = lon_val
                            status_list[idx] = st_val
                            if "Exato" in st_val:
                                web_exato_count += 1
                            else:
                                web_rua_count += 1
                        else:
                            status_list[idx] = "❌ Não Encontrado (Sem precisão de rua)"
                            falhas_count += 1

                        if count % 20 == 0 or count == total_pendentes:
                            prog = 40 + int((count / total_pendentes) * 55)
                            barra_progresso.progress(prog, text=f"🌍 Geocodificando: {count}/{total_pendentes} concluídos...")
                            m_web_exato.metric("ArcGIS/Google Exato", f"{web_exato_count}")
                            m_web_rua.metric("ArcGIS/Google Rua", f"{web_rua_count}")
                            m_falha.metric("Não Encontrado", f"{falhas_count}")

            t_total = time.time() - t_inicio
            barra_progresso.progress(100, text=f"✨ Concluído com sucesso em {t_total:.2f} segundos!")

            # Montagem do Resultado
            df_resultado = df.copy()
            if '__cod_ibge' in df_resultado.columns:
                df_resultado.drop(columns=['__cod_ibge'], inplace=True)

            df_resultado["Latitude"] = lats
            df_resultado["Longitude"] = lons
            df_resultado["Status_Geocodificacao"] = status_list

            total_encontrados = cnefe_exato_count + cnefe_rua_count + web_exato_count + web_rua_count
            taxa_sucesso = (total_encontrados / total) * 100 if total > 0 else 0

            st.success(f"""
            🎉 **Geocodificação Concluída!**
            - ⏱️ **Tempo Total**: {t_total:.2f}s
            - 🎯 **Taxa de Sucesso**: {taxa_sucesso:.1f}% ({total_encontrados}/{total})
            - 🏢 **CNEFE Exato**: {cnefe_exato_count} | 🏢 **CNEFE Rua**: {cnefe_rua_count}
            - 🌐 **ArcGIS/Google Exato**: {web_exato_count} | 🌐 **ArcGIS/Google Rua**: {web_rua_count}
            - ❌ **Não Encontrados**: {falhas_count}
            """)

            st.dataframe(df_resultado.head(20))

            # Exportação
            caminho_saida = "resultado_geocodificado.xlsx"
            df_resultado.to_excel(caminho_saida, index=False)
            with open(caminho_saida, "rb") as f:
                st.download_button(
                    label="📥 Baixar Planilha Geocodificada (.xlsx)",
                    data=f,
                    file_name="enderecos_geocodificados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
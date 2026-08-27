"""
Geocodificador Inteligente de Alta Performance
Integração CNEFE (IBGE 2022) via DuckDB/Parquet + IA/Heurística de Endereços + Fallback Geopy
Foco: Precisão Estrita (Número / Rua / Bairro) - Rejeição total de Centro de Cidade e cidades divergentes.
"""

import os
import sys
import time
import json
import re
from pathlib import Path

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

# Importa o módulo de inteligência de endereços
from endereco_ia import (
    normalizar_texto,
    normalizar_municipio,
    limpar_numero,
    extrair_numero_endereco,
    preparar_endereco,
    sem_prefixo_tipo_rua,
    calcular_similaridade,
    validar_resposta_geopy
)

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
ARQUIVO_IBGE_MUNICIPIOS = str(DIRETORIO_RAIZ / "ibge_municipios.json")
CACHE_GEOPY_ARQUIVO = str(DIRETORIO_RAIZ / "cache_geopy.json")
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
    for arquivo in [CACHE_GEOPY_ARQUIVO, str(DIRETORIO_RAIZ / "cache_geopy_v2.json")]:
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

def localizar_parquet_estado(estado, url_remota=None):
    """Localiza o arquivo Parquet do CNEFE correspondente ao estado (local, URL do Secrets ou URL digitada)."""
    # 1. URL digitada na interface
    if url_remota and url_remota.startswith("http"):
        return url_remota

    # 2. URL configurada no Streamlit Secrets / Env
    url_secret = obter_url_secret_cnefe(estado)
    if url_secret and url_secret.startswith("http"):
        return url_secret

    # 3. Arquivo local no disco
    estado_limpo = normalizar_texto(estado).lower()[:2]
    candidatos = [
        DIRETORIO_RAIZ / f"cnefe_{estado_limpo}.parquet",
        DIRETORIO_RAIZ / f"cnefe_{estado_limpo.upper()}.parquet",
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
        cod = uf_dict.get(mun_raw)
        
        if not cod and mun_raw and uf_dict:
            # Match fuzzy de alta velocidade no nome da cidade
            match = process.extractOne(mun_raw, list(uf_dict.keys()), scorer=fuzz.token_sort_ratio)
            if match and match[1] >= 80:
                cod = uf_dict[match[0]]
                
        cache_mun_match[chave] = cod
        cods_ibge.append(cod)
        
    return cods_ibge

def carregar_e_indexar_cnefe(parquet_source, cods_ibge):
    """
    Executa consulta vetorial única no DuckDB e monta o índice em memória
    para busca de ruas e números prediais em tempo O(1).
    """
    cods_validos = [str(int(c)) for c in set(cods_ibge) if c is not None and not pd.isna(c)]
    if not cods_validos:
        return {}

    cods_sql = ", ".join(cods_validos)
    
    con = duckdb.connect(database=":memory:")
    # Ativa httpfs se for URL remota (ex: Streamlit Cloud lendo S3/HuggingFace)
    if parquet_source.startswith("http"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
    
    parquet_escaped = parquet_source.replace("'", "''")
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
    try:
        cnefe_df = con.execute(query).df()
    except Exception as e:
        st.error(f"Erro ao ler CNEFE de '{parquet_source}': {e}")
        con.close()
        return {}
        
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
    **Motor de Geocodificação Estrita (Número / Rua / Bairro)**:
    - 🏢 **CNEFE IBGE 2022**: Coordenadas prediais e de face de quadra locais/remotas via DuckDB + RapidFuzz.
    - 🛡️ **Garantia Anti-Centro de Cidade**: Rejeita polígonos municipais genéricos para assegurar precisão de logradouro.
    - 🔒 **Validação Estrita de Cidade**: Nunca atribui coordenadas fora do município de destino.
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configurações de Precisão")
    limiar_score = st.sidebar.slider("Limiar de Similaridade de Logradouro (%)", min_value=60, max_value=95, value=75, step=1)
    geopy_delay = st.sidebar.slider("Delay Nominatim Fallback (segundos)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

    st.sidebar.header("📁 Base de Dados CNEFE (IBGE)")
    tem_cnefe_local = (DIRETORIO_RAIZ / "cnefe_sp.parquet").exists() or (DIRETORIO_RAIZ / "cnefe_rj.parquet").exists() or Path("cnefe_sp.parquet").exists()
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
                        return i + 1  # +1 por causa do '(Não informado)'
            return 0

        st.subheader("📋 Mapeamento de Colunas")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            col_rua = st.selectbox("Logradouro / Rua *", colunas, index=detectar_col(["rua", "logradouro", "endereco"]))
        with c2:
            sel_num = st.selectbox("Número", opcoes_com_nenhum, index=detectar_col_opcional(["numero", "num", "nº"]))
            col_num = None if sel_num == "(Não informado)" else sel_num
        with c3:
            sel_bairro = st.selectbox("Bairro", opcoes_com_nenhum, index=detectar_col_opcional(["bairro"]))
            col_bairro = None if sel_bairro == "(Não informado)" else sel_bairro
        with c4:
            sel_cep = st.selectbox("CEP", opcoes_com_nenhum, index=detectar_col_opcional(["cep"]))
            col_cep = None if sel_cep == "(Não informado)" else sel_cep
        with c5:
            col_mun = st.selectbox("Município *", colunas, index=detectar_col(["municipio", "cidade"]))
        with c6:
            sel_uf = st.selectbox("Estado / UF", opcoes_com_nenhum, index=detectar_col_opcional(["estado", "uf"]))
            col_uf = None if sel_uf == "(Não informado)" else sel_uf

        # Exibe preview sem colunas duplicadas
        cols_para_exibir = list(dict.fromkeys([c for c in [col_rua, col_num, col_bairro, col_cep, col_mun, col_uf] if c and c in df.columns]))
        with st.expander("🔍 Pré-visualizar dados carregados"):
            st.dataframe(df[cols_para_exibir].head(10))

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
            m_geopy_exato = m_col4.metric("Geopy Exato/Rua", "0")
            m_geopy_bairro = m_col5.metric("Geopy Bairro", "0")
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

            if cnefe_indices_por_uf:
                barra_progresso.progress(25, text="⚡ Executando varredura rápida CNEFE com IA e RapidFuzz C++...")
            else:
                st.info("ℹ️ Base CNEFE local não disponível. Prosseguindo diretamente com motor de Geocodificação Estrita...")

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
                    prog = 25 + int(((i + 1) / total) * 35)
                    barra_progresso.progress(prog, text=f"⚡ CNEFE: {i + 1}/{total} processados...")
                    m_cnefe_exato.metric("CNEFE Exato", f"{cnefe_exato_count}")
                    m_cnefe_rua.metric("CNEFE Logradouro", f"{cnefe_rua_count}")

            # Passo 3: Fallback Geopy Estrito para os não encontrados
            geopy_rua_count = 0
            geopy_bairro_count = 0
            falhas_count = 0
            total_pendentes = len(indices_pendentes)

            if total_pendentes > 0:
                barra_progresso.progress(60, text=f"🌍 Iniciando fallback Geopy Estrito para {total_pendentes} endereços...")
                geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=15)

                for k, idx in enumerate(indices_pendentes):
                    row = df.iloc[idx]
                    num_val = row.get(col_num, "") if col_num else ""
                    rua_prep = preparar_endereco(row.get(col_rua, ""), num_val)
                    rua = rua_prep["rua"]
                    num = rua_prep["numero"]
                    mun = normalizar_municipio(row.get(col_mun, ""))
                    uf = normalizar_texto(row.get(col_uf, "SP"))[:2] if col_uf else "SP"
                    bairro = normalizar_texto(row.get(col_bairro, "")) if col_bairro else ""

                    consultas = [
                        {"street": f"{num} {rua}".strip() if num else rua, "city": mun, "state": uf, "country": "Brasil"},
                        f"{rua}, {num}, {bairro}, {mun}, {uf}, Brasil" if bairro and num else f"{rua}, {num}, {mun}, {uf}, Brasil" if num else f"{rua}, {mun}, {uf}, Brasil",
                        f"{rua}, {mun}, {uf}, Brasil",
                        f"{bairro}, {mun}, {uf}, Brasil" if bairro else None
                    ]
                    consultas = [c for c in consultas if c is not None]

                    encontrou = False
                    for consulta in consultas:
                        chave = json.dumps(consulta, ensure_ascii=False, sort_keys=True)
                        if chave in cache_geopy:
                            cached = cache_geopy[chave]
                            if cached and len(cached) >= 3:
                                lats[idx], lons[idx] = cached[0], cached[1]
                                status_list[idx] = cached[2] + " (Cache)"
                                encontrou = True
                                if "Bairro" in cached[2]:
                                    geopy_bairro_count += 1
                                else:
                                    geopy_rua_count += 1
                                break
                            elif cached and len(cached) == 2:
                                lats[idx], lons[idx] = cached[0], cached[1]
                                status_list[idx] = "✅ Geopy (Cache)"
                                encontrou = True
                                geopy_rua_count += 1
                                break
                            continue

                        # Requisição online segura com validação rigorosa
                        for tentativa in range(3):
                            try:
                                loc = geolocator.geocode(consulta, addressdetails=True)
                                time.sleep(geopy_delay)
                                
                                # Validação estrita: Rejeita centro de cidade e cidades erradas
                                validado = validar_resposta_geopy(loc, mun, uf, num_esperado=num)
                                if validado:
                                    lat_val, lon_val, st_val = validado
                                    lats[idx], lons[idx] = lat_val, lon_val
                                    status_list[idx] = st_val
                                    cache_geopy[chave] = [lat_val, lon_val, st_val]
                                    encontrou = True
                                    if "Bairro" in st_val:
                                        geopy_bairro_count += 1
                                    else:
                                        geopy_rua_count += 1
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
                        status_list[idx] = "❌ Não Encontrado (Sem precisão de rua)"
                        falhas_count += 1

                    prog_geopy = 60 + int(((k + 1) / total_pendentes) * 40)
                    barra_progresso.progress(prog_geopy, text=f"🌍 Geopy: {k + 1}/{total_pendentes} processados...")
                    m_geopy_exato.metric("Geopy Exato/Rua", f"{geopy_rua_count}")
                    m_geopy_bairro.metric("Geopy Bairro", f"{geopy_bairro_count}")
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
            st.success(f"Geocodificação concluída em **{tempo_total:.2f} segundos** com garantia de precisão estrita!")

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
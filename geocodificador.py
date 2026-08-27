"""
Geocodificador Inteligente de Alta Performance
Integração CNEFE (IBGE 2022) via DuckDB/Parquet + ArcGIS World Geocoder + Google Maps API + IA de Endereços
Otimizado para Lotes Grandes (10.000+ linhas) com Memoização, Conexões Persistentes e Prevenção de OOM.
"""

import os
import sys
import time
import json
import re
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Garante que o diretório onde o script está localizado esteja no sys.path
DIRETORIO_RAIZ = Path(__file__).resolve().parent
if str(DIRETORIO_RAIZ) not in sys.path:
    sys.path.insert(0, str(DIRETORIO_RAIZ))

import pandas as pd
import duckdb
from rapidfuzz import process, fuzz
import streamlit as st

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
ARQUIVO_IBGE_MUNICIPIOS = str(DIRETORIO_RAIZ / "ibge_municipios.json")
CACHE_GEOPY_ARQUIVO = str(DIRETORIO_RAIZ / "cache_geopy.json")
DEFAULT_SCORE_CUTOFF = 75

# ============================================================
# DICIONÁRIOS DE EXPANSÃO E NORMALIZAÇÃO DE LOGRADOUROS
# ============================================================

PREFIXOS_EXPANSAO = {
    "R": "RUA", "R.": "RUA", "RUA": "RUA",
    "AV": "AVENIDA", "AV.": "AVENIDA", "AVN": "AVENIDA", "AVENIDA": "AVENIDA",
    "AL": "ALAMEDA", "AL.": "ALAMEDA", "ALAMEDA": "ALAMEDA",
    "TV": "TRAVESSA", "TV.": "TRAVESSA", "TRAV": "TRAVESSA", "TRAV.": "TRAVESSA", "TRAVESSA": "TRAVESSA",
    "PC": "PRACA", "PC.": "PRACA", "PCA": "PRACA", "PCA.": "PRACA", "PRACA": "PRACA",
    "JD": "JARDIM", "JD.": "JARDIM", "JARDIM": "JARDIM",
    "DR": "DOUTOR", "DR.": "DOUTOR", "DRA": "DOUTORA", "DRA.": "DOUTORA", "DOUTOR": "DOUTOR",
    "PROF": "PROFESSOR", "PROF.": "PROFESSOR", "PROFA": "PROFESSORA", "PROFA.": "PROFESSORA", "PROFESSOR": "PROFESSOR",
    "CEL": "CORONEL", "CEL.": "CORONEL", "CORONEL": "CORONEL",
    "EST": "ESTRADA", "EST.": "ESTRADA", "ESTR": "ESTRADA", "ESTR.": "ESTRADA", "ESTRADA": "ESTRADA",
    "ROD": "RODOVIA", "ROD.": "RODOVIA", "RODOVIA": "RODOVIA",
    "BR": "BARAO", "BR.": "BARAO", "BARAO": "BARAO", "BAR": "BARAO",
    "TEN": "TENENTE", "TEN.": "TENENTE", "TENENTE": "TENENTE",
    "MAJ": "MAJOR", "MAJ.": "MAJOR", "MAJOR": "MAJOR",
    "CAP": "CAPITAO", "CAP.": "CAPITAO", "CAPITAO": "CAPITAO",
    "MAL": "MARECHAL", "MAL.": "MARECHAL", "MARECHAL": "MARECHAL",
    "STA": "SANTA", "STA.": "SANTA", "SANTA": "SANTA",
    "STO": "SANTO", "STO.": "SANTO", "SANTO": "SANTO", "SAO": "SAO", "S.": "SAO", "S": "SAO",
    "PQ": "PARQUE", "PQ.": "PARQUE", "PARQUE": "PARQUE",
    "VL": "VILA", "VL.": "VILA", "VLA": "VILA", "VLA.": "VILA", "VILA": "VILA",
    "VIEL": "VIELA", "VIEL.": "VIELA", "VIELA": "VIELA",
    "VE": "VEREADOR", "VE.": "VEREADOR", "VER": "VEREADOR", "VER.": "VEREADOR", "VR": "VEREADOR", "VEREADOR": "VEREADOR",
    "DEP": "DEPUTADO", "DEP.": "DEPUTADO", "DEPUTADO": "DEPUTADO",
    "SEN": "SENADOR", "SEN.": "SENADOR", "SENADOR": "SENADOR",
    "DES": "DESEMBARGADOR", "DES.": "DESEMBARGADOR", "DESEMBARGADOR": "DESEMBARGADOR",
    "PE": "PADRE", "PE.": "PADRE", "PADRE": "PADRE",
    "PTO": "PREFEITO", "PTO.": "PREFEITO", "PREFEITO": "PREFEITO",
    "ENG": "ENGENHEIRO", "ENG.": "ENGENHEIRO", "ENGENHEIRO": "ENGENHEIRO",
    "LOT": "LOTEAMENTO", "LOT.": "LOTEAMENTO", "LOTEAMENTO": "LOTEAMENTO",
    "CONJ": "CONJUNTO", "CONJ.": "CONJUNTO", "CONJUNTO": "CONJUNTO",
    "RES": "RESIDENCIAL", "RES.": "RESIDENCIAL", "RESIDENCIAL": "RESIDENCIAL",
    "BC": "BECO", "BC.": "BECO", "BECO": "BECO",
    "PAS": "PASSEIO", "PAS.": "PASSEIO", "PASSEIO": "PASSEIO",
    "LRG": "LARGO", "LRG.": "LARGO", "LARGO": "LARGO",
    "SERV": "SERVIDAO", "SERV.": "SERVIDAO", "SERVIDAO": "SERVIDAO",
}

NUMEROS_EXTENSO = {
    "01": "UM", "1": "UM", "02": "DOIS", "2": "DOIS", "03": "TRES", "3": "TRES",
    "04": "QUATRO", "4": "QUATRO", "05": "CINCO", "5": "CINCO", "06": "SEIS", "6": "SEIS",
    "07": "SETE", "7": "SETE", "08": "OITO", "8": "OITO", "09": "NOVE", "9": "NOVE",
    "10": "DEZ", "11": "ONZE", "12": "DOZE", "13": "TREZE", "14": "QUATORZE", "15": "QUINZE"
}

CORRECOES_COMUNS = [
    (r"\bWHASHINGTON\b", "WASHINGTON"),
    (r"\bGER\+NIMO\b", "GERONIMO"),
    (r"\bANT\+NIO\b", "ANTONIO"),
    (r"\bJO\+O\b", "JOAO"),
    (r"\bS\+O\b", "SAO"),
    (r"\bJOS\+\b", "JOSE"),
    (r"\bIP\-\b", "IPE"),
    (r"\bIP\+\b", "IPE"),
]

RUIDOS_REGEX = [
    r"\bCOMPLEMENTO\s*:.*$",
    r"\bENTREGAR\s+(?:NO|NA)?\s+NUMERO.*$",
    r"\bCASA\s*\d*.*$",
    r"\bCS\s*\d*.*$",
    r"\bAPTO?\s*\d*.*$",
    r"\bAPT\s*\d*.*$",
    r"\bSOBRADO\b.*$",
    r"\bFUNDOS?\b.*$",
    r"\bFDS\b.*$",
    r"\bSALA\s*\d*.*$",
    r"\bBLOCO\s*[A-Z0-9]*.*$",
    r"\bBL\s*[A-Z0-9]*.*$",
    r"\bQUADRA\s*\d+.*$",
    r"\bQD\s*\d+.*$",
    r"\bLOTE\s*\d+.*$",
    r"\bLT\s*\d+.*$",
    r"\bGALPAO\b.*$",
    r"\bKM\s*\d+.*$",
    r"\bCHACARA\s*\d*.*$",
    r"\bSITIO\s*\d*.*$",
    r"\bESQUINA\s+COM\b.*$",
    r"\bESQ\.?\s+COM\b.*$",
    r"\bPERTO\s+(?:DE|DO|DA)?\b.*$",
    r"\bPROXIMO\s+(?:A|AO|DA)?\b.*$",
    r"\bCLINICA\b.*$",
    r"\bESCRITORIO\b.*$",
    r"\bBURG(?:ER|UER)\w*\b.*$",
    r"\bFRIGORIFICO\b.*$",
    r"\bJAPABURGUER\b.*$",
]

TIPOS_LOGRADOURO_REGEX = re.compile(
    r"^(RUA|AVENIDA|ALAMEDA|TRAVESSA|PRACA|ESTRADA|RODOVIA|ROD|VIELA|BECO|LARGO|PRAIA|SERVIDAO|PASSEIO|PARQUE|LOTEAMENTO|CONJUNTO|RESIDENCIAL)\s+",
    re.I
)

# ============================================================
# FUNÇÕES DE TRATAMENTO E NORMALIZAÇÃO DE ENDEREÇOS
# ============================================================

def normalizar_texto(valor):
    """Remove acentos, caracteres especiais corrompidos e padroniza em caixa alta."""
    if valor is None or pd.isna(valor):
        return ""
    t = str(valor).upper().strip()
    t = t.replace("¦+", "C").replace("+", "O").replace("¦", "C").replace("?", "A")
    for padrao, subst in CORRECOES_COMUNS:
        t = re.sub(padrao, subst, t, flags=re.I)
    t = unicodedata.normalize("NFKD", t)
    t = t.encode("ASCII", "ignore").decode("ASCII")
    t = re.sub(r"[,;|]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def normalizar_municipio(valor):
    """Limpa nome do município removendo siglas de estado agregadas (ex: SP-CAMPINAS -> CAMPINAS)."""
    t = normalizar_texto(valor)
    t = re.sub(r"^(SP|RJ|MG|PR|MS|GO|BA|ES|SC|RS|MT|PE|CE|PA|PB|AL|SE|RN|PI|MA|TO|RO|AC|AM|RR|AP|DF)[-_\s]+", "", t)
    t = re.sub(r"[-_\s]+(SP|RJ|MG|PR|MS|GO|BA|ES|SC|RS|MT|PE|CE|PA|PB|AL|SE|RN|PI|MA|TO|RO|AC|AM|RR|AP|DF)$", "", t)
    t = re.sub(r"\((SP|RJ|MG|PR|MS|GO|BA|ES|SC|RS|MT|PE|CE|PA|PB|AL|SE|RN|PI|MA|TO|RO|AC|AM|RR|AP|DF)\)", "", t)
    return t.strip()

def limpar_numero(valor):
    """Padroniza números prediais, removendo strings como S/N, 0, None."""
    if valor is None or pd.isna(valor):
        return ""
    t = str(valor).strip()
    if t.lower() in {"", "nan", "none", "null", "0", "0.0", "s/n", "sn", "s/nº", "sem numero", "sem nº", "00"}:
        return ""
    if re.fullmatch(r"\d+\.0", t):
        t = t[:-2]
    return t

def extrair_numero_endereco(rua, numero):
    """Inteligência para extrair número predial contido dentro do campo de logradouro."""
    rua = normalizar_texto(rua)
    num = limpar_numero(numero)
    
    if num:
        rua = re.sub(rf"\b{re.escape(num)}\b", "", rua).strip(" ,-")
        return rua, num

    m = re.fullmatch(r"(\d{1,4}\s+DE\s+[A-Z]+)\s+(\d{1,5})", rua)
    if m:
        return m.group(1), m.group(2)

    m = re.fullmatch(r"(RUA\s+\d{1,4})\s+(\d{1,5})", rua)
    if m:
        return m.group(1), m.group(2)

    m = re.search(r",\s*(\d+[A-Z]?)\b", rua)
    if m:
        num = m.group(1)
        rua = rua[:m.start()].strip(" ,-")
        return rua, num

    m = re.search(r"(?:\bN[Oº°]?|\bNUM(?:ERO)?)[\s.:#-]*(\d+[A-Z]?)\b", rua)
    if m:
        num = m.group(1)
        rua = rua[:m.start()].strip(" ,-#")
        return rua, num

    m = re.search(r"\s+(\d{1,6}[A-Z]?)$", rua)
    if m:
        num = m.group(1)
        rua = rua[:m.start()].strip(" ,-")
        return rua, num

    return rua, ""

def remover_complementos(texto):
    """Remove complementos prediais e ruídos de entrega que atrapalham o geocoding."""
    t = texto
    for padrao in RUIDOS_REGEX:
        t = re.sub(padrao, "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip(" ,-.")

def expandir_prefixo(texto):
    """Expande abreviações como 'R.', 'AV.', 'BR.', 'CEL.', 'VE.'."""
    t = texto.strip()
    tokens = t.split()
    if not tokens:
        return ""
    if tokens[0] in PREFIXOS_EXPANSAO:
        tokens[0] = PREFIXOS_EXPANSAO[tokens[0]]
    if len(tokens) > 1 and tokens[1] in PREFIXOS_EXPANSAO:
        tokens[1] = PREFIXOS_EXPANSAO[tokens[1]]
    return " ".join(tokens)

def sem_prefixo_tipo_rua(texto):
    """Retorna o nome do logradouro sem o tipo inicial (ex: 'RUA PAULISTA' -> 'PAULISTA')."""
    return TIPOS_LOGRADOURO_REGEX.sub("", texto).strip()

def preparar_endereco(rua, numero):
    """Pipeline de IA / Heurística para estruturar um endereço em múltiplas variantes de busca."""
    rua, numero = extrair_numero_endereco(rua, numero)
    rua = remover_complementos(rua)
    rua = expandir_prefixo(rua)
    rua = re.sub(r"\s+", " ", rua).strip()

    rua_sem_num = re.sub(
        rf"\s+{re.escape(numero)}[A-Z]?\s*$" if numero else r"$^",
        "", rua, flags=re.I,
    ).strip()

    variantes = []
    def add(v):
        v = re.sub(r"\s+", " ", v).strip(" ,-")
        if v and v not in variantes:
            variantes.append(v)

    add(rua_sem_num)
    add(expandir_prefixo(rua_sem_num))
    add(sem_prefixo_tipo_rua(rua_sem_num))
    add(rua_sem_num.replace("-", " "))
    add(rua_sem_num.replace("/", " "))
    
    sem_conectivos = re.sub(r"\b(DE|DA|DO|DOS|DAS)\b", "", rua_sem_num)
    sem_conectivos = re.sub(r"\s+", " ", sem_conectivos).strip()
    if sem_conectivos:
        add(sem_conectivos)
        add(sem_prefixo_tipo_rua(sem_conectivos))

    for num_digito, extenso in NUMEROS_EXTENSO.items():
        if re.search(rf"\b{num_digito}\b", rua_sem_num):
            add(re.sub(rf"\b{num_digito}\b", extenso, rua_sem_num))

    return {"rua": rua_sem_num, "numero": numero, "variantes": variantes}

# ============================================================
# MOTORES DE GEOCODIFICAÇÃO DE ALTA PRECISÃO
# ============================================================

def criar_sessao_http():
    """Cria uma sessão HTTP com connection pooling e retries automáticos."""
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def consultar_arcgis(rua_raw, num_raw, bairro_raw, mun_raw, uf_raw, session=None):
    """
    Consulta o ArcGIS World Geocoding Service (Esri).
    Altíssima precisão no Brasil com suporte a números prediais (PointAddress)
    e validação estrita anti-erro de cidade.
    """
    prep = preparar_endereco(rua_raw, num_raw)
    rua = prep["rua"]
    num = prep["numero"]
    mun = normalizar_municipio(mun_raw)
    uf = normalizar_texto(uf_raw)[:2] if uf_raw else "SP"
    bairro = normalizar_texto(bairro_raw) if bairro_raw and bairro_raw not in ["(Não informado)", "CENTRO", "RURAL"] else ""

    if not rua or not mun:
        return None

    consultas = []
    if num:
        if bairro:
            consultas.append(f"{rua}, {num}, {bairro}, {mun}, {uf}, Brasil")
        consultas.append(f"{rua}, {num}, {mun}, {uf}, Brasil")
        consultas.append(f"{rua}, {num}, {mun}, Brasil")
    
    if bairro:
        consultas.append(f"{rua}, {bairro}, {mun}, {uf}, Brasil")
    consultas.append(f"{rua}, {mun}, {uf}, Brasil")
    consultas.append(f"{rua}, {mun}, Brasil")

    url = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    http = session or requests

    for query_str in consultas:
        params = {
            "SingleLine": query_str,
            "f": "json",
            "outFields": "Match_addr,Addr_type,City,Subregion,Region,Postal",
            "maxLocations": 3,
            "countryCode": "BRA"
        }
        try:
            r = http.get(url, params=params, timeout=5)
            data = r.json()
            candidates = data.get("candidates", [])
            for cand in candidates:
                score = cand.get("score", 0)
                if score < 75:
                    continue
                attrs = cand.get("attributes", {})
                addr_type = attrs.get("Addr_type", "")
                city_ret = normalizar_municipio(attrs.get("City", ""))
                subregion = normalizar_municipio(attrs.get("Subregion", ""))
                loc = cand.get("location", {})
                
                # Validação Estrita de Cidade
                sim_cidade = max(fuzz.token_sort_ratio(mun, city_ret), fuzz.token_sort_ratio(mun, subregion))
                if sim_cidade < 70 and mun not in city_ret and city_ret not in mun:
                    continue

                lat, lon = loc.get("y"), loc.get("x")
                if not lat or not lon:
                    continue

                if addr_type in ["PointAddress", "StreetAddress"]:
                    return lat, lon, "✅ Exato (Número/Imóvel)", cand.get("address", "")
                elif addr_type in ["StreetName", "StreetInt"]:
                    return lat, lon, "✅ Logradouro (Rua)", cand.get("address", "")
                elif addr_type in ["Locality", "Sublocality", "Neighborhood"] and bairro and fuzz.token_sort_ratio(bairro, addr_type) >= 70:
                    return lat, lon, f"🟡 Bairro ({bairro})", cand.get("address", "")
        except Exception:
            pass

    return None

def consultar_google_maps(rua_raw, num_raw, bairro_raw, mun_raw, uf_raw, api_key, session=None):
    """
    Consulta a API oficial do Google Maps (Geocoding API).
    Precisão máxima absoluta com resolução de coordenadas no telhado (ROOFTOP).
    """
    if not api_key:
        return None

    prep = preparar_endereco(rua_raw, num_raw)
    rua = prep["rua"]
    num = prep["numero"]
    mun = normalizar_municipio(mun_raw)
    uf = normalizar_texto(uf_raw)[:2] if uf_raw else "SP"
    bairro = normalizar_texto(bairro_raw) if bairro_raw and bairro_raw not in ["(Não informado)", "CENTRO", "RURAL"] else ""

    if not rua or not mun:
        return None

    endereco = f"{rua}, {num}".strip(" ,-") if num else rua
    if bairro:
        endereco += f", {bairro}"
    endereco += f", {mun} - {uf}, Brasil"

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": endereco,
        "key": api_key,
        "language": "pt-BR",
        "region": "br"
    }
    http = session or requests

    try:
        r = http.get(url, params=params, timeout=5)
        data = r.json()
        if data.get("status") == "OK" and data.get("results"):
            res = data["results"][0]
            loc = res.get("geometry", {}).get("location", {})
            loc_type = res.get("geometry", {}).get("location_type", "")
            tipos = set(res.get("types", []))
            
            cidade_ok = False
            for comp in res.get("address_components", []):
                comp_types = comp.get("types", [])
                if "administrative_area_level_2" in comp_types or "locality" in comp_types:
                    cidade_cand = normalizar_municipio(comp.get("long_name", ""))
                    if fuzz.token_sort_ratio(mun, cidade_cand) >= 70 or mun in cidade_cand or cidade_cand in mun:
                        cidade_ok = True
                        break

            if not cidade_ok:
                return None

            lat = loc.get("lat")
            lon = loc.get("lng")
            match_addr = res.get("formatted_address", "")

            if loc_type == "ROOFTOP" or ("street_number" in tipos) or ("premise" in tipos):
                return lat, lon, "✅ Google Maps Exato (Número)", match_addr
            elif loc_type in ["RANGE_INTERPOLATED", "GEOMETRIC_CENTER"] and "route" in tipos:
                return lat, lon, "✅ Google Maps Logradouro (Rua)", match_addr
            elif "sublocality" in tipos or "neighborhood" in tipos:
                return lat, lon, "🟡 Google Maps Bairro", match_addr
    except Exception:
        pass

    return None

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

def eh_url_valida(url):
    """Verifica se a URL é válida e não é apenas um placeholder de exemplo."""
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    placeholders = ["SEU_USUARIO", "SEU_DATASET", "EXEMPLO", "YOUR_USER", "YOUR_DATASET"]
    for p in placeholders:
        if p.lower() in u.lower():
            return False
    return True

def obter_url_secret_cnefe(estado):
    """Obtém a URL do CNEFE a partir do st.secrets ou variável de ambiente."""
    uf = normalizar_texto(estado).upper()[:2]
    chaves = [f"CNEFE_{uf}_URL", "CNEFE_URL", f"cnefe_{uf.lower()}_url", "cnefe_url"]
    
    try:
        if hasattr(st, "secrets"):
            for k in chaves:
                if k in st.secrets:
                    val = str(st.secrets[k]).strip()
                    if eh_url_valida(val):
                        return val
    except Exception:
        pass

    for k in chaves:
        val = os.getenv(k)
        if val and eh_url_valida(val):
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
    if url_remota and eh_url_valida(url_remota):
        return url_remota.strip()

    url_secret = obter_url_secret_cnefe(estado)
    if url_secret and eh_url_valida(url_secret):
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

        if mun_raw in uf_dict:
            cod = uf_dict[mun_raw]
            cods_ibge.append(cod)
            cache_mun_match[chave] = cod
            continue

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
        st.warning(f"⚠️ Não foi possível carregar a base CNEFE remota/local: {e}. O processamento continuará automaticamente pelos motores ArcGIS / Google Maps.")
        con.close()
        return {}
    finally:
        try:
            con.close()
        except Exception:
            pass

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

    candidatos = {}
    for v in consulta["variantes"]:
        if v in ruas_dict:
            candidatos[v] = 100.0

    if not candidatos:
        for v in consulta["variantes"]:
            matches = process.extract(
                v, ruas_list,
                scorer=fuzz.WRatio,
                limit=5,
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
    **Motor Multi-Camadas de Alta Performance (Capacidade para 10.000+ linhas)**:
    - 🏢 **Camada 1: CNEFE IBGE 2022**: Busca ultrarrápida vetorial em memória via DuckDB e RapidFuzz.
    - 🗺️ **Camada 2: Google Maps API**: Precisão máxima de coordenadas prediais no telhado (opcional).
    - 🌐 **Camada 3: ArcGIS World Geocoder (Esri)**: Geocodificador empresarial paralelo com localização exata de número predial e rua.
    - ⚡ **Otimização de Escala**: Memoização inteligente, HTTP Connection Pooling (10 threads) e prevenção de timeouts.
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

        df_completo = pd.read_excel(caminho_temp)
        total_linhas_planilha = len(df_completo)
        st.success(f"Planilha carregada com sucesso! Total de registros: **{total_linhas_planilha}**")

        df_completo = df_completo.loc[:, ~df_completo.columns.duplicated()].copy()
        colunas = list(df_completo.columns)
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

        # Seletor de Escopo de Execução (Tudo vs Intervalo)
        st.subheader("🎯 Escopo de Processamento")
        col_escopo1, col_escopo2 = st.columns(2)
        with col_escopo1:
            modo_execucao = st.radio(
                "Escolha o escopo de linhas a processar:",
                ["Processar Planilha Completa", "Processar Intervalo Específico (Lotes)"],
                horizontal=True
            )
        
        inicio_linha = 0
        fim_linha = total_linhas_planilha
        if modo_execucao == "Processar Intervalo Específico (Lotes)":
            with col_escopo2:
                c_ini, c_fim = st.columns(2)
                with c_ini:
                    inicio_linha = st.number_input("Linha Inicial (1-indexada):", min_value=1, max_value=total_linhas_planilha, value=1) - 1
                with c_fim:
                    fim_linha = st.number_input("Linha Final:", min_value=1, max_value=total_linhas_planilha, value=min(2000, total_linhas_planilha))

        df = df_completo.iloc[inicio_linha:fim_linha].copy().reset_index(drop=True)

        cols_preview = [c for c in [col_rua, col_num, col_bairro, col_cep, col_mun, col_uf] if c]
        cols_preview_unicas = list(dict.fromkeys(cols_preview))
        st.dataframe(df[cols_preview_unicas].head(10))

        if st.button("🚀 Iniciar Geocodificação Turbo", type="primary"):
            t_inicio = time.time()
            total = len(df)
            lats = [""] * total
            lons = [""] * total
            status_list = [""] * total

            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            m_total = m_col1.metric("Total Lote", f"{total}")
            m_cnefe_exato = m_col2.metric("CNEFE Exato", "0")
            m_cnefe_rua = m_col3.metric("CNEFE Rua", "0")
            m_web_exato = m_col4.metric("ArcGIS/Google Exato", "0")
            m_web_rua = m_col5.metric("ArcGIS/Google Rua", "0")
            m_falha = m_col6.metric("Não Encontrado", "0")

            barra_progresso = st.progress(0, text="🔍 Mapeando municípios pelo IBGE...")

            # 1. Mapear códigos IBGE
            cods_ibge = resolver_codigos_ibge(df, col_mun, col_uf, ibge_base)
            df['__cod_ibge'] = cods_ibge

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

            # 2. Executar matching CNEFE com memoização de alta performance
            cnefe_exato_count = 0
            cnefe_rua_count = 0
            indices_pendentes = []
            cnefe_memo_cache = {}
            prep_memo_cache = {}

            ultimo_update_ui = time.time()

            for i in range(total):
                row = df.iloc[i]
                cod = row['__cod_ibge']
                uf = normalizar_texto(row.get(col_uf, "SP"))[:2] if col_uf else "SP"
                if uf not in cnefe_indices_por_uf:
                    uf = "SP"

                cnefe_idx = cnefe_indices_por_uf.get(uf, {})
                rua_raw = str(row.get(col_rua, ""))
                num_raw = str(row.get(col_num, "")) if col_num else ""
                
                chave_cnefe = (cod, rua_raw, num_raw)
                if chave_cnefe in cnefe_memo_cache:
                    res = cnefe_memo_cache[chave_cnefe]
                else:
                    chave_prep = (rua_raw, num_raw)
                    if chave_prep not in prep_memo_cache:
                        prep_memo_cache[chave_prep] = preparar_endereco(rua_raw, num_raw)
                    consulta = prep_memo_cache[chave_prep]

                    res = None
                    if cod and not pd.isna(cod) and int(cod) in cnefe_idx:
                        res = buscar_endereco_no_indice(consulta, int(cod), cnefe_idx, score_cutoff=limiar_score)
                    cnefe_memo_cache[chave_cnefe] = res

                if res:
                    lats[i], lons[i], status_list[i] = res
                    if "Exato" in status_list[i]:
                        cnefe_exato_count += 1
                    else:
                        cnefe_rua_count += 1
                else:
                    indices_pendentes.append(i)

                # Atualização throttled da UI a cada 100 linhas ou 1s para evitar saturação
                agora = time.time()
                if (i + 1) % 100 == 0 or (agora - ultimo_update_ui) > 1.2 or (i + 1) == total:
                    prog = int(((i + 1) / total) * 40)
                    barra_progresso.progress(prog, text=f"⚡ CNEFE: {i + 1}/{total} processados...")
                    m_cnefe_exato.metric("CNEFE Exato", f"{cnefe_exato_count}")
                    m_cnefe_rua.metric("CNEFE Rua", f"{cnefe_rua_count}")
                    ultimo_update_ui = agora

            # 3. Motores de Alta Precisão Web (Google Maps / ArcGIS World Geocoder) em Paralelo
            web_exato_count = 0
            web_rua_count = 0
            falhas_count = 0
            total_pendentes = len(indices_pendentes)

            if total_pendentes > 0:
                barra_progresso.progress(45, text=f"🌍 Geocodificando {total_pendentes} endereços pendentes com precisão Google Maps / ArcGIS...")
                session = criar_sessao_http()
                web_memo_cache = {}

                def processar_item_web(idx):
                    row = df.iloc[idx]
                    rua_val = str(row.get(col_rua, ""))
                    num_val = str(row.get(col_num, "")) if col_num else ""
                    bairro_val = str(row.get(col_bairro, "")) if col_bairro else ""
                    mun_val = str(row.get(col_mun, ""))
                    uf_val = str(row.get(col_uf, "SP")) if col_uf else "SP"

                    chave_web = (rua_val, num_val, bairro_val, mun_val, uf_val)
                    if chave_web in web_memo_cache:
                        return idx, web_memo_cache[chave_web]

                    # 1. Tenta Google Maps API se chave fornecida
                    if google_api_key:
                        res_google = consultar_google_maps(rua_val, num_val, bairro_val, mun_val, uf_val, google_api_key, session=session)
                        if res_google:
                            lat, lon, st_p, match_addr = res_google
                            ret = (lat, lon, st_p)
                            web_memo_cache[chave_web] = ret
                            return idx, ret

                    # 2. Tenta ArcGIS World Geocoder (Alta precisão)
                    res_arc = consultar_arcgis(rua_val, num_val, bairro_val, mun_val, uf_val, session=session)
                    if res_arc:
                        lat, lon, st_p, match_addr = res_arc
                        ret = (lat, lon, st_p)
                        web_memo_cache[chave_web] = ret
                        return idx, ret

                    web_memo_cache[chave_web] = (None, None, None)
                    return idx, (None, None, None)

                # Processamento com Pool de 10 Threads
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futuros = {executor.submit(processar_item_web, idx): idx for idx in indices_pendentes}
                    count = 0
                    for fut in as_completed(futuros):
                        count += 1
                        idx, (lat_val, lon_val, st_val) = fut.result()
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

                        agora = time.time()
                        if count % 50 == 0 or (agora - ultimo_update_ui) > 1.2 or count == total_pendentes:
                            prog = 45 + int((count / total_pendentes) * 50)
                            barra_progresso.progress(prog, text=f"🌍 Geocodificando Web: {count}/{total_pendentes} concluídos...")
                            m_web_exato.metric("ArcGIS/Google Exato", f"{web_exato_count}")
                            m_web_rua.metric("ArcGIS/Google Rua", f"{web_rua_count}")
                            m_falha.metric("Não Encontrado", f"{falhas_count}")
                            ultimo_update_ui = agora

            t_total = time.time() - t_inicio
            barra_progresso.progress(100, text=f"✨ Concluído com sucesso em {t_total:.2f} segundos!")

            df_resultado = df.copy()
            if '__cod_ibge' in df_resultado.columns:
                df_resultado.drop(columns=['__cod_ibge'], inplace=True)

            df_resultado["Latitude"] = lats
            df_resultado["Longitude"] = lons
            df_resultado["Status_Geocodificacao"] = status_list

            # Salva na session_state para manter persistência
            st.session_state["df_resultado"] = df_resultado

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
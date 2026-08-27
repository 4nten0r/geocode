"""
Módulo de Inteligência e Limpeza de Endereços Brasileiros (endereco_ia.py)
Aplica heurísticas, correção fonética/ortográfica e estruturação de dados de logradouros.
"""

import re
import unicodedata
import pandas as pd
from rapidfuzz import fuzz

# ============================================================
# DICIONÁRIOS DE EXPANSÃO E NORMALIZAÇÃO
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
    "BR": "BARAO", "BR.": "BARAO", "BARAO": "BARAO",
    "TEN": "TENENTE", "TEN.": "TENENTE", "TENENTE": "TENENTE",
    "MAJ": "MAJOR", "MAJ.": "MAJOR", "MAJOR": "MAJOR",
    "CAP": "CAPITAO", "CAP.": "CAPITAO", "CAPITAO": "CAPITAO",
    "MAL": "MARECHAL", "MAL.": "MARECHAL", "MARECHAL": "MARECHAL",
    "STA": "SANTA", "STA.": "SANTA", "SANTA": "SANTA",
    "STO": "SANTO", "STO.": "SANTO", "SANTO": "SANTO", "SAO": "SAO", "S.": "SAO", "S": "SAO",
    "PQ": "PARQUE", "PQ.": "PARQUE", "PARQUE": "PARQUE",
    "VL": "VILA", "VL.": "VILA", "VLA": "VILA", "VLA.": "VILA", "VILA": "VILA",
    "VIEL": "VIELA", "VIEL.": "VIELA", "VIELA": "VIELA",
    "LOT": "LOTEAMENTO", "LOT.": "LOTEAMENTO", "LOTEAMENTO": "LOTEAMENTO",
    "CONJ": "CONJUNTO", "CONJ.": "CONJUNTO", "CONJUNTO": "CONJUNTO",
    "RES": "RESIDENCIAL", "RES.": "RESIDENCIAL", "RESIDENCIAL": "RESIDENCIAL",
    "BC": "BECO", "BC.": "BECO", "BECO": "BECO",
    "PAS": "PASSEIO", "PAS.": "PASSEIO", "PASSEIO": "PASSEIO",
    "LRG": "LARGO", "LRG.": "LARGO", "LARGO": "LARGO",
    "SERV": "SERVIDAO", "SERV.": "SERVIDAO", "SERVIDAO": "SERVIDAO",
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
# FUNÇÕES DE TRATAMENTO
# ============================================================

def normalizar_texto(valor):
    """Remove acentos, caracteres especiais e padroniza em caixa alta."""
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
    """Limpa nome do município removendo siglas de estado agregadas (ex: SP-CAMPINAS)."""
    t = normalizar_texto(valor)
    t = re.sub(r"^(SP|RJ|MG|PR|MS|GO|BA|ES|SC|RS|MT|PE|CE|PA|PB|AL|SE|RN|PI|MA|TO|RO|AC|AM|RR|AP|DF)[-_\s]+", "", t)
    t = re.sub(r"[-_\s]+(SP|RJ|MG|PR|MS|GO|BA|ES|SC|RS|MT|PE|CE|PA|PB|AL|SE|RN|PI|MA|TO|RO|AC|AM|RR|AP|DF)$", "", t)
    return t.strip()

def limpar_numero(valor):
    """Padroniza números prediais, removendo strings como S/N, 0, None."""
    if valor is None or pd.isna(valor):
        return ""
    t = str(valor).strip()
    if t.lower() in {"", "nan", "none", "null", "0", "0.0", "s/n", "sn", "s/nº", "sem numero", "sem nº"}:
        return ""
    if re.fullmatch(r"\d+\.0", t):
        t = t[:-2]
    return t

def extrair_numero_endereco(rua, numero):
    """
    Inteligência para extrair número predial contido dentro do campo de logradouro
    ou validar o campo número fornecido.
    """
    rua = normalizar_texto(rua)
    num = limpar_numero(numero)
    if num:
        return rua, num

    # Ex: '8 DE DEZEMBRO 600' -> '8 DE DEZEMBRO', '600'
    m = re.fullmatch(r"(\d{1,4}\s+DE\s+[A-Z]+)\s+(\d{1,5})", rua)
    if m:
        return m.group(1), m.group(2)

    # Ex: 'RUA 15 120' -> 'RUA 15', '120'
    m = re.fullmatch(r"(RUA\s+\d{1,4})\s+(\d{1,5})", rua)
    if m:
        return m.group(1), m.group(2)

    # Ex: 'RUA TAL, 123' ou 'RUA TAL ,123A'
    m = re.search(r",\s*(\d+[A-Z]?)\b", rua)
    if m:
        num = m.group(1)
        rua = rua[:m.start()].strip(" ,-")
        return rua, num

    # Ex: 'RUA TAL Nº 123' ou 'RUA TAL NUMERO 123'
    m = re.search(r"(?:\bN[Oº°]?|\bNUM(?:ERO)?)[\s.:#-]*(\d+[A-Z]?)\b", rua)
    if m:
        num = m.group(1)
        rua = rua[:m.start()].strip(" ,-#")
        return rua, num

    # Ex: 'AV BRASIL 1500' (no final da string)
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
    """Expande abreviações como 'R.', 'AV.', 'BR.', 'CEL.'."""
    t = texto.strip()
    m = re.match(r"^([A-Z]{1,5}\.?)\s+(.+)$", t)
    if m:
        pref = m.group(1)
        if pref in PREFIXOS_EXPANSAO:
            return PREFIXOS_EXPANSAO[pref] + " " + m.group(2)
    return t

def sem_prefixo_tipo_rua(texto):
    """Retorna o nome do logradouro sem o tipo inicial (ex: 'RUA PAULISTA' -> 'PAULISTA')."""
    return TIPOS_LOGRADOURO_REGEX.sub("", texto).strip()

def preparar_endereco(rua, numero):
    """
    Pipeline completo de IA / Heurística para estruturar um endereço em múltiplas variantes de busca.
    """
    rua, numero = extrair_numero_endereco(rua, numero)
    rua = remover_complementos(rua)
    rua = expandir_prefixo(rua)
    rua = re.sub(r"\s+", " ", rua).strip()

    # Remove número se ainda estiver no final
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
    
    # Variante sem conectivos 'DE', 'DA', 'DO', 'DOS', 'DAS'
    sem_conectivos = re.sub(r"\b(DE|DA|DO|DOS|DAS)\b", "", rua_sem_num)
    sem_conectivos = re.sub(r"\s+", " ", sem_conectivos).strip()
    if sem_conectivos:
        add(sem_conectivos)
        add(sem_prefixo_tipo_rua(sem_conectivos))

    return {"rua": rua_sem_num, "numero": numero, "variantes": variantes}

def calcular_similaridade(a, b):
    """
    Calcula score ponderado de similaridade entre duas strings usando C++ RapidFuzz.
    """
    if not a or not b:
        return 0.0
    return (
        0.40 * fuzz.token_set_ratio(a, b) +
        0.30 * fuzz.token_sort_ratio(a, b) +
        0.30 * fuzz.WRatio(a, b)
    )


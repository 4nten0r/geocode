# 🌍 Geocodificador IA Turbo (IBGE CNEFE + Geopy)

Sistema de alta performance para geocodificação de endereços brasileiros em lote com interface Streamlit.

---

## ⚡ Recursos Principais

- **Motor CNEFE IBGE 2022**: Consultas vetoriais ultrarrápidas em DuckDB com indexação em memória e *RapidFuzz* nativo (C++).
- **Módulo de IA para Endereços (`endereco_ia.py`)**:
  - Correção de codificação e erros fonéticos/ortográficos.
  - Expansão automática de abreviações e títulos (`AV.`, `R.`, `BR.`, `MAL.`, etc.).
  - Remoção de ruídos e complementos irrelevantes (`APTO`, `BLOCO`, `QUADRA`, `LOTE`, `FUNDOS`, etc.).
  - Extração inteligente de número predial embutido.
- **Mapeamento Oficial de Municípios**: Mais de 5.500 municípios do IBGE mapeados offline (`ibge_municipios.json`).
- **Fallback Geopy/Nominatim**: Para endereços não cobertos na base local, com cache persistente e proteção contra *Rate Limit* (HTTP 429).
- **Painel Interativo Streamlit**: Métricas em tempo real e visualização de mapas dos pontos localizados.

---

## 🚀 Como Executar Localmente

### 1. Clonar o repositório
```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <PASTA_DO_PROJETO>
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Iniciar a aplicação
```bash
streamlit run geocodificador.py
```

---

## ☁️ Deploy no GitHub e Nuvem

### Deploy no GitHub:
```bash
git init
git add .
git commit -m "feat: Geocodificador IA Turbo inicial"
git branch -M main
git remote add origin https://github.com/<SEU_USUARIO>/<SEU_REPOSITORIO>.git
git push -u origin main
```

### Deploy no Streamlit Community Cloud (Recomendado):
1. Acesse [share.streamlit.io](https://share.streamlit.io).
2. Conecte sua conta do GitHub.
3. Selecione o repositório, a branch `main` e o arquivo principal `geocodificador.py`.
4. Clique em **Deploy**!

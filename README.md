# DMS-Educação × Censo Escolar (local)

Aplicação **Streamlit** para importar dados agregados da **DMS-Educação** e do **Censo Escolar**, normalizar, cruzar e produzir indicadores. **Tudo roda offline** neste equipamento — sem APIs externas, sem IA e sem nuvem.

## Requisitos

- **Python 3.12+**
- Ficheiros de entrada em **CSV** ou **XLSX** (.xlsx)

## Instalação

Na pasta `app`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(macOS/Linux: `source .venv/bin/activate`)

## Execução

A partir da pasta **`app`** (onde está `app.py`):

```powershell
python -m streamlit run app.py
```

O navegador abrirá por defeito (ex.: `http://localhost:8501`).

## Etapa 1 — Resumo

Upload de **DMS** e **Censo** (CSV/XLSX), visualização de nome, linhas e colunas.

## Etapa 2 — O que faz

1. **DMS:** deteção automática da linha de **cabeçalho** (ignora linhas em branco à frente e penaliza linhas tipo “título” de relatório); corrige colunas **Unnamed**/vazias; desduplica nomes.
2. **Censo:** leitura pelo fluxo padrão (`file_io`) + mesmo saneamento de colunas.
3. **Mapeamento dinâmico** (selectboxes): CNPJ, razão social e quantidade na DMS; CNPJ, nome da escola e matrículas no Censo.
4. **CNPJ:** apenas dígitos, `zfill` a 14 dígitos, validação de **dígitos verificadores**; contagem **vazios / inválidos / válidos** por base.
5. **Cache** `@st.cache_data` em `services/ingest_cache.py` para repetir leituras sem reprocessar bytes.
6. **Logs** em `outputs/app.log`.
7. Preview de tabelas maior (altura fixa) + amostra de CNPJ normalizado (DMS).

**Ainda não faz:** merge (`join`), indicadores exportados, fuzzy, gráficos.

## Pastas e ficheiros principais

| Item | Utilidade |
|------|-----------|
| `app.py` | UI Streamlit |
| `services/ingest_cache.py` | Envoltório com `st.cache_data` para DMS e Censo |
| `utils/file_io.py` | CSV/XLSX sem deteção inteligente (base Censo cru) |
| `utils/dms_ingest.py` | Matriz → cabeçalho heurístico → DataFrame + `fix_unnamed_*` |
| `utils/cnpj.py` | Normalização e estatísticas por coluna |
| `uploads/` | Reservado (exports manuais opcionais) |
| `outputs/` | `app.log` e Excel nas etapas seguintes |
| `requirements.txt` | Dependências |

## CSV e encoding

Deteção de separador **`;` vs `,`** nas primeiras linhas; encodings `utf-8-sig`, `utf-8`, `latin-1`, `cp1252`. Microdados INEP costumam usar **`;`**.

## Limpar cache Streamlit

Menu **⋮** (canto superior direito) → **Clear cache**, se trocar o ficheiro mas o painel mostrar dados antigos.

## Próximo passo

Quando a Etapa 2 estiver validada, pedir **“implementar a Etapa 3”** (merge por CNPJ normalizado e `consolidado.xlsx`).

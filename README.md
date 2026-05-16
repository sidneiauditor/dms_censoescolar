# DMS-Educação × Censo Escolar (local)

Aplicação **Streamlit** offline para carregar bases por **tipo** (sem nomes fixos de ficheiro), consolidar **Censo Escola ⊕ Matrícula** em campos lógicos estáveis e cruzar com a **DMS**.

## Execução

```powershell
cd ...\app
python -m streamlit run app.py
```

## Arquitetura (resumo)

1. **Tipos de base** (`domain/dataset_kind.py`): `DMS_EDUCACAO`, `CENSO_ESCOLA`, `CENSO_MATRICULA` — independentes do ano do INEP.
2. **Campos lógicos** (`domain/census_logical.py`): papéis semânticos (`CO_ENTIDADE`, `NO_ENTIDADE`, `matriculas`, …) mapeados pelos cabeçalhos reais do CSV/XLSX.
3. **Carregamento** (`services/table_loader.py`): `@st.cache_data` por `(tipo, bytes, nome)` — o nome só serve à extensão; não há convenção `Tabela_*_AAAA.csv`.
4. **Consolidação** (`services/census_consolidator.py`): projeção das colunas físicas → nomes lógicos; **merge externo** por `CO_ENTIDADE`; metadados `censo_exercicio`, `censo_fonte_*`.
5. **Compatibilidade** (`services/ingest_cache.py`): delegação para `table_loader`.

Fluxo UI: **Carregar** (3 slots) → **Mapear** → **Consolidar Censo** → **Etapa 2** (CNPJ) → **Etapa 3** (fuzzy texto).

## Pastas relevantes

| Caminho | Função |
|---------|--------|
| `domain/` | Tipos de conjunto + especificação de campos lógicos |
| `services/table_loader.py` | Cache + encaminhamento DMS smart vs leitura plana |
| `services/census_consolidator.py` | Merge Escola⊕Matrícula |
| `services/text_fuzzy_merge.py` | Etapa 3 RapidFuzz |
| `utils/` | CSV/XLSX, texto, DMS ingest, CNPJ |

## Exercício (ano)

O campo **“Exercício do Censo”** na barra lateral não altera leitura de ficheiros; apenas grava **`censo_exercicio`** na base consolidada para rastreabilidade (2024, 2025, 2026, …).

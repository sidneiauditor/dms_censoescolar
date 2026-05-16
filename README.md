# DMS-Educação × Censo Escolar (local)

Aplicação **Streamlit** offline com contexto **municipal**: carrega bases por tipo, recorta Escola (+ Matrícula) ao município quando possível, consolida o Censo em colunas lógicas e cruza com a **DMS** com **prioridade ao CNPJ determinístico** (texto RapidFuzz só onde não há CNPJ válido na DMS).

## Execução

```powershell
cd ...\app
python -m streamlit run app.py
```

## Fluxo na UI (resumo)

1. **Uploads** — DMS, Escola INEP/export, Matrícula (opcional).
2. **Modo simples / avançado** — modo simples esconde mapeamentos INEP até ser inevitável; avançado mostra todas as associações lógicas.
3. **Etapa 0** — ano de exercício (``censo_exercicio``) + UF + município (filtro antes do merge, por defeito); no avançado pode desativar o recorte territorial.
4. **Consolidar Censo municipal** — aplicar filtro territorial bruto ⇢ recortar matrícula ao mesmo ``CO_ENTIDADE`` quando possível ⇢ ``consolidate_census_escolar``; gravar metadados `censo_ctx_*` quando há filtro.
5. **Etapa 2** — escolha as colunas físicas de contribuinte. As colunas internas são criadas com **`utils.cnpj.add_normalized_cnpj_column`** (somente dígitos, até 14 com `zfill`).
6. **Antes da Etapa 3** — `ensure_normalized_cnpj_workframes` em `app.py` reaplica sempre `add_normalized_cnpj_column` ao upload **DMS** e ao DataFrame **`censo_consolidado`**, combinando os valores persistidos em `column_map`. Se apenas existirem variantes tipo ``CNPJ_base_escola`` / ``CNPJ_base_matricula`` após Escola⊕Matrícula, `resolve_census_cnpj_physical_column` (`inferred_mapping.py`) descobre a coluna física correta e daí surge **`__cnpj_norm_censo`**.
7. **Etapa 3** — primeiro **merge igualdade estrita de CNPJ** (`services/cnpj_merge.py`); texto (`services/text_fuzzy_merge.py`) apenas para linhas onde a DMS **não** tem CNPJ normalizável/validado segundo `classify_cnpj_cell`.

## Merge determinístico (Etapa 3)

| Conceito | Colunas / comportamento |
|----------|--------------------------|
| Chave fiscal | Comparar ``__cnpj_norm_dms`` com ``__cnpj_norm_censo``. |
| `match_status_principal` | `match_cnpj_exato`, `multiplas_escolas_mesmo_cnpj`, `sem_correspondencia_cnpj`, `sem_cnpj_utilizavel_dms`, `cnpj_dms_invalido`, `match_textual_complementar`, `sem_correspondencia_texto`. |
| Divergências | Várias escolas partilham o mesmo CNPJ municipal, formato inválido ou chave válida só no lado DMS. |
| Métricas | Contagens específicas + agregação de divergências + (opcional) estatísticas do passe textual restrito. |
| Alta confiança | `merge_confianca == alta_conf_cnpj_exato` **apenas** nos matches unicidade 1⇄1 pela chave. |

Saídas típicas: `outputs/consolidado.xlsx`, `outputs/app.log`.

## Módulos principais

| Ficheiro / pasta | Papel |
|------------------|--------|
| `app.py` | Orquestração Streamlit — Etapa 0, modos, Etapas 2–3. |
| `domain/census_logical.py` | Papéis lógicos Escola/Matrícula (+ UF/município). |
| `services/inferred_mapping.py` | Propostas automáticas INEP/export + **`resolve_census_cnpj_physical_column`** (inclui sufixos ``CNPJ_base_*``). |
| `services/municipality_filter.py` | Filtro antes do merge. |
| `services/census_consolidator.py` | Junção Escola ⊕ Matrícula lógicas. |
| `services/cnpj_merge.py` | Merge igualdade de CNPJ + estados + costura texto opcional. |
| `services/text_fuzzy_merge.py` | RapidFuzz (somente onde é permitido pela Etapa 3). |
| `utils/cnpj.py` | Extrair dígitos, `zfill` 14, checksum. |

### Exercício (ano)

O ano deixa de ficar apenas na lateral: é pedido na **Etapa 0**, reproduzido em ``censo_exercicio`` após consolidar.

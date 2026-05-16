"""
App local Streamlit — cruzamento DMS-Educação × Censo Escolar.

Arquitetura (UX atual):
- **Etapa / Carregamento** — uploads DMS, Censo Escola e opcionalmente Matrícula (qualquer nome de ficheiro).
- **Modo simples** — pré-mapeamento automático pelos aliases INEP/export; Etapa 0 (exercício + UF +
  município) corta Escola antes do merge; menos opções na UI técnica.
- **Modo avançado** — mapeamento lógico explícito (todas as colunas) + opção de desativar filtro municipal.
- **Consolidação** — apenas escolas do município (por defeito) ⊕ Matrícula recortada ao mesmo conjunto de
  ``CO_ENTIDADE`` quando possível; colunas lógicas estáveis no resultado.
- **Etapa 2** — apontar colunas físicas CNPJ nos dois datasets; sempre que há par válido são criadas
  ``__cnpj_norm_dms`` e ``__cnpj_norm_censo``. Antes da Etapa 3 a função ``ensure_normalized_cnpj_workframes`` volta
  a aplicar ``add_normalized_cnpj_column`` usando o consolidado + resoluções automáticas (ex.: ``CNPJ_base_escola``).
- **Etapa 3** — **merge determinístico por CNPJ** como chave; RapidFuzz só opcional/complementar nas linhas da
  DMS **sem** CNPJ utilizável → menos falsos positivos.
"""

from __future__ import annotations

import io
import logging
import traceback
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from domain.census_logical import (
    CENSO_ESCOLA_FIELDS,
    CENSO_MATRICULA_FIELDS,
    LogicalFieldSpec,
)
from domain.dataset_kind import DatasetKind, label as dataset_kind_label
from services.census_consolidator import (
    CensusMergeError,
    consolidate_census_escolar,
    normalize_co_entidade,
)
from services.cnpj_merge import (
    MATCH_CNPJ_EXATO,
    MATCH_MULTIPLAS_ESCOLAS,
    MATCH_TEXTO_COMPLEMENTAR,
    ORDEM_COLUMN,
    SEM_CORRESP_CNPJ,
    SEM_CORRESP_TEXTO,
    SEM_CNPJ_DMS,
    CNPJ_INVALIDO_DMS,
    compute_merge_debug_snapshot,
    deterministic_merge_by_cnpj,
    merge_status_qualifies_textual_complement,
    stitch_complementary_textual_into_base,
)
from services.inferred_mapping import (
    propose_dms_mapping,
    propose_escola_mapping,
    propose_matricula_mapping,
    resolve_census_cnpj_physical_column,
)
from services.municipality_filter import filter_escola_by_municipality, restrict_matricula_to_entidades
from services.table_loader import load_dataset_bundle, spinner_message
from services.text_fuzzy_merge import run_textual_fuzzy_merge
from utils.cnpj import add_normalized_cnpj_column, summarize_cnpj_column
from utils.file_io import FileValidationError

APP_DIR = Path(__file__).resolve().parent
SELECT_SENTINEL = "-- Selecionar coluna --"

UX_SIMPLES = "Simples (recomendado) — município + automático"
UX_AVANCADO = "Avançado — mapeamento manual e diagnósticos"

LOG = logging.getLogger(__name__)


def ensure_normalized_cnpj_workframes(
    *,
    column_map: dict[str, Any],
    df_dms: pd.DataFrame,
    df_censo_consolidado: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    """
    Etapa obrigatória antes do merge determinístico: cria ``__cnpj_norm_dms`` e ``__cnpj_norm_censo`` via
    :func:`add_normalized_cnpj_column`.

    Também corrige cenários onde o consolidado ganhou nomes tipo ``CNPJ_base_escola`` após Escola⊕Matrícula
    (:func:`~services.inferred_mapping.resolve_census_cnpj_physical_column`).
    """

    cm_out: dict[str, Any] = dict(column_map or {})
    d_cols = {str(c) for c in df_dms.columns}

    dms_pick = cm_out.get("dms_cnpj")
    if (
        not isinstance(dms_pick, str)
        or dms_pick == SELECT_SENTINEL
        or not dms_pick.strip()
        or dms_pick not in d_cols
    ):
        inferred_d = propose_dms_mapping([str(c) for c in df_dms.columns]).get("CNPJ")
        if inferred_d and inferred_d in df_dms.columns:
            dms_pick = inferred_d
            cm_out["dms_cnpj"] = inferred_d
            LOG.info(
                "CNPJ DMS inferido automaticamente para normalização Etapa 3: `%s`.",
                inferred_d,
            )
        else:
            st.error(
                "**DMS**: não conseguimos localizar uma coluna de CNPJ física válida neste arquivo. Na Etapa 2 "
                "escolha manualmente a coluna do contribuinte."
            )
            return None

    censo_pick = cm_out.get("censo_cnpj")
    if (
        not isinstance(censo_pick, str)
        or censo_pick == SELECT_SENTINEL
        or not censo_pick.strip()
        or censo_pick not in df_censo_consolidado.columns
    ):
        resolved_c = resolve_census_cnpj_physical_column(df_censo_consolidado.columns)
        if resolved_c:
            censo_pick = resolved_c
            cm_out["censo_cnpj"] = resolved_c
            LOG.info(
                "Coluna física CNPJ censo inferida antes do merge: `%s` (fallback sufixo merge possível).",
                resolved_c,
            )
        else:
            st.error(
                "**Censo municipal sem coluna CNPJ reconhecida.** Inclua o campo lógico **`CNPJ`** no mapeamento "
                "da Escola (INEP) ou verifique se o consolidado expõe ``CNPJ_base_escola`` / ``CNPJ_base_matricula``."
            )
            return None

    dms_work = add_normalized_cnpj_column(df_dms, dms_pick, "__cnpj_norm_dms")
    censo_work = add_normalized_cnpj_column(df_censo_consolidado, censo_pick, "__cnpj_norm_censo")

    if "__cnpj_norm_dms" not in dms_work.columns or "__cnpj_norm_censo" not in censo_work.columns:
        st.error("Falha ao materializar colunas internas ``__cnpj_norm_*`` — contacte a equipa técnica.")
        return None

    norm_censo = censo_work["__cnpj_norm_censo"].astype(str).str.strip()
    if not (norm_censo.str.len() > 0).any():
        st.warning(
            f"A coluna **`{censo_pick}`** existe, mas **não há CNPJ normalizável** (14 dígitos) nas linhas — "
            "confirme o mapeamento ou se a base municipal realmente contém PJ com número de inscrição."
        )

    return dms_work, censo_work, cm_out


def configure_logging() -> None:
    """Logs para ficheiro (DEBUG) e consola (INFO). Evita handlers duplicados em reruns Streamlit."""

    root = logging.getLogger()
    if root.handlers:
        return

    log_dir = APP_DIR / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)


def _friendly_file_error(where: str, err: BaseException) -> None:
    st.error(f"**{where}**: não foi possível utilizar este ficheiro.")
    if isinstance(err, FileValidationError):
        st.warning(str(err))
        LOG.warning("Validação falhou [%s]: %s", where, err)
        return
    st.warning(
        "Ocorreu um erro inesperado. Sugestões:\n\n"
        "- Confirme se o formato é CSV ou Excel (.xlsx) íntegro.\n\n"
        "- Para CSV grave em UTF‑8 ou Windows‑1252.\n\n"
        "- Experimente remover palavras‑passe de folha Excel."
    )
    LOG.exception("Erro ao processar [%s]", where)
    with st.expander("Detalhe técnico (equipa técnica)"):
        st.code(str(err))


def _preview_dataframe(title: str, df: pd.DataFrame, caption: str) -> None:
    st.markdown(f"**{title}** — {caption}")
    st.dataframe(df, use_container_width=True, height=320)


def column_options(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return [SELECT_SENTINEL]
    return [SELECT_SENTINEL] + [str(c) for c in df.columns]


def collect_logical_mapping(prefix_key: str, specs: tuple[LogicalFieldSpec, ...]) -> dict[str, str]:
    """Lê ``st.session_state`` preenchido pelos ``selectbox`` de mapeamento lógico."""

    result: dict[str, str] = {}
    for spec in specs:
        val = st.session_state.get(f"{prefix_key}_{spec.key}", SELECT_SENTINEL)
        if val != SELECT_SENTINEL:
            result[spec.key] = val
    return result


def resolve_escola_mapping(cols_escola: list[str], *, ui_simples: bool) -> dict[str, str]:
    """Junta inferência automática com overrides do modo avançado (widgets já renderizados)."""

    proposals = propose_escola_mapping(list(cols_escola))
    if ui_simples:
        return proposals

    merged: dict[str, str] = dict(proposals)
    widgets = collect_logical_mapping("logical_escola", CENSO_ESCOLA_FIELDS)
    for logical_key, physical in widgets.items():
        if physical and physical != SELECT_SENTINEL and physical in cols_escola:
            merged[logical_key] = physical
    return merged


def resolve_matricula_mapping(cols_mat: list[str] | None, *, ui_simples: bool) -> dict[str, str]:
    if not cols_mat:
        return {}
    proposals = propose_matricula_mapping(list(cols_mat))
    if ui_simples:
        return proposals
    merged: dict[str, str] = dict(proposals)
    widgets = collect_logical_mapping("logical_matricula", CENSO_MATRICULA_FIELDS)
    for logical_key, physical in widgets.items():
        if physical and physical != SELECT_SENTINEL and physical in cols_mat:
            merged[logical_key] = physical
    return merged


def _sanitize_municipio_codigo_cell(value: object) -> str:
    texto = "" if pd.isna(value) else str(value).strip()
    if texto.endswith(".0") and texto.replace(".0", "").isdigit():
        texto = texto[:-2]
    return texto


def render_etapa0_contexto_municipal(
    df_escola: pd.DataFrame,
    map_geo: dict[str, str],
    *,
    ui_simples: bool,
) -> dict[str, Any]:
    """
    Etapa 0 — exercício + UF + município. ``map_geo`` deve refletir o mapeamento **resolvido** actual
    (automático ou manual) para ``SG_UF`` / ``CO_MUNICIPIO`` / ``NO_MUNICIPIO``.
    """

    st.divider()
    st.header("Etapa 0 — Contexto municipal")
    if ui_simples:
        st.caption(
            "Indique o ano de referência e o **município**. Só ficam escolas dessa cidade no Censo "
            "**antes** da junção — processamento mais leve."
        )
    else:
        st.caption(
            "Igual ao modo simples, com opções extra: pode **desligar** o recorte territorial se o ficheiro "
            "já estiver pré-filtrado ou não usar colunas típicas INEP."
        )

    exercise_default = int(st.session_state.get("ctx_exercise_default", 2025))
    ex = st.number_input(
        "Exercício do Censo (ano de referência)",
        min_value=1996,
        max_value=2050,
        value=exercise_default,
        step=1,
        key="ctx_exercise_year",
        help='Aparece em ``censo_exercicio`` na base consolidada.',
    )
    st.session_state["ctx_exercise_default"] = int(ex)

    uf_phys = map_geo.get("SG_UF")
    co_phys = map_geo.get("CO_MUNICIPIO")
    no_phys = map_geo.get("NO_MUNICIPIO")

    skip_geo = False
    if not ui_simples:
        skip_geo = st.checkbox(
            "Não aplicar filtro municipal (usar todas as linhas da Escola carregada)",
            value=bool(st.session_state.get("ctx_skip_municipality", False)),
            key="ctx_skip_municipality",
        )

    out: dict[str, Any] = {
        "exercise": int(ex),
        "uf": None,
        "mun_code": None,
        "mun_label": "",
        "skip_geo": bool(skip_geo),
        "filtro_ativo": False,
        "filtro_impossivel_geo": False,
    }

    if skip_geo:
        st.info(
            "**Filtro municipal desativado.** Será usado o conjunto inteiro da tabela Escola em memória — "
            "único cenário válido quando o arquivo já está recortado ou não há UF/município."
        )
        return out

    if uf_phys is None or co_phys is None:
        out["filtro_impossivel_geo"] = True
        st.warning(
            "As colunas físicas típicas de **UF / município** não foram encontradas nem mapeadas.\n\n"
            "- Modo simples: confirme se o ficheiro é o microdados **Escola** INEP (com ``SG_UF`` / ``CO_MUNICIPIO``).\n"
            "- Modo avançado: associe explicitamente esses dois campos lógicos no mapeamento **ou** desative o filtro acima."
        )
        return out

    if uf_phys not in df_escola.columns or co_phys not in df_escola.columns:
        out["filtro_impossivel_geo"] = True
        st.error("As colunas de localização definidas pelo mapeamento **não existem** nesta tabela Escola.")
        return out

    ufs = df_escola[uf_phys].dropna().astype(str).str.strip().str.upper().unique()
    ufs_sorted = sorted(u for u in ufs if u)
    if not ufs_sorted:
        st.warning("Sem valores de UF na coluna física configurada.")
        out["filtro_impossivel_geo"] = True
        return out

    uf_sel = st.selectbox("UF", ufs_sorted, key="ctx_uf_select")
    out["uf"] = uf_sel

    sub_mask = df_escola[uf_phys].astype(str).str.strip().str.upper() == str(uf_sel).strip().upper()
    sub = df_escola.loc[sub_mask]
    use_name = bool(no_phys and no_phys in df_escola.columns)

    labels: list[str] = []
    label_to_code: dict[str, str] = {}
    for _, row in sub[[co_phys] + ([no_phys] if use_name else [])].drop_duplicates().iterrows():
        code_raw = row[co_phys]
        code = _sanitize_municipio_codigo_cell(code_raw)
        if use_name:
            nome = str(row[no_phys]).strip() if not pd.isna(row[no_phys]) else ""
            label = f"{nome} — código IBGE {code}" if nome else f"Município código IBGE {code}"
        else:
            label = f"Município código IBGE {code}"
        if label not in label_to_code:
            labels.append(label)
            label_to_code[label] = code

    labels = sorted(labels, key=lambda s: (label_to_code.get(s, ""), s))
    if not labels:
        st.warning("Nenhum código de município encontrado para a UF seleccionada neste ficheiro.")
        out["filtro_impossivel_geo"] = True
        return out

    chosen = st.selectbox("Município", labels, key="ctx_municipio_select")
    out["mun_code"] = label_to_code.get(chosen)
    out["mun_label"] = chosen
    out["filtro_ativo"] = True
    st.success(
        f"Vamos **filtrar o Censo** para apenas escolas em **{chosen.split(' — ')[0].strip()}** "
        f"({uf_sel}) antes de juntar com Matrícula."
    )
    return out


def append_censo_context_columns(df: pd.DataFrame, ctx: dict[str, Any]) -> pd.DataFrame:
    """Replica metadados de contexto em todas as linhas."""

    out = df.copy()
    if ctx.get("uf"):
        out["censo_ctx_UF"] = str(ctx["uf"])
    if ctx.get("mun_code") is not None:
        out["censo_ctx_municipio_codigo"] = str(ctx["mun_code"])
    if ctx.get("mun_label"):
        out["censo_ctx_municipio_rotulo_ui"] = str(ctx["mun_label"])
    filt = "sim" if ctx.get("filtro_ativo") else "nao"
    out["censo_ctx_filtro_municipal_aplicado"] = filt
    return out


def render_logical_mapper(
    titulo: str,
    df: pd.DataFrame,
    specs: tuple[LogicalFieldSpec, ...],
    prefix_key: str,
) -> None:
    st.markdown(f"#### {titulo}")
    opts = column_options(df)
    for spec in specs:
        obrig = "**obrigatório**" if getattr(spec, "obrigatorio_escola", False) or getattr(
            spec, "obrigatorio_matricula", False
        ) else "opcional"
        st.selectbox(
            f"`{spec.key}` ({obrig}) — {spec.description_pt}",
            opts,
            key=f"{prefix_key}_{spec.key}",
        )


def render_cnpj_stats_block(label: str, df: pd.DataFrame, col_name: str) -> None:
    """Mostra contagem vazios / inválidos / válidos (com dígitos verificadores)."""

    st.markdown(label)
    if col_name == SELECT_SENTINEL or col_name not in df.columns:
        st.info("Selecione primeiro a coluna de CNPJ.")
        return
    stats = summarize_cnpj_column(df[col_name])
    c_empty, c_inv, c_ok = st.columns(3)
    c_empty.metric("Vazio", f"{stats.empty:,}")
    c_inv.metric("Inválido formato/DV", f"{stats.invalid_format_or_checksum:,}")
    c_ok.metric("Válidos (DV ok)", f"{stats.valid_checksum:,}")
    pct = (
        round(100.0 * stats.valid_checksum / stats.total, 1)
        if stats.total
        else 0.0
    )
    st.caption(
        f"Total de registos avaliados: **{stats.total:,}**. "
        f"Percentagem com CNPJ aceite: **{pct} %**."
    )
    LOG.info(
        "CNPJ %s [%s]: vazio=%s inválido=%s válido=%s",
        label,
        col_name,
        stats.empty,
        stats.invalid_format_or_checksum,
        stats.valid_checksum,
    )


def run_etapa2_mapping(
    dms_df: pd.DataFrame,
    censo_df: pd.DataFrame,
    *,
    ui_simples: bool,
    up_dms_name: str,
    up_escola_name: str,
) -> None:
    """Seleção compacta das colunas de ligação DMS ↔ Censo + normalização de CNPJ."""

    st.divider()
    st.header("Etapa 2 — Ligação fiscal com o Censo municipal")
    if ui_simples:
        st.markdown(
            "O passo anterior já **nomeou logicamente** o CNPJ da escola, o nome público (`NO_ENTIDADE`) e as "
            "matrículas sempre que foram reconhecidos no ficheiro INEP ou no mapeamento avançado. "
            "Aqui concentramo-nos apenas em **alinhar os campos ao export da DMS**."
        )
    else:
        st.caption(
            "As colunas do Censo consolidado mantêm nomes estáveis (`CNPJ`, `NO_ENTIDADE`, `matriculas`, … — "
            "podem estar ausentes quando não mapeadas). Os CNPJs são normalizados e validados (DV)."
        )

    sig_uid = "|".join(
        (
            up_dms_name,
            up_escola_name,
            str(len(dms_df.index)),
            str(len(censo_df.index)),
            ",".join(map(str, list(censo_df.columns[:16]))),
        )
    )
    if st.session_state.get("_etapa2_payload_sig") != sig_uid:
        st.session_state["_etapa2_payload_sig"] = sig_uid
        for k in ("map_dms_cnpj", "map_dms_razao", "map_dms_qtd", "map_censo_cnpj", "map_censo_nome", "map_censo_mat"):
            st.session_state.pop(k, None)
        dm = propose_dms_mapping([str(c) for c in dms_df.columns])
        cols_d = set(map(str, dms_df.columns))
        cols_c = set(map(str, censo_df.columns))
        prop_d_cnpj = dm.get("CNPJ")
        if isinstance(prop_d_cnpj, str) and prop_d_cnpj in cols_d:
            st.session_state["map_dms_cnpj"] = prop_d_cnpj
        prop_r = dm.get("razao_social")
        if isinstance(prop_r, str) and prop_r in cols_d:
            st.session_state["map_dms_razao"] = prop_r
        prop_q = dm.get("quantidade")
        if isinstance(prop_q, str) and prop_q in cols_d:
            st.session_state["map_dms_qtd"] = prop_q
        hit_cnpj_cons = resolve_census_cnpj_physical_column(list(cols_c))
        if hit_cnpj_cons:
            st.session_state["map_censo_cnpj"] = hit_cnpj_cons
        if "NO_ENTIDADE" in cols_c:
            st.session_state["map_censo_nome"] = "NO_ENTIDADE"
        if "matriculas" in cols_c:
            st.session_state["map_censo_mat"] = "matriculas"

    opts_d = column_options(dms_df)
    opts_c = column_options(censo_df)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("DMS-Educação")
        dms_cnpj = st.selectbox(
            "Coluna que contém o CNPJ jurídico",
            opts_d,
            key="map_dms_cnpj",
            help='Procuramos automaticamente por aliases comuns (``CNPJ``, ``NU_CNPJ``, …).',
        )
        dms_razao = st.selectbox(
            "Texto institucional (razão ou nome próximo)",
            opts_d,
            key="map_dms_razao",
            help='Usado **mais tarde** no matching textual quando o fiscal quiser conferir também por texto.',
        )
        dms_qtd = st.selectbox(
            "Quantidade relacionada ao serviço/alunos (opcional)",
            opts_d,
            key="map_dms_qtd",
        )

    with c2:
        st.subheader("Censo (já com colunas lógicas quando possível)")
        censo_cnpj = st.selectbox(
            "CNPJ institucional do Censo",
            opts_c,
            key="map_censo_cnpj",
        )
        censo_nome = st.selectbox(
            "Denominação pública reconhecível",
            opts_c,
            key="map_censo_nome",
        )
        censo_mat = st.selectbox(
            "Campo agregado de matrículas (opcional)",
            opts_c,
            key="map_censo_mat",
        )

    diag_exp = st.expander("Diagnóstico de CNPJ (modo avançado)", expanded=not ui_simples)
    with diag_exp:
        left, right = st.columns(2)
        with left:
            render_cnpj_stats_block("DMS", dms_df, dms_cnpj)
        with right:
            render_cnpj_stats_block("Censo municipal", censo_df, censo_cnpj)

    if ui_simples:
        with st.expander("Ajustar campos manualmente (se o ficheiro DMS tiver nomes atípicos)"):
            st.caption(
                "Só precisa de alterar alguma coisa se a deteção automática não bater com o layout real—por exemplo "
                "colunas internas com prefixos do sistema contabilístico."
            )

    if (
        dms_cnpj != SELECT_SENTINEL
        and censo_cnpj != SELECT_SENTINEL
        and dms_cnpj in dms_df.columns
        and censo_cnpj in censo_df.columns
    ):
        dms_work = add_normalized_cnpj_column(dms_df, dms_cnpj, "__cnpj_norm_dms")
        censo_work = add_normalized_cnpj_column(censo_df, censo_cnpj, "__cnpj_norm_censo")
        st.session_state["dms_work"] = dms_work
        st.session_state["censo_work"] = censo_work
        st.session_state["column_map"] = {
            "dms_cnpj": dms_cnpj,
            "dms_razao": dms_razao,
            "dms_qtd": dms_qtd,
            "censo_cnpj": censo_cnpj,
            "censo_nome": censo_nome,
            "censo_mat": censo_mat,
        }
        if ui_simples:
            st.success(
                "**Etapa 2 OK.** CNPJ guardado em ``__cnpj_norm_*`` — na Etapa 3 o **merge vai primeiro por esse "
                "número (14 dígitos)** e o texto (RapidFuzz) aparece apenas se a linha não tiver CNPJ DMS válido."
            )
        else:
            sample = dms_work[[dms_cnpj, "__cnpj_norm_dms"]].head(8)
            with st.expander("Pré-visualização de CNPJ normalizado (DMS — primeiras linhas)"):
                st.dataframe(sample, use_container_width=True, hide_index=True)
        LOG.info("Mapa da Etapa 2 persistido em session_state.")
    else:
        for key in ("dms_work", "censo_work", "column_map"):
            st.session_state.pop(key, None)
        if ui_simples:
            st.info("Escolha o **CNPJ** na base DMS e no Censo — é o único par obrigatório para continuar.")
        else:
            st.info(
                "Seleccione o par **CNPJ** nas duas bases para normalizar dígitos (``__cnpj_norm_*``) — é o primeiro "
                "passo obrigatório antes do merge determinístico da Etapa 3."
            )


def default_select_index(options: list[str], preferred: str | None) -> int:
    """Índice inicial do ``selectbox`` quando o nome preferido existe nas opções."""

    if preferred and preferred in options:
        return options.index(preferred)
    return 0


def _series_nonempty_cell_count(series: pd.Series) -> int:
    """Células com conteúdo utilizável (não NaN, não vazio após strip, não placeholder textual)."""

    mask = series.notna()
    strv = series.astype(str).str.strip()
    mask &= strv.ne("") & ~strv.str.lower().isin(["nan", "none"])
    return int(mask.sum())


def render_etapa3_premerge_diagnostics(
    dms_work: pd.DataFrame,
    censo_work: pd.DataFrame,
    cm: dict[str, Any],
) -> None:
    """Inspeção explícita das bases de trabalho antes do merge determinístico."""

    with st.expander(
        "**Diagnóstico pré-merge** — dados entregues à Etapa 3",
        expanded=True,
    ):
        st.markdown("##### Colunas em `dms_work`")
        st.code("\n".join(map(str, dms_work.columns)), language="text")
        st.caption(
            f"**{len(dms_work.columns)}** colunas · **{len(dms_work.index):,}** linhas · "
            f"`__cnpj_norm_dms` presente: **{'sim' if '__cnpj_norm_dms' in dms_work.columns else 'não'}**"
        )

        st.markdown("##### Colunas em `censo_work`")
        st.code("\n".join(map(str, censo_work.columns)), language="text")
        st.caption(
            f"**{len(censo_work.columns)}** colunas · **{len(censo_work.index):,}** linhas · "
            f"`__cnpj_norm_censo` presente: **{'sim' if '__cnpj_norm_censo' in censo_work.columns else 'não'}**"
        )

        col_censo_fis = cm.get("censo_cnpj")
        st.markdown("##### CNPJ do Censo — coluna física e preenchimento")
        if (
            isinstance(col_censo_fis, str)
            and col_censo_fis.strip()
            and col_censo_fis != SELECT_SENTINEL
        ):
            st.write(f"**`column_map.censo_cnpj`:** `{col_censo_fis}`")
            if col_censo_fis in censo_work.columns:
                raw = censo_work[col_censo_fis]
                n_nonempty = _series_nonempty_cell_count(raw)
                st.metric(
                    "Valores não vazios (célula com texto)",
                    f"{n_nonempty:,} / {len(raw):,}",
                )
            else:
                st.warning(
                    f"A coluna **`{col_censo_fis}`** está no mapeamento mas **não existe** em `censo_work`. "
                    "Causas prováveis: consolidado desatualizado após alterar o Censo ou os mapeamentos INEP; "
                    "renomeação de colunas na consolidação; ou ficheiro de escolas diferente. "
                    "**Reconsolide** o Censo no passo anterior ou confira os nomes listados acima."
                )
        else:
            st.info(
                "O CNPJ físico do Censo **não está definido** em `column_map` (Etapa 2). "
                "Sem isso, o fluxo não fixa a coluna-fonte para normalização."
            )

        st.markdown("##### Coluna interna `__cnpj_norm_censo`")
        if "__cnpj_norm_censo" in censo_work.columns:
            s_norm = censo_work["__cnpj_norm_censo"]
            n_filled = int(
                (s_norm.fillna("").astype(str).str.strip().ne("")).sum()
            )
            st.success("A coluna **`__cnpj_norm_censo`** existe neste `DataFrame`.")
            st.write(f"- **dtype:** `{s_norm.dtype}`")
            st.metric(
                "Linhas com normalização não vazia (após extrair dígitos)",
                f"{n_filled:,} / {len(censo_work.index):,}",
            )
            if (
                isinstance(col_censo_fis, str)
                and col_censo_fis.strip()
                and col_censo_fis != SELECT_SENTINEL
                and col_censo_fis in censo_work.columns
            ):
                preview = censo_work[
                    [col_censo_fis, "__cnpj_norm_censo"]
                ].head(25)
                st.markdown("**Pré-visualização:** CNPJ original × `__cnpj_norm_censo` (até 25 linhas)")
                st.dataframe(preview, use_container_width=True, hide_index=True)
            else:
                st.caption(
                    "Pré-visualização lado a lado indisponível: coluna física do Censo em falta ou inválida no `column_map`."
                )
        else:
            st.warning(
                "**`__cnpj_norm_censo` não existe** neste `DataFrame`. Causas prováveis:\n\n"
                "- A Etapa 2 não ficou com um **par CNPJ válido** (DMS + Censo) ou os `selectbox` ainda estão no sentinel.\n"
                "- O passo `ensure_normalized_cnpj_workframes` não correu **depois** da Etapa 2 *ou* falhou ao resolver a "
                "coluna física no consolidado (ex.: `CNPJ` vs `CNPJ_base_escola`).\n"
                "- O consolidado foi **alterado ou invalidado** (ficheiros, município, exercício) sem reexecutar o encadeamento.\n\n"
                "Corrija na **Etapa 2**, **consolide** de novo o Censo se mudou o contexto, e recarregue a DMS se necessário."
            )


def run_etapa3_merge_pipeline(
    dms_work: pd.DataFrame,
    censo_work: pd.DataFrame,
    *,
    ux_simples: bool = False,
) -> None:
    """Pipeline Etapa 3: merge determinístico por CNPJ + texto opcional apenas sem CNPJ na DMS."""

    st.divider()
    st.header("Etapa 3 — Cruzamento DMS × Censo (**CNPJ primeiro, confiança alta**)")
    st.markdown(
        "1. **Chave CNPJ determinística** (`__cnpj_norm_*`): apenas dígitos, **14 dígitos** (`zfill` quando "
        "há menos de 14 dígitos — ver `utils.cnpj`).\n"
        "2. **Classificação** por linha DMS antes de texto: correspondência única (`match_cnpj_exato`), "
        "várias escolas no Censo com o mesmo número (`multiplas_escolas_mesmo_cnpj`), falta na base municipal "
        "(`sem_correspondencia_cnpj`), campo vazio/inutilizável na DMS ou CNPJ DMS invalidado por regras DV.\n"
        "3. **RapidFuzz** aparece apenas como passe **complementar** nas linhas em que **não há CNPJ "
        "utilizável na DMS** — nunca sobrepõe resultados já definidos pela chave fiscal."
    )
    if ux_simples:
        st.caption("No modo simples o fluxo sugere apenas o merge por CNPJ; o texto fica dentro do expander abaixo.")
    else:
        with st.expander("Arquitetura determinística (para equipa técnica)"):
            st.markdown(
                "- **Join lógico** — esquerda sempre a fatia inteira da DMS; lado direito primeira escola encontrada "
                "por CNPJ no Censo municipal (uso de `lookup_first.groupby(...).head(1)`), com contagem "
                "`cnpj_censo_candidatos_mesmo_numero` sempre visível quando existem várias hipóteses.\n"
                "- **Alto grau de confiança (`merge_confianca = alta_conf_cnpj_exato`)** — exclusivo onde existe "
                "unicidade bilateral da chave 14 dígitos.\n"
                "- **`merge_metodo_primario`** — inicialmente `cnpj_14_digitos` apenas quando faz sentido usar a "
                "chave fiscal; ficará vazio no ramo texto-only até opcionalmente preencher com `texto_complementar`."
            )

    cm = st.session_state.get("column_map") or {}
    render_etapa3_premerge_diagnostics(dms_work, censo_work, cm)

    col_dms_raw = cm.get("dms_cnpj")

    opts_dms = column_options(dms_work)
    opts_censo = column_options(censo_work)

    sig_det_now = (
        str(col_dms_raw),
        str(cm.get("censo_cnpj")),
        len(dms_work.index),
        len(censo_work.index),
        "__cnpj_norm_dms" in dms_work.columns,
        "__cnpj_norm_censo" in censo_work.columns,
    )

    merge_bloqueado = False
    if not col_dms_raw or col_dms_raw == SELECT_SENTINEL:
        st.error("Finalize a Etapa 2 definindo explicitamente as colunas de CNPJ DMS.")
        merge_bloqueado = True
    elif "__cnpj_norm_dms" not in dms_work.columns:
        st.error(
            "Falta ``__cnpj_norm_dms`` na DMS transformada — a normalização deveria aplicar‑se assim que as colunas "
            "CNPJ forem válidas. Recarregue os ficheiros ou limpe cache e gere novamente a Etapa 2."
        )
        merge_bloqueado = True
    elif "__cnpj_norm_censo" not in censo_work.columns:
        st.error(
            "**Merge indisponível:** falta ``__cnpj_norm_censo`` na base municipal de trabalho. "
            "Consulte o **diagnóstico pré-merge** acima para causas prováveis e passos de correção."
        )
        merge_bloqueado = True

    det_clicked = False
    if not merge_bloqueado:
        det_clicked = st.button(
            "Executar merge determinístico (CNPJ 14 dígitos)",
            type="primary",
            key="btn_etapa3_merge_cnpj",
        )

    if det_clicked:
        prog = st.progress(0)

        def _cb_det(progress: float) -> None:
            prog.progress(min(max(progress, 0.0), 1.0))

        merge_snap = compute_merge_debug_snapshot(
            dms_work,
            censo_work,
            col_dms_norm="__cnpj_norm_dms",
            col_censo_norm="__cnpj_norm_censo",
        )
        LOG.info("Etapa 3 — snapshot antes do merge determinístico: %s", merge_snap)

        try:
            consolidado_cnpj, summary_cnpj = deterministic_merge_by_cnpj(
                dms_work,
                censo_work,
                col_dms_raw_cnpj=str(col_dms_raw),
                col_dms_norm="__cnpj_norm_dms",
                col_censo_norm="__cnpj_norm_censo",
                progress_callback=_cb_det,
            )
        except Exception as exc:  # pylint: disable=broad-except
            prog.empty()
            tb_full = traceback.format_exc()
            st.error(
                "**Erro durante o merge determinístico por CNPJ.** "
                "Use o expander abaixo para o traceback completo e o snapshot das bases."
            )
            LOG.exception("Etapa 3 — deterministic merge (traceback completo no campo 'stacktrace_completo')")

            with st.expander("**Debug — exceção e traceback completos**", expanded=True):
                st.markdown(f"**Tipo da exceção:** `{type(exc).__name__}`")
                st.markdown(f"**Mensagem:** `{exc}`")
                cause = getattr(exc, "__cause__", None)
                if cause is not None:
                    st.markdown(
                        f"**Encadeamento (`__cause__`):** `{type(cause).__name__}` — `{cause}`"
                    )
                st.markdown("**Stacktrace (`traceback.format_exc()`):**")
                st.code(tb_full, language="text")

            with st.expander("**Diagnóstico numérico no momento do merge**", expanded=True):
                c1, c2 = st.columns(2)
                c1.markdown("**`dms_work`**")
                c1.metric("shape (linhas × cols)", str(merge_snap.get("dms_shape")))
                c2.markdown("**`censo_work`**")
                c2.metric("shape (linhas × cols)", str(merge_snap.get("censo_shape")))
                st.markdown("**CNPJs únicos (14 dígitos na DMS; não vazio no Censo)** e **chaves com mais do que uma linha** no mesmo lado:")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("DMS — únicos 14d", str(merge_snap.get("dms_cnpj_unicos_14_digitos", "—")))
                m2.metric("DMS — chaves com duplicidade", str(merge_snap.get("dms_chaves_com_mais_de_uma_linha", "—")))
                m3.metric("Censo — únicos (chave ≠ '')", str(merge_snap.get("censo_cnpj_unicos_nao_vazio", "—")))
                m4.metric("Censo — chaves multi-linha", str(merge_snap.get("censo_chaves_com_mais_de_uma_linha", "—")))
                st.json(merge_snap)
            return

        prog.empty()
        st.session_state["consolidado_df"] = consolidado_cnpj
        st.session_state.pop("consolidado_summary", None)
        st.session_state["etapa3_cnpj_summary"] = summary_cnpj
        st.session_state["etapa3_det_sig"] = sig_det_now
        st.session_state.pop("etapa3_comp_text_summary", None)
        st.session_state.pop("etapa3_fuzzy_sig", None)

        out_path = APP_DIR / "outputs" / "consolidado.xlsx"
        try:
            consolidado_cnpj.to_excel(out_path, index=False, engine="openpyxl")
            LOG.info(
                "consolidado.xlsx atualizado pelo merge determinístico (linhas=%s)",
                len(consolidado_cnpj.index),
            )
            st.success(f"**Merge por CNPJ concluído.** Ficheiro em `{out_path}`.")
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Escrita opcional falhou (pode sempre descarregar): {exc}")
            LOG.exception("Falha consolidado deterministic write")

    if merge_bloqueado:
        st.info(
            "O merge determinístico **não está disponível** até existirem as colunas internas "
            "``__cnpj_norm_dms`` e ``__cnpj_norm_censo`` nos `DataFrame` de trabalho. "
            "Corrija conforme o **diagnóstico pré-merge** e a Etapa 2."
        )
        return

    sdet = st.session_state.get("etapa3_det_sig")

    consolidado_raw = st.session_state.get("consolidado_df")
    summary_det = st.session_state.get("etapa3_cnpj_summary")

    if consolidado_raw is None or summary_det is None:
        st.info(
            "Clique **Executar merge determinístico (CNPJ 14 dígitos)** para produzir o consolidado por chave fiscal "
            "e só depois aparecem métricas, pré-visualização e o texto complementar."
        )
        return

    if isinstance(sdet, tuple) and sdet != sig_det_now:
        st.warning(
            "**Colunas CNPJ ou tamanhos das bases mudaram** face ao último merge determinístico. "
            "Volte a executar **merge por CNPJ (14 dígitos)** para manter dados consistentes."
        )

    texto_elegivel_n = 0
    if "match_status_principal" in consolidado_raw.columns:
        estado_status = consolidado_raw["match_status_principal"]
        mascara_texto_opt = estado_status.astype(str).apply(
            lambda s: merge_status_qualifies_textual_complement(s),
        )
        mascara_texto_opt = mascara_texto_opt.fillna(False)
        texto_elegivel_n = int(mascara_texto_opt.sum())

    with st.expander(
        "**Texto opcional (RapidFuzz)** — apenas linhas sem CNPJ utilizável na DMS",
        expanded=not ux_simples,
    ):
        st.caption(
            "**Complementar.** Todas as outras categorias ficam definidas apenas pela **chave fiscal CNPJ** "
            f"(elimina falsos positivos texto). `{texto_elegivel_n}` linha(s) atualmente elegíveis."
        )

        col_razao = st.selectbox(
            "Texto institucional na DMS (razão / nome público próximo)",
            opts_dms,
            index=default_select_index(opts_dms, cm.get("dms_razao")),
            key="etapa3_col_dms_razao",
        )
        col_nome = st.selectbox(
            "Denominação pública escola ou entidade (Censo consolidado municipal)",
            opts_censo,
            index=default_select_index(opts_censo, cm.get("censo_nome")),
            key="etapa3_col_censo_nome",
        )
        cutoff = st.radio(
            "Pontuação mínima só no passe texto (WRatio RapidFuzz, 0–100)",
            options=[70, 80, 90],
            index=1,
            horizontal=True,
            key="etapa3_cutoff",
        )

        texto_pronto = (
            col_razao != SELECT_SENTINEL and col_nome != SELECT_SENTINEL and texto_elegivel_n > 0
        )
        if not texto_pronto and texto_elegivel_n == 0:
            st.info("Nesta execução **não há** linhas só com fallback textual.")
        elif not texto_pronto:
            st.info("Seleccione as duas colunas de texto válidas antes de executar RapidFuzz.")

        if st.button(
            "Correr passe textual só nas linhas sem CNPJ válido",
            disabled=not texto_pronto,
            key="btn_etapa3_optional_text_run",
            type="secondary",
        ):
            mascara = consolidado_raw["match_status_principal"].astype(str).apply(
                lambda s: merge_status_qualifies_textual_complement(s),
            ).fillna(False)
            if not bool(mascara.any()):
                st.warning("Nenhuma linha textual elegível após filtros deterministicos mais recentes.")
            else:
                subset_dms = dms_work.loc[mascara.to_numpy()].copy().reset_index(drop=True)
                orden_col = pd.to_numeric(
                    consolidado_raw.loc[mascara, f"dms__{ORDEM_COLUMN}"],
                    errors="coerce",
                )
                orden_col_series = orden_col.reset_index(drop=True)
                subset_dms.loc[:, ORDEM_COLUMN] = orden_col_series.to_numpy(dtype=int)

                fz_prog = st.progress(0)

                def _cb_fuzz(progress: float) -> None:
                    fz_prog.progress(min(max(progress, 0.0), 1.0))

                try:
                    fz_resultado, _fz_summary = run_textual_fuzzy_merge(
                        subset_dms,
                        censo_work,
                        col_dms_razao=col_razao,
                        col_censo_nome=col_nome,
                        score_cutoff=float(cutoff),
                        progress_callback=_cb_fuzz,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    fz_prog.empty()
                    st.error("Falha durante RapidFuzz complementar.")
                    LOG.exception("Etapa 3 — fuzzy opcional")
                    with st.expander("Detalhe técnico"):
                        st.code(str(exc))
                else:
                    fz_prog.empty()
                    consolidado_atualizado, texto_sumario = stitch_complementary_textual_into_base(
                        consolidado_raw,
                        fz_resultado,
                        score_cutoff_used=float(cutoff),
                    )
                    st.session_state["consolidado_df"] = consolidado_atualizado
                    st.session_state["etapa3_comp_text_summary"] = texto_sumario
                    st.session_state["etapa3_fuzzy_sig"] = (col_razao, col_nome, int(cutoff))

                    try:
                        (APP_DIR / "outputs").mkdir(parents=True, exist_ok=True)
                        atual_path = APP_DIR / "outputs" / "consolidado.xlsx"
                        consolidado_atualizado.to_excel(atual_path, index=False, engine="openpyxl")
                        LOG.info(
                            "consolidado.xlsx atualizado texto opcional (%s matches textual).",
                            texto_sumario.matches_texto,
                        )
                        st.success(
                            "**Passe texto aplicado** só onde faltava CNPJ DMS válido · "
                            f"{texto_sumario.matches_texto} match texto / {texto_sumario.linhas_elegiveis} elegíveis."
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        st.warning(f"Gravação após texto falhou ({exc}) — pode descarregar manualmente.")
                        LOG.exception("Falhou escrita texto complement")

    refinado = st.session_state.get("consolidado_df")
    if not isinstance(refinado, pd.DataFrame):
        refinado = consolidado_raw
    summary_actual = summary_det

    diver_agregado_linhas_dms = int(
        getattr(summary_actual, "multiplas_escolas_mesmo_cnpj", 0)
        + getattr(summary_actual, "cnpj_dms_invalido", 0)
        + getattr(summary_actual, "sem_correspondencia_cnpj", 0)
    )

    st.subheader("Métricas — resultado determinístico + divergências resumidas")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Merge exato alta confiança (CNPJ)", f"{summary_actual.match_cnpj_exato:,}")
    r2.metric("Linhas sem CNPJ DMS válido (= texto opcional futuro)", f"{texto_elegivel_n:,}")
    r3.metric("Divergências agregadas (multi + inválidos + falta censo)", f"{diver_agregado_linhas_dms:,}")
    r4.metric("Dup. várias escolas / mesmo número", f"{summary_actual.multiplas_escolas_mesmo_cnpj:,}")
    r5.metric("Chaves Censo duplicadas (globalmente)", f"{summary_actual.chaves_com_multiplos_cnpj_no_censo:,}")

    segundo = st.columns(4)
    segundo[0].metric("Sem correspondência censo (CNPJ DMS válido ausente censo municipal)", f"{summary_actual.sem_correspondencia_cnpj:,}")
    segundo[1].metric("Linhas DMS vazias (sem dígitos normalizados)", f"{summary_actual.sem_cnpj_dms:,}")
    segundo[2].metric("CNPJ DMS classificado como inválido (DV/formato)", f"{summary_actual.cnpj_dms_invalido:,}")
    segundo[3].metric("Tempo passe determinístico (s)", f"{summary_actual.tempo_segundos:.3f}")

    texto_extra = st.session_state.get("etapa3_comp_text_summary")
    if texto_extra:
        z1, z2, z3 = st.columns(3)
        z1.metric("Textual elegível (sem CNPJ válido inicialmente)", f"{texto_extra.linhas_elegiveis:,}")
        z2.metric("Sucesso texto complement", f"{texto_extra.matches_texto:,}")
        z3.metric("Sem match textual mesmo após extra", f"{texto_extra.sem_correspondencia:,}")
        fz_sig = st.session_state.get("etapa3_fuzzy_sig")
        if fz_sig:
            fra, fro, fc = fz_sig
            st.caption(f"Último passe texto rápido: **`{fra}` × `{fro}`** · cutoff WRatio **≥ {fc}**.")

    st.subheader("Pré-visualização + export")
    filt = st.radio(
        "Segmentar resultado",
        [
            "Todos",
            "Só alta confiança (match_cnpj_exato)",
            "Divergência — várias escolas / mesmo número",
            "Sem correspondência censo mesmo com CNPJ DMS válido",
            "Sem CNPJ normalizável na DMS (+ inválidos — elegível ao texto opcional)",
            "Só passe textual complementar",
            "Linhas mesmo sem resultado textual opcional aplicado",
        ],
        horizontal=True,
        key="etapa3_preview_filter",
    )
    view_df = refinado
    estado = refinado["match_status_principal"].astype(str) if "match_status_principal" in refinado.columns else None

    if estado is not None and filt == "Só alta confiança (match_cnpj_exato)":
        view_df = refinado.loc[estado.eq(MATCH_CNPJ_EXATO)].copy()
    elif estado is not None and filt == "Divergência — várias escolas / mesmo número":
        view_df = refinado.loc[estado.eq(MATCH_MULTIPLAS_ESCOLAS)].copy()
    elif estado is not None and filt == "Sem correspondência censo mesmo com CNPJ DMS válido":
        view_df = refinado.loc[estado.eq(SEM_CORRESP_CNPJ)].copy()
    elif estado is not None and filt.startswith("Sem CNPJ"):
        view_df = refinado.loc[estado.isin({SEM_CNPJ_DMS, CNPJ_INVALIDO_DMS})].copy()
    elif estado is not None and filt == "Só passe textual complementar":
        view_df = refinado.loc[estado.eq(MATCH_TEXTO_COMPLEMENTAR)].copy()
    elif estado is not None and filt.startswith("Linhas mesmo sem resultado textual"):
        view_df = refinado.loc[estado.eq(SEM_CORRESP_TEXTO)].copy()

    st.dataframe(view_df.head(200), use_container_width=True, height=420)

    buf = io.BytesIO()
    refinado.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        label="Descarregar consolidado.xlsx (último merge)",
        data=buf.getvalue(),
        file_name="consolidado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_consolidado",
    )


def _load_bundle(kind: DatasetKind, uploaded: Any, label_erro: str) -> dict[str, Any] | None:
    """Carrega upload para estrutura ``bundle`` ou devolve ``None`` com erro na UI."""

    if uploaded is None:
        return None
    raw = uploaded.getvalue()
    fname = uploaded.name
    try:
        with st.spinner(spinner_message(kind.value)):
            bundle = load_dataset_bundle(kind.value, raw, fname)
        LOG.info(
            "Carregado %s (%s): linhas=%s colunas=%s",
            dataset_kind_label(kind),
            fname,
            len(bundle["dataframe"].index),
            len(bundle["dataframe"].columns),
        )
        return bundle
    except (FileValidationError, ValueError) as exc:
        _friendly_file_error(label_erro, exc)
    except Exception as exc:  # pylint: disable=broad-except
        _friendly_file_error(label_erro, exc)
    return None


def main() -> None:
    configure_logging()
    st.set_page_config(
        page_title="DMS × Censo Escolar",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("DMS-Educação × Censo Escolar (contexto municipal)")
    st.caption(
        "**Modo simples** orienta pelo município e pré-mapeia colunas típicas de INEP. "
        "**Modo avançado** abre todas as ferramentas técnicas (mapeamentos manuais, diagnósticos detalhados)."
    )

    with st.sidebar:
        st.header("Ajuda rápida")
        st.markdown(
            "- Fluxo típico: **carregar Escola (+ Matrícula)** → definir UF/município → consolidar municipal "
            "→ **DMS** → **Etapa 2 normalizar CNPJ** → **Etapa 3 merge primeiro por CNPJ determinístico** "
            "(texto apenas sem CNPJ válido na DMS).\n"
            "- Export **consolidado** na Etapa 3.\n"
            "- Logs: `outputs/app.log` · **⋮ → Clear cache** quando trocar ficheiros grandes."
        )
        APP_DIR.mkdir(parents=True, exist_ok=True)
        (APP_DIR / "uploads").mkdir(parents=True, exist_ok=True)
        (APP_DIR / "outputs").mkdir(parents=True, exist_ok=True)

    st.header("1. Carregar bases")
    st.caption(
        "Separe sempre **DMS** (fiscal), **Escola INEP/export** e, se disponível, a agregação de **Matrículas**. "
        "O nome dos ficheiros não importa — só precisamos de CSV/XLSX íntegro."
    )

    u1, u2, u3 = st.columns(3)
    with u1:
        up_dms = st.file_uploader(
            dataset_kind_label(DatasetKind.DMS_EDUCACAO),
            type=["csv", "xlsx"],
            key="upload_slot_dms",
            help="Exportação da DMS-Educação (CSV ou Excel).",
        )
    with u2:
        up_escola = st.file_uploader(
            dataset_kind_label(DatasetKind.CENSO_ESCOLA),
            type=["csv", "xlsx"],
            key="upload_slot_censo_escola",
            help="Tabela de escolas do Censo (qualquer exercício).",
        )
    with u3:
        up_mat = st.file_uploader(
            dataset_kind_label(DatasetKind.CENSO_MATRICULA),
            type=["csv", "xlsx"],
            key="upload_slot_censo_matricula",
            help="Opcional mas recomenda-se para pré-filtrar antes do merge nacional.",
        )

    ux_mode_choice = st.radio(
        "**Modo de trabalho**",
        [UX_SIMPLES, UX_AVANCADO],
        horizontal=False,
        key="ux_flow_mode_radio",
        help="O modo simples esconde mapeamentos técnicos até ser realmente preciso corrigi-los manualmente.",
    )
    ui_simples = ux_mode_choice == UX_SIMPLES

    dms_bundle = _load_bundle(DatasetKind.DMS_EDUCACAO, up_dms, "DMS-Educação")
    escola_bundle = _load_bundle(DatasetKind.CENSO_ESCOLA, up_escola, "Censo Escola")
    mat_bundle = _load_bundle(DatasetKind.CENSO_MATRICULA, up_mat, "Censo Matrícula")

    dms_df: pd.DataFrame | None = None
    dms_meta: dict[str, Any] = {}
    df_escola: pd.DataFrame | None = None
    df_mat: pd.DataFrame | None = None

    if dms_bundle:
        dms_df = dms_bundle["dataframe"]
        dms_meta = dms_bundle.get("meta") or {}
    if escola_bundle:
        df_escola = escola_bundle["dataframe"]
    if mat_bundle:
        df_mat = mat_bundle["dataframe"]

    pv1, pv2, pv3 = st.columns(3)
    preview_h = 260 if ui_simples else 320
    with pv1:
        st.subheader("Pré-visualização — DMS")
        if dms_df is None:
            st.info("Sem ficheiro.")
        else:
            st.success(f"`{up_dms.name}` · {len(dms_df):,} × {len(dms_df.columns)}")
            if dms_meta and not ui_simples:
                with st.expander("Meta cabeçalho export DMS"):
                    slim = {k: v for k, v in dms_meta.items() if k != "columns"}
                    st.json(slim)
            st.markdown("**Amostra** — primeiras linhas")
            st.dataframe(dms_df.head(14), use_container_width=True, height=preview_h)
    with pv2:
        st.subheader("Pré-visualização — Escola")
        if df_escola is None:
            st.info("Sem ficheiro.")
        else:
            st.success(f"`{up_escola.name}` · {len(df_escola):,} × {len(df_escola.columns)}")
            st.markdown("**Amostra** — antes do recorte municipal")
            st.dataframe(df_escola.head(14), use_container_width=True, height=preview_h)
    with pv3:
        st.subheader("Pré-visualização — Matrícula")
        if df_mat is None:
            st.info("Sem ficheiro (opcional).")
        else:
            st.success(f"`{up_mat.name}` · {len(df_mat):,} × {len(df_mat.columns)}")
            st.markdown("**Amostra**")
            st.dataframe(df_mat.head(14), use_container_width=True, height=preview_h)

    st.divider()
    st.header("2. Orientar Escola ▸ Matrícula ao município")
    cols_esc = list(map(str, df_escola.columns)) if df_escola is not None else []

    if not ui_simples and df_escola is not None:
        st.markdown("#### Mapeamento manual (somente modo avançado)")
        render_logical_mapper("Tabela Escola", df_escola, CENSO_ESCOLA_FIELDS, "logical_escola")

    if not ui_simples and df_mat is not None:
        render_logical_mapper("Tabela Matrícula", df_mat, CENSO_MATRICULA_FIELDS, "logical_matricula")

    if df_escola is None:
        st.warning("Precisamos do ficheiro **Escola** para continuar até à consolidação.")
        return

    resolved_escola_map = resolve_escola_mapping(cols_esc, ui_simples=ui_simples)

    co_auto = resolved_escola_map.get("CO_ENTIDADE")
    if co_auto:
        st.success(
            f"**Identificação automática.** A coluna **`{co_auto}`** ficou ligada ao papel lógico `CO_ENTIDADE` "
            "(código típico de escola / INEP)."
        )

    geo_ctx_snapshot = render_etapa0_contexto_municipal(
        df_escola,
        resolved_escola_map,
        ui_simples=ui_simples,
    )
    exercise_year_ctx = int(geo_ctx_snapshot["exercise"])

    if st.button("Consolidar Censo municipal (Escola ⊕ Matrícula)", type="primary", key="btn_consolidar_censo"):
        map_e_click = resolve_escola_mapping(cols_esc, ui_simples=ui_simples)
        map_m_click = resolve_matricula_mapping(list(map(str, df_mat.columns)) if df_mat is not None else None, ui_simples=ui_simples)

        df_esc_eff = df_escola.copy()
        df_mat_eff = df_mat.copy() if df_mat is not None else None

        filt_stats: dict[str, str | int] = {}
        if geo_ctx_snapshot.get("filtro_ativo") and geo_ctx_snapshot.get("uf") and geo_ctx_snapshot.get("mun_code"):
            df_esc_eff, filt_stats = filter_escola_by_municipality(
                df_esc_eff,
                map_e_click,
                uf_escolha=str(geo_ctx_snapshot["uf"]),
                municipio_codigo=str(geo_ctx_snapshot["mun_code"]),
            )
            antes = filt_stats.get("antes", "")
            depois = filt_stats.get("depois", "")
            motivo_txt = filt_stats.get("motivo", "")
            if motivo_txt:
                st.warning(f"Filtro municipal não aplicável: **{motivo_txt}**")
            else:
                st.info(f"Linhas Escola **antes ▸ depois** do filtro municipal: `{antes:,}` ▸ `{depois:,}`.")

        phys_co_esc = map_e_click.get("CO_ENTIDADE")
        if (
            isinstance(df_mat_eff, pd.DataFrame)
            and phys_co_esc is not None
            and phys_co_esc in df_esc_eff.columns
        ):
            subset_co = normalize_co_entidade(df_esc_eff[phys_co_esc])
            entidades = set(subset_co.tolist())
            entidades.discard("")
            phys_co_mat = map_m_click.get("CO_ENTIDADE")
            if (
                phys_co_mat
                and isinstance(phys_co_mat, str)
                and phys_co_mat in df_mat_eff.columns
                and entidades
            ):
                df_mat_eff_, mat_trim = restrict_matricula_to_entidades(df_mat_eff, phys_co_mat, entidades)
                df_mat_eff = df_mat_eff_
                st.success(
                    f"Matrículas também **recortadas** ao mesmo conjunto de escolas municipais (`{phys_co_mat}`): "
                    f"{mat_trim['antes_mat']:,} → {mat_trim['depois_mat']:,} linhas."
                )

        fn_esc = up_escola.name if up_escola else ""
        fn_mat = up_mat.name if up_mat else ""
        merged_out: pd.DataFrame | None = None
        try:
            merged_out = consolidate_census_escolar(
                df_esc_eff,
                df_mat_eff,
                map_e_click,
                map_m_click,
                int(exercise_year_ctx),
                source_escola_label=fn_esc,
                source_matricula_label=fn_mat or "",
            )
        except CensusMergeError as exc:
            st.error(str(exc))
            LOG.warning("Consolidação Censo recusada: %s", exc)
        except Exception as exc:  # pylint: disable=broad-except
            st.error("Erro inesperado na consolidação.")
            LOG.exception("merge censo")
            st.code(str(exc))
        else:
            if merged_out is not None:
                merged_final = append_censo_context_columns(merged_out, geo_ctx_snapshot)
                st.session_state["censo_consolidado_df"] = merged_final
                LOG.debug("Mapeamento escola aplicado: %s | matricula: %s", map_e_click, map_m_click)
                st.success(f"Censo consolidado (**{len(merged_final.index):,}** linhas**) com metadados de contexto.")

                stale_sig_anchor = tuple(
                    sorted(
                        {
                            ("uf_pick", geo_ctx_snapshot.get("uf")),
                            ("mun_pick", geo_ctx_snapshot.get("mun_code")),
                            ("filtro_geo", geo_ctx_snapshot.get("filtro_ativo")),
                            ("skip_geo_flag", geo_ctx_snapshot.get("skip_geo")),
                        }
                    )
                )
                st.session_state["censo_consolidado_signature"] = (
                    fn_esc,
                    fn_mat or "",
                    int(exercise_year_ctx),
                    bool(ui_simples),
                    tuple(sorted(map_e_click.items())),
                    tuple(sorted(map_m_click.items())),
                    stale_sig_anchor,
                )

    censo_consolidado = st.session_state.get("censo_consolidado_df")
    sig_store = st.session_state.get("censo_consolidado_signature")

    resolved_now = resolve_escola_mapping(list(map(str, df_escola.columns)), ui_simples=ui_simples)
    map_m_live = resolve_matricula_mapping(list(map(str, df_mat.columns)) if df_mat is not None else None, ui_simples=ui_simples)
    cur_ctx_stub = geo_ctx_snapshot
    stale_anchor_now = tuple(
        sorted(
            {
                ("uf_pick", cur_ctx_stub.get("uf")),
                ("mun_pick", cur_ctx_stub.get("mun_code")),
                ("filtro_geo", cur_ctx_stub.get("filtro_ativo")),
                ("skip_geo_flag", cur_ctx_stub.get("skip_geo")),
            }
        )
    )
    current_sig_attempt = (
        up_escola.name if up_escola else "",
        up_mat.name if up_mat else "",
        int(cur_ctx_stub["exercise"]),
        bool(ui_simples),
        tuple(sorted(resolved_now.items())),
        tuple(sorted(map_m_live.items())),
        stale_anchor_now,
    )

    if isinstance(censo_consolidado, pd.DataFrame):
        st.subheader("Base Censo consolidada disponível nesta sessão")
        meta_cols = [
            c
            for c in censo_consolidado.columns
            if str(c).startswith(("censo_", "censo_ctx_")) or str(c) == "censo_exercicio"
        ]
        if meta_cols:
            st.caption("Metadados colados à consolidação: " + ", ".join(f"`{c}`" for c in meta_cols[:11]))
        st.dataframe(censo_consolidado.head(30), use_container_width=True, height=360)
        if sig_store is not None and sig_store != current_sig_attempt:
            st.warning(
                "Os ficheiros, o modo UX, os mapeamentos **ou** o contexto municipal/exercício mudaram "
                "**desde a última consolidação bem-sucedida**. Clique novamente em **Consolidar** para manter "
                "a base alinhada com o formulário atual."
            )

    st.divider()
    st.header("3. Cruzamento com DMS")

    _maybe_continue_dms_etapas(dms_df, censo_consolidado, ui_simples, up_dms, up_escola)


def _maybe_continue_dms_etapas(
    dms_df: pd.DataFrame | None,
    censo_consolidado: object,
    ui_simples: bool,
    up_dms: Any,
    up_escola: Any,
) -> None:
    if not isinstance(censo_consolidado, pd.DataFrame):
        st.warning(
            "Após configurar UF/município e garantir reconhecimento das colunas, clique em **Consolidar** "
            "para gerar a base única usada pela DMS."
        )
        return

    if dms_df is None:
        st.info(
            "**Carregue a DMS‑Educação** para iniciar Etapa 2 (CNPJ normalizado em memória) e Etapa 3 (matching)."
        )
        return

    dms_nm = getattr(up_dms, "name", "?")
    escola_nm = getattr(up_escola, "name", "?") if up_escola else "?"
    run_etapa2_mapping(
        dms_df,
        censo_consolidado,
        ui_simples=ui_simples,
        up_dms_name=str(dms_nm),
        up_escola_name=str(escola_nm),
    )

    cm_live = dict(st.session_state.get("column_map") or {})
    rebuilt = ensure_normalized_cnpj_workframes(
        column_map=cm_live,
        df_dms=dms_df,
        df_censo_consolidado=censo_consolidado,
    )
    if rebuilt is None:
        if ui_simples:
            st.info(
                "Precisamos de **colunas físicas válidas em ambas bases** antes do merge pela chave **CNPJ**. "
                "Use a Etapa 2 quando as mensagens de erro acima forem sanadas."
            )
        else:
            st.warning("Corrija o mapeamento de **CNPJ** na Etapa 2 para continuar até ao merge determinístico.")
        return

    dms_work_ready, censo_work_ready, cm_new = rebuilt
    st.session_state["column_map"] = cm_new
    st.session_state["dms_work"] = dms_work_ready
    st.session_state["censo_work"] = censo_work_ready
    # Não atribuir a `map_dms_cnpj` / `map_censo_cnpj` aqui: são keys de widget da Etapa 2 e o Streamlit
    # proíbe escrever `session_state[key]` depois de o widget com `key=` ser instanciado neste rerun.

    run_etapa3_merge_pipeline(dms_work_ready, censo_work_ready, ux_simples=ui_simples)


if __name__ == "__main__":
    main()

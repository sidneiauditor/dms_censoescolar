"""
App local Streamlit — cruzamento DMS-Educação × Censo Escolar.

Arquitetura:
- Carregamento **genérico** (qualquer nome de ficheiro) tipificado como DMS / Censo Escola / Censo Matrícula.
- **Campos lógicos** estáveis (CO_ENTIDADE, NO_ENTIDADE, …) mapeados sobre colunas físicas variáveis por exercício.
- **Consolidação interna** Escola ⊕ Matrícula por CO_ENTIDADE.
- Etapas seguintes operam sempre sobre o **Censo consolidado**, independentemente do ano (metadado ``censo_exercicio``).

Etapa 3: matching textual RapidFuzz entre DMS e base Censo consolidada.
"""

from __future__ import annotations

import io
import logging
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
from services.census_consolidator import CensusMergeError, consolidate_census_escolar
from services.table_loader import load_dataset_bundle, spinner_message
from services.text_fuzzy_merge import run_textual_fuzzy_merge
from utils.cnpj import add_normalized_cnpj_column, summarize_cnpj_column
from utils.file_io import FileValidationError

APP_DIR = Path(__file__).resolve().parent
SELECT_SENTINEL = "-- Selecionar coluna --"

LOG = logging.getLogger(__name__)


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


def run_etapa2_mapping(dms_df: pd.DataFrame, censo_df: pd.DataFrame) -> None:
    """Seleção dinâmica de colunas + normalização em memória."""

    st.divider()
    st.header("Etapa 2 — Normalização e validação")
    st.caption(
        "Escolha as colunas correspondentes. Os CNPJs são normalizados "
        "(só dígitos, completar zeros à esquerda até 14 posições) e validados em relação aos dígitos verificadores."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("DMS-Educação")
        dms_cnpj = st.selectbox(
            "Coluna CNPJ (DMS)",
            column_options(dms_df),
            key="map_dms_cnpj",
        )
        dms_razao = st.selectbox(
            "Razão social (DMS)",
            column_options(dms_df),
            key="map_dms_razao",
        )
        dms_qtd = st.selectbox(
            "Quantidade (DMS)",
            column_options(dms_df),
            key="map_dms_qtd",
        )

    with c2:
        st.subheader("Censo consolidado")
        opts_c = column_options(censo_df)
        censo_cnpj = st.selectbox(
            "Coluna CNPJ (Censo)",
            opts_c,
            index=default_select_index(opts_c, "CNPJ" if "CNPJ" in censo_df.columns else None),
            key="map_censo_cnpj",
        )
        censo_nome = st.selectbox(
            "Nome da escola (Censo)",
            opts_c,
            index=default_select_index(opts_c, "NO_ENTIDADE" if "NO_ENTIDADE" in censo_df.columns else None),
            key="map_censo_nome",
        )
        censo_mat = st.selectbox(
            "Quantidade de matrículas (Censo)",
            opts_c,
            index=default_select_index(opts_c, "matriculas" if "matriculas" in censo_df.columns else None),
            key="map_censo_mat",
        )

    st.subheader("Diagnóstico de CNPJ")
    left, right = st.columns(2)
    with left:
        render_cnpj_stats_block("DMS", dms_df, dms_cnpj)
    with right:
        render_cnpj_stats_block("Censo consolidado", censo_df, censo_cnpj)

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
        sample = dms_work[[dms_cnpj, "__cnpj_norm_dms"]].head(8)
        with st.expander("Pré‑visualização de CNPJ normalizado (DMS — primeiras linhas)"):
            st.dataframe(sample, use_container_width=True, hide_index=True)
        LOG.info("Mapa de colunas gravado em session_state para Etapas seguintes.")
    else:
        for key in ("dms_work", "censo_work", "column_map"):
            st.session_state.pop(key, None)
        st.info(
            "Quando selecionar colunas de CNPJ em **ambas** as bases, "
            "serão criadas colunas internas `__cnpj_norm_dms` e `__cnpj_norm_censo` "
            "em memória. A **Etapa 3** usa também **matching textual** (RapidFuzz)."
        )


def default_select_index(options: list[str], preferred: str | None) -> int:
    """Índice inicial do ``selectbox`` quando o nome preferido existe nas opções."""

    if preferred and preferred in options:
        return options.index(preferred)
    return 0


def run_etapa3_textual_merge(dms_work: pd.DataFrame, censo_work: pd.DataFrame) -> None:
    """Matching textual + métricas + preview + export ``consolidado.xlsx``."""

    st.divider()
    st.header("Etapa 3 — Cruzamento textual (RapidFuzz)")
    st.caption(
        "Para cada linha da DMS compara-se a razão social **normalizada** com os nomes "
        "do Censo consolidado (bloqueio por prefixo para bases grandes). "
        "Scorer: **WRatio**. Saída: **outputs/consolidado.xlsx**."
    )

    cm = st.session_state.get("column_map") or {}
    opts_dms = column_options(dms_work)
    opts_censo = column_options(censo_work)

    g1, g2 = st.columns(2)
    with g1:
        col_razao = st.selectbox(
            "NM razão social / texto (DMS)",
            opts_dms,
            index=default_select_index(opts_dms, cm.get("dms_razao")),
            key="etapa3_col_dms_razao",
        )
    with g2:
        col_nome = st.selectbox(
            "Nome da escola (Censo)",
            opts_censo,
            index=default_select_index(opts_censo, cm.get("censo_nome")),
            key="etapa3_col_censo_nome",
        )

    cutoff = st.radio(
        "Pontuação mínima RapidFuzz (0–100)",
        options=[70, 80, 90],
        index=1,
        horizontal=True,
        key="etapa3_cutoff",
    )

    if col_razao == SELECT_SENTINEL or col_nome == SELECT_SENTINEL:
        st.warning("Seleccione as duas colunas de texto para executar o matching.")
        return

    run_clicked = st.button("Executar matching textual", type="primary", key="btn_etapa3_run")

    if run_clicked:
        prog = st.progress(0)

        def _cb(progress: float) -> None:
            prog.progress(min(max(progress, 0.0), 1.0))

        try:
            consolidado, summary = run_textual_fuzzy_merge(
                dms_work,
                censo_work,
                col_dms_razao=col_razao,
                col_censo_nome=col_nome,
                score_cutoff=float(cutoff),
                progress_callback=_cb,
            )
        except Exception as exc:  # pylint: disable=broad-except
            prog.empty()
            st.error("Não foi possível concluir o matching textual.")
            LOG.exception("Etapa 3 — erro")
            with st.expander("Detalhe técnico"):
                st.code(str(exc))
            return

        prog.empty()

        st.session_state["consolidado_df"] = consolidado
        st.session_state["consolidado_summary"] = summary
        st.session_state["etapa3_run_signature"] = (col_razao, col_nome, int(cutoff))

        out_path = APP_DIR / "outputs" / "consolidado.xlsx"
        try:
            consolidado.to_excel(out_path, index=False, engine="openpyxl")
            LOG.info(
                "consolidado.xlsx gravado em %s (%s linhas, %s matches).",
                out_path,
                len(consolidado.index),
                summary.encontrados,
            )
            st.success(f"Consolidado gravado em `{out_path}`.")
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Gravação em disco falhou (permite mes assim descarregar): {exc}")
            LOG.exception("Falha ao gravar consolidado.xlsx")

    summary_obj = st.session_state.get("consolidado_summary")
    consolidado = st.session_state.get("consolidado_df")
    sig_stored = st.session_state.get("etapa3_run_signature")
    sig_now = (col_razao, col_nome, int(cutoff))

    if consolidado is None or summary_obj is None:
        return

    if sig_stored != sig_now:
        st.info(
            "**Parâmetros alterados** relativamente ao último resultado (colunas ou corte). "
            "Clique novamente em **Executar matching textual** para atualizar."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linhas DMS", f"{summary_obj.linhas_dms:,}")
    m2.metric("Matches encontrados", f"{summary_obj.encontrados:,}")
    m3.metric("Sem correspondência", f"{summary_obj.sem_correspondencia:,}")
    aderencia = (
        100.0 * summary_obj.encontrados / summary_obj.linhas_dms
        if summary_obj.linhas_dms
        else 0.0
    )
    m4.metric("Percentagem match", f"{aderencia:.1f} %")

    st.caption(
        f"Censo com **{summary_obj.linhas_censo:,}** registos · tempo **{summary_obj.tempo_segundos:.2f} s** · "
        f"corte **≥ {summary_obj.score_min_usado:.0f}**."
    )

    filt = st.radio(
        "Pré-visualização consolidado",
        ["Todos", "Só matches", "Só sem correspondência"],
        horizontal=True,
        key="etapa3_preview_filter",
    )
    view = consolidado
    if filt == "Só matches":
        view = consolidado.loc[consolidado["match_status"] == "match_textual"].copy()
    elif filt == "Só sem correspondência":
        view = consolidado.loc[consolidado["match_status"] == "sem_correspondencia"].copy()

    st.dataframe(view.head(200), use_container_width=True, height=420)

    buf = io.BytesIO()
    consolidado.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        label="Descarregar consolidado.xlsx",
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

    st.title("DMS-Educação × Censo Escolar")
    st.caption(
        "Carregamento genérico por **tipo de base**, consolidação Escola⊕Matrícula e etapas de validação/matching textual."
    )

    with st.sidebar:
        st.header("Metadados")
        exercise_year = st.number_input(
            "Exercício do Censo (referência)",
            min_value=1996,
            max_value=2050,
            value=2025,
            step=1,
            help="Não altera leitura de ficheiros — fica registado na base consolidada (`censo_exercicio`).",
        )
        st.divider()
        st.markdown(
            "- Processamento **offline**.\n"
            "- Qualquer nome de ficheiro CSV/XLSX.\n"
            "- **⋮ → Clear cache** após substituir ficheiros.\n"
            "- Logs: `outputs/app.log`"
        )
        APP_DIR.mkdir(parents=True, exist_ok=True)
        (APP_DIR / "uploads").mkdir(parents=True, exist_ok=True)
        (APP_DIR / "outputs").mkdir(parents=True, exist_ok=True)

    st.header("1. Carregar bases")
    st.caption(
        "Três slots independentes correspondem aos tipos **DMS Educação**, **Censo Escola** e **Censo Matrícula**. "
        "O ano do microdados é irrelevante para o nome do ficheiro."
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
            help="Tabela agregada de matrículas (opcional mas recomendada para merge).",
        )

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
    with pv1:
        st.subheader("Pré-visualização — DMS")
        if dms_df is None:
            st.info("Sem ficheiro.")
        else:
            st.success(f"`{up_dms.name}` · {len(dms_df):,} × {len(dms_df.columns)}")
            if dms_meta:
                with st.expander("Meta cabeçalho DMS"):
                    slim = {k: v for k, v in dms_meta.items() if k != "columns"}
                    st.json(slim)
            _preview_dataframe("Amostra", dms_df.head(20), "20 linhas")
    with pv2:
        st.subheader("Pré-visualização — Escola")
        if df_escola is None:
            st.info("Sem ficheiro.")
        else:
            st.success(f"`{up_escola.name}` · {len(df_escola):,} × {len(df_escola.columns)}")
            _preview_dataframe("Amostra", df_escola.head(20), "20 linhas")
    with pv3:
        st.subheader("Pré-visualização — Matrícula")
        if df_mat is None:
            st.info("Sem ficheiro (opcional).")
        else:
            st.success(f"`{up_mat.name}` · {len(df_mat):,} × {len(df_mat.columns)}")
            _preview_dataframe("Amostra", df_mat.head(20), "20 linhas")

    st.divider()
    st.header("2. Mapeamento lógico do Censo")
    st.caption(
        "Associe colunas **reais** do ficheiro a papéis estáveis (`CO_ENTIDADE`, …). "
        "Estes nomes são os utilizados na base consolidada para qualquer exercício."
    )

    if df_escola is not None:
        render_logical_mapper(
            "Tabela Escola",
            df_escola,
            CENSO_ESCOLA_FIELDS,
            "logical_escola",
        )
    else:
        st.warning("Carregue a **tabela Escola** para mapear campos obrigatórios.")

    if df_mat is not None:
        render_logical_mapper(
            "Tabela Matrícula",
            df_mat,
            CENSO_MATRICULA_FIELDS,
            "logical_matricula",
        )

    if df_escola is not None:
        if st.button("Consolidar Censo (Escola ⊕ Matrícula)", type="primary", key="btn_consolidar_censo"):
            map_e = collect_logical_mapping("logical_escola", CENSO_ESCOLA_FIELDS)
            map_m = collect_logical_mapping("logical_matricula", CENSO_MATRICULA_FIELDS)
            fn_esc = up_escola.name if up_escola else ""
            fn_mat = up_mat.name if up_mat else ""
            try:
                merged = consolidate_census_escolar(
                    df_escola,
                    df_mat,
                    map_e,
                    map_m,
                    int(exercise_year),
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
                st.session_state["censo_consolidado_df"] = merged
                LOG.debug(
                    "Mapeamento escola aplicado: %s | matricula: %s",
                    map_e,
                    map_m,
                )
                st.success(f"Censo consolidado: **{len(merged.index):,}** linhas.")
                st.session_state["censo_consolidado_signature"] = (
                    fn_esc,
                    fn_mat,
                    int(exercise_year),
                    tuple(sorted(map_e.items())),
                    tuple(sorted(map_m.items())),
                )

    censo_consolidado = st.session_state.get("censo_consolidado_df")
    sig_store = st.session_state.get("censo_consolidado_signature")

    if isinstance(censo_consolidado, pd.DataFrame):
        st.subheader("Base Censo consolidada (visão atual)")
        meta_cols = [
            c
            for c in censo_consolidado.columns
            if str(c).startswith("censo_") or str(c) == "censo_exercicio"
        ]
        if meta_cols:
            st.caption("Metadados: " + ", ".join(f"`{c}`" for c in meta_cols[:8]))
        _preview_dataframe(
            "Pré-visualização consolidado",
            censo_consolidado.head(40),
            "40 linhas · colunas lógicas + matrículas após merge",
        )

        current_sig_attempt = (
            up_escola.name if up_escola else "",
            up_mat.name if up_mat else "",
            int(exercise_year),
            tuple(sorted(collect_logical_mapping("logical_escola", CENSO_ESCOLA_FIELDS).items())),
            tuple(sorted(collect_logical_mapping("logical_matricula", CENSO_MATRICULA_FIELDS).items())),
        )
        if sig_store is not None and sig_store != current_sig_attempt:
            st.info(
                "**Nota:** ficheiros, exercício ou mapeamento mudaram desde a última consolidação. "
                "Volte a clicar em **Consolidar** para alinhar o resultado à UI."
            )

    st.divider()
    st.header("3. Cruzamento com DMS")

    if not isinstance(censo_consolidado, pd.DataFrame):
        st.warning(
            "Conclua o **mapeamento** e clique em **Consolidar Censo** para gerar a base única do Censo."
        )
        return

    if dms_df is None:
        st.info(
            "Para **Etapa 2** (CNPJ) e **Etapa 3** (matching textual), carregue também o ficheiro da **DMS-Educação**."
        )
        return

    run_etapa2_mapping(dms_df, censo_consolidado)

    dms_work = st.session_state.get("dms_work")
    censo_work = st.session_state.get("censo_work")
    if isinstance(dms_work, pd.DataFrame) and isinstance(censo_work, pd.DataFrame):
        run_etapa3_textual_merge(dms_work, censo_work)
    else:
        st.warning(
            "Complete o **mapeamento de CNPJ** na Etapa 2 para habilitar o matching textual (Etapa 3)."
        )


if __name__ == "__main__":
    main()

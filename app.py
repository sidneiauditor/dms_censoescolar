"""
App local Streamlit — cruzamento DMS-Educação × Censo Escolar (Etapas incrementais).

Etapa 2: ingestão DMS com cabeçalho automático, Censo padrão, mapeamento de colunas,
normalização métricas de CNPJ, cache Streamlit para leituras pesadas.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from services.ingest_cache import load_censo_cached, load_dms_cached
from utils.cnpj import add_normalized_cnpj_column, summarize_cnpj_column
from utils.file_io import FileValidationError

APP_DIR = Path(__file__).resolve().parent
SELECT_SENTINEL = "-- Selecionar coluna --"

LOG = logging.getLogger(__name__)


def configure_logging() -> None:
    """Logs simples para ficheiro local + stderr (sem serviços externos)."""

    log_dir = APP_DIR / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )


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
    st.dataframe(df, use_container_width=True, height=360)


def column_options(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return [SELECT_SENTINEL]
    return [SELECT_SENTINEL] + [str(c) for c in df.columns]


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
        st.subheader("Censo Escolar")
        censo_cnpj = st.selectbox(
            "Coluna CNPJ (Censo)",
            column_options(censo_df),
            key="map_censo_cnpj",
        )
        censo_nome = st.selectbox(
            "Nome da escola (Censo)",
            column_options(censo_df),
            key="map_censo_nome",
        )
        censo_mat = st.selectbox(
            "Quantidade de matrículas (Censo)",
            column_options(censo_df),
            key="map_censo_mat",
        )

    st.subheader("Diagnóstico de CNPJ")
    left, right = st.columns(2)
    with left:
        render_cnpj_stats_block("DMS", dms_df, dms_cnpj)
    with right:
        render_cnpj_stats_block("Censo", censo_df, censo_cnpj)

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
            "apenas em memória (para o cruzamento na Etapa 3)."
        )


def main() -> None:
    configure_logging()
    st.set_page_config(
        page_title="DMS × Censo Escolar",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("DMS-Educação × Censo Escolar")
    st.caption("Etapa 2 — Normalização, validação e mapeamento de colunas (100% local).")

    with st.sidebar:
        st.header("Sobre")
        st.markdown(
            "- Processamento **offline** (sem APIs / nuvem).\n"
            "- DMS: deteção automática de cabeçalho e correção de colunas `Unnamed`.\n"
            "- **Cache Streamlit** acelera reexecuções com o mesmo ficheiro.\n"
            "- Limpar cache: menu **⋮** → *Clear cache* se alterar o ficheiro e o preview não atualizar."
        )
        uploads_dir = APP_DIR / "uploads"
        outputs_dir = APP_DIR / "outputs"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        st.caption(f"Registos: `{outputs_dir / 'app.log'}`")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        dms_upload: Any = st.file_uploader(
            "DMS-Educação",
            type=["csv", "xlsx"],
            key="upload_dms",
            help="Relatório exportado da DMS (pode conter linhas de título antes da tabela).",
        )
    with col_u2:
        censo_upload: Any = st.file_uploader(
            "Censo Escolar",
            type=["csv", "xlsx"],
            key="upload_censo",
            help="Ficheiro agregado do Censo (CSV `;` ou Excel).",
        )

    dms_df: pd.DataFrame | None = None
    dms_meta: dict[str, Any] = {}
    censo_df: pd.DataFrame | None = None

    if dms_upload is not None:
        try:
            raw = dms_upload.getvalue()
            dms_df, dms_meta = load_dms_cached(raw, dms_upload.name)
            st.session_state["dms_raw_name"] = dms_upload.name
            LOG.info(
                "DMS carregada: %s linhas=%s cols=%s",
                dms_upload.name,
                len(dms_df.index),
                len(dms_df.columns),
            )
        except FileValidationError as exc:
            _friendly_file_error("DMS-Educação", exc)
        except Exception as exc:  # pylint: disable=broad-except
            _friendly_file_error("DMS-Educação", exc)

    if censo_upload is not None:
        try:
            raw_c = censo_upload.getvalue()
            censo_df = load_censo_cached(raw_c, censo_upload.name)
            st.session_state["censo_raw_name"] = censo_upload.name
            LOG.info(
                "Censo carregado: %s linhas=%s cols=%s",
                censo_upload.name,
                len(censo_df.index),
                len(censo_df.columns),
            )
        except FileValidationError as exc:
            _friendly_file_error("Censo Escolar", exc)
        except Exception as exc:  # pylint: disable=broad-except
            _friendly_file_error("Censo Escolar", exc)

    prev1, prev2 = st.columns(2)
    with prev1:
        st.subheader("Pré‑visualização — DMS")
        if dms_df is None:
            st.info("Carregue o ficheiro da DMS para ver a amostra.")
        else:
            st.success(
                f"`{dms_upload.name}` — **{len(dms_df):,}** linhas · **{len(dms_df.columns)}** colunas"
            )
            with st.expander("Metadados da deteção de cabeçalho", expanded=False):
                meta_compact = {
                    key: value
                    for key, value in dms_meta.items()
                    if key != "columns"
                }
                st.json(meta_compact)
                st.caption("Lista de colunas após saneamento:")
                st.dataframe(
                    pd.DataFrame({"coluna": dms_meta.get("columns", [])}),
                    use_container_width=True,
                    height=240,
                    hide_index=True,
                )
            _preview_dataframe("Amostra de dados", dms_df.head(30), "30 primeiras linhas")
    with prev2:
        st.subheader("Pré‑visualização — Censo")
        if censo_df is None:
            st.info("Carregue o ficheiro do Censo para ver a amostra.")
        else:
            st.success(
                f"`{censo_upload.name}` — **{len(censo_df):,}** linhas · **{len(censo_df.columns)}** colunas"
            )
            _preview_dataframe("Amostra de dados", censo_df.head(30), "30 primeiras linhas")

    if dms_df is not None and censo_df is not None:
        run_etapa2_mapping(dms_df, censo_df)
    elif dms_df is not None or censo_df is not None:
        st.divider()
        st.warning("Carregue **ambos** os ficheiros para concluir o mapeamento da Etapa 2.")


if __name__ == "__main__":
    main()

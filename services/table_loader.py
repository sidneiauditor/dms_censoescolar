"""Carregamento genérico em cache por tipo de base (sem nomes fixos de ficheiro)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from domain.dataset_kind import DatasetKind
from utils.dms_ingest import fix_unnamed_and_empty_columns, load_dms_with_smart_header
from utils.file_io import FileValidationError, load_dataframe


@st.cache_data(show_spinner=False, max_entries=32)
def load_dataset_bundle(dataset_kind_value: str, raw: bytes, filename: str) -> dict[str, Any]:
    """
    Carrega um ficheiro CSV/XLSX conforme o tipo de base.

    Parameters
    ----------
    dataset_kind_value
        Um valor de ``DatasetKind`` (string).
    filename
        Nome original escolhido pelo utilizador — **só** para inferir extensão (CSV/XLSX).

    Returns
    -------
    dict
        ``dataframe``, ``meta`` (metadados extras), ``dataset_kind``.
    """

    if not raw:
        raise FileValidationError("Ficheiro vazio.")

    kind = DatasetKind(dataset_kind_value)

    if kind == DatasetKind.DMS_EDUCACAO:
        df, meta = load_dms_with_smart_header(filename, raw)
        return {
            "dataframe": df,
            "meta": meta,
            "dataset_kind": kind.value,
        }

    df_plain = load_dataframe(filename, raw)
    df_clean = fix_unnamed_and_empty_columns(df_plain)
    return {
        "dataframe": df_clean,
        "meta": {"dataset_kind": kind.value, "filename": filename},
        "dataset_kind": kind.value,
    }


def spinner_message(kind_value: str) -> str:
    kind = DatasetKind(kind_value)
    if kind == DatasetKind.DMS_EDUCACAO:
        return "A processar DMS-Educação…"
    if kind == DatasetKind.CENSO_ESCOLA:
        return "A carregar tabela Censo Escola…"
    return "A carregar tabela Censo Matrícula…"

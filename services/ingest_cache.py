"""Funções pesadas encapsuladas com ``st.cache_data`` (Etapa 2+)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from utils.dms_ingest import fix_unnamed_and_empty_columns, load_dms_with_smart_header
from utils.file_io import load_dataframe


@st.cache_data(show_spinner="A processar DMS-Educação (deteção de cabeçalho)…", max_entries=16)
def load_dms_cached(raw: bytes, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replica em cache o pipeline completo da DMS."""

    df, meta = load_dms_with_smart_header(filename, raw)
    return df, meta


@st.cache_data(show_spinner="A carregar dados do Censo…", max_entries=16)
def load_censo_cached(raw: bytes, filename: str) -> pd.DataFrame:
    """Censo: leitura padrão + saneamento ``Unnamed`` / colunas vazias."""

    base = load_dataframe(filename, raw)
    return fix_unnamed_and_empty_columns(base)

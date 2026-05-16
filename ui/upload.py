"""Área única «Carregar arquivos» — compacta para modo operacional; reutilizável no técnico."""

from __future__ import annotations

from typing import Any

import streamlit as st

from domain.dataset_kind import DatasetKind, label as dataset_kind_label


def render_secao_carregar_arquivos(
    *,
    compacto_operacional: bool,
) -> tuple[Any | None, Any | None, Any | None]:
    """
    Três uploads DMS × Escola × Matrícula.

    ``compacto_operacional``: menos texto e linguagem dirigida ao auditor.
    """

    if compacto_operacional:
        st.subheader("Carregar arquivos")
        st.caption(
            "Envie os três conjuntos habitualmente utilizados pela CIF neste cruzamento. "
            "Formatos: CSV ou Excel (.xlsx)."
        )
        c1, c2, c3 = st.columns(3)
    else:
        st.header("1. Carregar bases")
        st.caption(
            "Separe sempre **DMS** (fiscal), **Escola INEP/export** e, se disponível, **Matrículas**. "
            "O nome dos ficheiros pode ser qualquer — o conteúdo é validado pela estrutura."
        )
        c1, c2, c3 = st.columns(3)

    with c1:
        up_dms = st.file_uploader(
            "DMS-Educação" if compacto_operacional else dataset_kind_label(DatasetKind.DMS_EDUCACAO),
            type=["csv", "xlsx"],
            key="upload_slot_dms",
            help="Extrato/contabilização da DMS-Educação (CSV ou Excel).",
        )
    with c2:
        lab_esc = (
            "Censo Escola (INEP / exportação)"
            if compacto_operacional
            else dataset_kind_label(DatasetKind.CENSO_ESCOLA)
        )
        up_escola = st.file_uploader(lab_esc, type=["csv", "xlsx"], key="upload_slot_censo_escola")
    with c3:
        lab_mat = (
            "Censo Matrícula"
            if compacto_operacional
            else dataset_kind_label(DatasetKind.CENSO_MATRICULA)
        )
        up_mat = st.file_uploader(lab_mat, type=["csv", "xlsx"], key="upload_slot_censo_matricula")

    return up_dms, up_escola, up_mat

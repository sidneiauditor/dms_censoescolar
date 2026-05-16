"""Painel Salvador mínimo — só Streamlit; cálculos em :mod:`services.enrollment_divergence`."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from services.enrollment_divergence import (
    build_enrollment_divergence_table,
    compute_enrollment_kpis,
    describe_column_bindings,
)


def render_operacional_enrollment_dashboard(df: pd.DataFrame, column_map: dict[str, Any]) -> None:
    """
    KPIs compactos no topo + tabela de divergências (sem gráficos).
    Mantém apenas o trabalho visual; agregações e ordenação ficam na camada ``services``.
    """

    cm = dict(column_map or {})
    kpis = compute_enrollment_kpis(df, cm)
    st.subheader("Divergências DMS × Censo (matrículas)")
    st.caption(
        "Comparação por linha do consolidado: **matrículas declaradas na DMS** vs **total municipal (Censo)**. "
        "Ordenação: maior **|percentual|** primeiro."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total escolas", f"{kpis.total_escolas:,}")
    c2.metric("Match exato", f"{kpis.match_exato:,}")
    c3.metric("Total divergências", f"{kpis.total_divergencias:,}")
    c4.metric("ISS total", f"{kpis.total_iss:,.2f}")

    tabela = build_enrollment_divergence_table(df, cm)

    bindings = describe_column_bindings(df, cm)
    faltantes: list[str] = []
    if not bindings.get("matriculas_censo_prefixed"):
        faltantes.append("coluna matrículas Censo (`censo__…`)")
    if not bindings.get("matriculas_dms_prefixed"):
        faltantes.append("coluna matrículas ou quantidade DMS (`dms__…`)")
    if faltantes:
        st.markdown(
            "- **Resolução de colunas incompleta:** " + "; ".join(faltantes) + "."
        )

    if not bindings.get("iss_prefixed"):
        st.markdown(
            "- **ISS:** campo não encontrado sob prefixo `dms__` — o KPI *ISS total* fica zero até existir VLIMPOSTO ou nome equivalente."
        )

    st.dataframe(tabela, use_container_width=True, height=560, hide_index=True)

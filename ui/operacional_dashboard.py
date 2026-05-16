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


def render_operacional_enrollment_dashboard(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None = None,
    censo_work: pd.DataFrame | None = None,
) -> None:
    """KPIs compactos no topo + tabela de divergências (contribuintes agregados por CNPJ)."""

    cm = dict(column_map or {})
    kpis = compute_enrollment_kpis(df, cm, dms_work=dms_work, censo_work=censo_work)
    st.subheader("Divergências DMS × Censo (matrículas)")
    st.caption(
        "Por **contribuinte (CNPJ)** após Etapa 8.1: soma **`QUANTIDADE` na DMS** vs **`QT_MAT_BAS` no extracto municipal** "
        "(Educação Básica — total oficial não passível de somar com subconjuntos hierárquicos). "
        "Ordenação: maior **|percentual|** primeiro."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contribuintes (CNPJ)", f"{kpis.total_escolas:,}")
    c2.metric("Match exato", f"{kpis.match_exato:,}")
    c3.metric("Total divergências", f"{kpis.total_divergencias:,}")
    c4.metric("ISS total", f"{kpis.total_iss:,.2f}")

    tabela = build_enrollment_divergence_table(df, cm, dms_work=dms_work, censo_work=censo_work)

    bindings = describe_column_bindings(df, cm, dms_work=dms_work, censo_work=censo_work)

    modo = bindings.get("aggregation_mode")
    qt_bas = bindings.get("qt_mat_bas_physical_censo_work")
    if modo == "aggregated_frames":
        if qt_bas:
            st.markdown(
                f"- **Etapa 8.1:** agregação por `censo_work` / `dms_work` — total municipal apenas **`QT_MAT_BAS`** (coluna física `{qt_bas}`)."
            )
        else:
            st.markdown(
                "- **Etapa 8.1:** agregação por frames de trabalho, mas **`QT_MAT_BAS`** não aparece nomeado nos cabeçalhos do Censo — confirme o extracto Escola⊕Matrícula."
            )
    if bindings.get("aggregation_note"):
        st.markdown("- " + bindings["aggregation_note"])

    if modo == "fallback_consolidado" and not bindings.get("censo_prefixed_QT_MAT_BAS_consolidado"):
        st.markdown(
            "- **Fallback:** só disponível consolidado ou falta **`censo__QT_MAT_BAS`** — processe novamente os três ficheiros em **Salvador** para materializar trabalho completo por CNPJ."
        )

    st.dataframe(tabela, use_container_width=True, height=560, hide_index=True)

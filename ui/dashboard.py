"""Painel operacional Etapa 6.2 — só apresentação; agregações em ``services.dashboard_metrics``."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from services.dashboard_metrics import (
    MatriculaFaixa,
    build_operational_table,
    compute_divergence_counts,
    compute_kpis,
    compute_rankings,
    figure_bar_iss_by_status,
    figure_donut_match_status,
    figure_scatter_matriculas_iss,
    filter_operational_dataframe,
    resolved_paths_for_dashboard,
)


def render_dashboard_ranking_fiscal(
    df: pd.DataFrame,
    cm: dict[str, Any],
    *,
    lingua_cif_operacional: bool,
) -> None:
    """

    lingua_cif_operacional:
      ``True`` — rótulos para auditor (menos vocabulário de pipeline interno).

    """

    FAIXA_LABELS: dict[str, MatriculaFaixa] = {
        "Todas": "todas",
        "0 matrículas": "zero",
        "1–50 matrículas": "1_50",
        "51–200 matrículas": "51_200",
        "≥201 matrículas": "201_mais",
    }

    st.divider()
    titulo_painel = (
        "**Painel fiscal** — divergências, métricas e rankings"
        if lingua_cif_operacional
        else "Etapa 6.2 — Painel de divergências e ranking fiscal"
    )
    st.header(titulo_painel)
    st.caption(
        "Análise sobre a **base integrada** (DMS × Censo municipal). Os filtros aplicam-se antes dos totais e gráficos."
        if lingua_cif_operacional
        else "Base: **consolidado atual** (Etapa 3 + 6.1). Filtros e agregações recalculados a cada rerun."
    )

    cms = dict(cm or {})
    paths = resolved_paths_for_dashboard(df, cms)

    exp_aberto = not lingua_cif_operacional
    with st.expander(
        "Notas técnicas (arquitectura do painel)" if lingua_cif_operacional else "Arquitetura do painel 6.2 (equipa técnica)",
        expanded=exp_aberto,
    ):
        st.markdown(
            "- **Agregações** em `services.dashboard_metrics` (sem lógica neste ficheiro).\n"
            "- **Filtros** por situação de cruzamento, dependência administrativa e faixa de matrículas.\n"
            "- **Gráficos** Plotly (donut, barras, dispersão).\n"
            "- **Tabela operacional** para uso imediato pelo auditor."
        )

    status_col = "match_status_principal"
    status_vals: list[str] = []
    if status_col in df.columns:
        status_vals = sorted(df[status_col].dropna().astype(str).unique().tolist())

    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        sel_stat = st.multiselect(
            "Situação do cruzamento" if lingua_cif_operacional else "Filtro: `match_status_principal`",
            options=status_vals,
            default=status_vals if status_vals else [],
            disabled=not bool(status_vals),
            key="dash_filter_match_status",
            help="Vazio = sem filtro por situação.",
        )
    dep_col = paths.get("dependencia")
    dep_vals: list[str] = []
    if dep_col and dep_col in df.columns:
        dep_vals = sorted({str(x).strip() for x in df[dep_col].dropna().unique().tolist() if str(x).strip()})
    with c_f2:
        sel_dep = st.multiselect(
            "Dependência administrativa (Censo)",
            options=dep_vals,
            default=[],
            key="dash_filter_dependencia",
            disabled=not dep_vals,
            help="Vazio = sem filtro.",
        )
    with c_f3:
        faixa_lbl = st.radio(
            "Faixa de matrícula (Censo)",
            options=list(FAIXA_LABELS.keys()),
            key="dash_filter_faixa_matricula",
        )

    faixa: MatriculaFaixa = FAIXA_LABELS[faixa_lbl]
    dash_df = filter_operational_dataframe(
        df,
        column_map=cms,
        match_status=sel_stat if sel_stat else None,
        dependencia=sel_dep if sel_dep else None,
        matricula_faixa=faixa,
    )

    kpis = compute_kpis(dash_df, cms)
    divs = compute_divergence_counts(dash_df, cms)
    rankings = compute_rankings(dash_df, cms, top_n=12)

    g0, g1, g2, g3, g4, g5 = st.columns(6)
    g0.metric("Linhas filtradas", f"{kpis.linhas_filtradas:,}")
    g1.metric("Total ISS", f"{kpis.total_iss:,.2f}")
    g2.metric("Σ matrículas (linhas)", f"{kpis.total_matriculas:,.0f}")
    g3.metric("Escolas (CNPJ DMS distintos)", f"{kpis.total_escolas:,}")
    g4.metric("Cruzamento exacto (linhas)", f"{kpis.total_match_exato:,}")
    g5.metric("Sem correspondência (linhas)", f"{kpis.total_sem_correspondencia:,}")

    st.subheader("Divergências (subconjunto filtrado)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Sem correspondência CNPJ", f"{divs.sem_correspondencia:,}")
    d2.metric("Múltiplas escolas / CNPJ", f"{divs.multiplas_escolas:,}")
    d3.metric("Sem matrícula ou ≤0", f"{divs.sem_matricula:,}")
    d4.metric("CNPJ DMS inválido", f"{divs.cnpj_invalido:,}")

    st.subheader("Rankings (top 12 linhas)")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**Maior ISS (valor bruto)**")
        st.dataframe(rankings.top_iss, use_container_width=True, height=260, hide_index=True)
    with r2:
        st.markdown("**Maior ISS por matrícula**")
        st.dataframe(rankings.top_iss_por_matricula, use_container_width=True, height=260, hide_index=True)
    with r3:
        st.markdown("**Maior mensalidade por aluno**")
        st.dataframe(rankings.top_mensalidade_por_aluno, use_container_width=True, height=260, hide_index=True)

    st.subheader("Gráficos")
    gc1, gc2 = st.columns(2)
    with gc1:
        st.plotly_chart(
            figure_donut_match_status(dash_df),
            use_container_width=True,
            key="dash_plotly_donut_match",
        )
    with gc2:
        st.plotly_chart(
            figure_bar_iss_by_status(dash_df, cms),
            use_container_width=True,
            key="dash_plotly_bar_iss_status",
        )
    st.plotly_chart(
        figure_scatter_matriculas_iss(dash_df, cms),
        use_container_width=True,
        key="dash_plotly_scatter_mat_iss",
    )

    st.subheader("Tabela operacional (até 500 linhas filtradas)")
    st.dataframe(
        build_operational_table(dash_df, cms, max_rows=500),
        use_container_width=True,
        height=420,
        hide_index=True,
    )

    if not paths.get("iss"):
        st.warning("Não foi detectada coluna de **ISS** na DMS integrada — alguns totais e gráficos podem ficar vazios.")
    if not paths.get("matriculas"):
        st.warning("Não foi detectada coluna de **matrículas** no Censo integrado — faixas e dispersão podem falhar.")

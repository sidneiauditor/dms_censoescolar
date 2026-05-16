"""Painel Salvador mínimo — só Streamlit; cálculos em :mod:`services.enrollment_divergence`."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from services.cnpj_aggregation import filter_dms_to_reference_month
from services.dms_quantidade_exploration import build_quantidade_exploration
from services.enrollment_divergence import (
    GRANULARITY_ESTABLISHMENT,
    GRANULARITY_ROOT,
    Granularity,
    build_enrollment_divergence_table,
    compute_enrollment_kpis,
    describe_column_bindings,
    get_merged_aggregate_for_audits,
)


def render_operacional_enrollment_dashboard(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None = None,
    censo_work: pd.DataFrame | None = None,
) -> None:
    """Painel de divergências com granularidade CNPJ 14 ou raiz 8 e filtro mensal DMS (Etapa 8.2)."""

    cm = dict(column_map or {})
    ref_year = int(st.session_state.get("ctx_exercise_default", 2025))

    choice = st.radio(
        "Granularidade da análise",
        ("Estabelecimento (CNPJ 14)", "Grupo económico (raiz 8)"),
        horizontal=True,
        key="cif_ops_enrollment_granularity",
    )
    granularity: Granularity = GRANULARITY_ESTABLISHMENT if choice.startswith("Estabelecimento") else GRANULARITY_ROOT

    only_private = st.checkbox(
        "Apenas rede privada (exclui públicas do Censo)",
        value=True,
        key="cif_ops_only_private_censo",
        help=(
            "Remove do Censo escolas com TP_DEPENDENCIA 1, 2 ou 3 (imunes/isentas de ISS). "
            "Requer coluna de dependência no consolidado (ex.: dependencia_administrativa)."
        ),
    )
    exclude_superior = st.checkbox(
        "Excluir ensino superior sem Educação Básica (QT_MAT_BAS = 0)",
        value=True,
        key="cif_ops_exclude_superior_puro",
        help=(
            "Escolas privadas só de nível superior costumam não ter matrícula no Censo Básico; "
            "com DMS = 0 geram falsa «omissão». Mantém redes com colégio + faculdade (QT_MAT_BAS > 0)."
        ),
    )

    kw = {
        "reference_year": ref_year,
        "use_reference_month": True,
        "reference_month": 5,
        "only_private_censo": only_private,
        "exclude_superior_puro": exclude_superior,
    }

    kpis = compute_enrollment_kpis(
        df, cm, dms_work=dms_work, censo_work=censo_work, granularity=granularity, **kw
    )

    st.subheader("Divergências DMS × Censo (matrículas)")
    if granularity == GRANULARITY_ROOT:
        st.caption(
            "Por **raiz do CNPJ (8 dígitos)** — **`QUANTIDADE` (DMS)** no **mês de referência (maio)** vs "
            "**`QT_MAT_BAS` (Censo)** por grupo económico. Ordenação por **|percentual|**."
        )
    else:
        st.caption(
            "Por **CNPJ 14** — **`QUANTIDADE`** no mês de referência (maio ou fallback) vs **`QT_MAT_BAS`**. "
            "Ordenação por **|percentual|**."
        )

    c1, c2, c3, c4 = st.columns(4)
    if granularity == GRANULARITY_ROOT:
        c1.metric("Grupos económicos", f"{kpis.unidades_analise:,}")
        c2.metric("Match exato", f"{kpis.match_exato:,}")
        c3.metric("Total divergências", f"{kpis.total_divergencias:,}")
        c4.metric("ISS total", f"{kpis.total_iss:,.2f}")
        e1, e2, e3 = st.columns(3)
        e1.metric("Estabelecimentos (Σ filiais DMS)", f"{kpis.total_estabelecimentos:,}")
        e2.metric(
            "Média filiais / raiz",
            f"{kpis.media_filiais_por_raiz:,.2f}" if kpis.media_filiais_por_raiz is not None else "—",
        )
        e3.metric("—", "—")
    else:
        c1.metric("Contribuintes (CNPJ 14)", f"{kpis.unidades_analise:,}")
        c2.metric("Match exato", f"{kpis.match_exato:,}")
        c3.metric("Total divergências", f"{kpis.total_divergencias:,}")
        c4.metric("ISS total", f"{kpis.total_iss:,.2f}")

    _, merge_meta = get_merged_aggregate_for_audits(
        df, cm, dms_work=dms_work, censo_work=censo_work, granularity=granularity, **kw
    )
    if isinstance(merge_meta, dict):
        if merge_meta.get("dependencia_col_missing") and only_private:
            st.warning(
                "Coluna **TP_DEPENDENCIA** / **dependencia_administrativa** não encontrada no Censo — "
                "públicas isentas **não** foram excluídas. Reprocesse com a tabela **Escola** do INEP."
            )
        if only_private or exclude_superior:
            parts: list[str] = []
            n_priv = merge_meta.get("n_privadas_censo")
            if isinstance(n_priv, int) and only_private:
                parts.append(f"**{n_priv:,}** linhas privadas com EB no cruzamento")
            n_pub = merge_meta.get("n_publicas_excluidas")
            if isinstance(n_pub, int) and n_pub > 0:
                parts.append(f"**{n_pub:,}** públicas excluídas")
            n_sup = merge_meta.get("n_superior_puro_excluidas")
            if isinstance(n_sup, int) and n_sup > 0 and exclude_superior:
                parts.append(f"**{n_sup:,}** sem QT_MAT_BAS excluídas (superior puro)")
            dep_col = merge_meta.get("dependencia_col")
            if parts:
                extra = f" (col. `{dep_col}`)" if dep_col else ""
                st.caption("Censo filtrado: " + " · ".join(parts) + extra)
    ref_month_meta = merge_meta.get("ref_month_meta") if isinstance(merge_meta.get("ref_month_meta"), dict) else {}
    st.session_state["ref_month_meta"] = ref_month_meta

    if ref_month_meta:
        with st.expander("Mês de referência DMS (QUANTIDADE)", expanded=False):
            mes_usado = ref_month_meta.get("reference_month_used", "?")
            ano_usado = ref_month_meta.get("reference_year", "?")
            st.markdown(
                f"**Mês de referência usado:** {mes_usado}/{ano_usado}  \n"
                f"CNPJs com maio disponível: **{ref_month_meta.get('cnpjs_com_maio', 0):,}**  \n"
                f"CNPJs com mês alternativo (fallback): **{ref_month_meta.get('cnpjs_com_fallback', 0):,}**  \n"
                f"CNPJs sem competência válida: **{ref_month_meta.get('cnpjs_sem_competencia', 0):,}**"
            )
            if ref_month_meta.get("coluna_competencia_nao_encontrada"):
                st.warning(
                    "Coluna de competência não encontrada na DMS. "
                    "QUANTIDADE foi somada sobre todos os meses (comportamento legado)."
                )
            fb = ref_month_meta.get("fallback_detail")
            if isinstance(fb, pd.DataFrame) and not fb.empty:
                st.dataframe(fb, use_container_width=True, hide_index=True)

    tabela = build_enrollment_divergence_table(
        df, cm, dms_work=dms_work, censo_work=censo_work, granularity=granularity, **kw
    )

    bindings = describe_column_bindings(
        df, cm, dms_work=dms_work, censo_work=censo_work, granularity=granularity, **kw
    )

    modo = bindings.get("aggregation_mode")
    qt_bas = bindings.get("qt_mat_bas_physical_censo_work")
    if modo == "aggregated_frames" and granularity == GRANULARITY_ESTABLISHMENT and qt_bas:
        st.markdown(f"- **Etapa 8.1/8.2:** por estabelecimento — **`QT_MAT_BAS`** (`{qt_bas}`), DMS no mês de referência.")
    if modo == "aggregated_frames_root" and granularity == GRANULARITY_ROOT:
        st.markdown(
            "- **Etapa 8.2:** totais por **raiz**; DMS filtrada por competência mensal antes da soma."
        )
    if bindings.get("aggregation_note"):
        st.markdown("- " + bindings["aggregation_note"])

    if modo == "fallback_consolidado" and not bindings.get("censo_prefixed_QT_MAT_BAS_consolidado"):
        st.markdown(
            "- **Fallback:** reprocessar em **Salvador** com `censo_work`/`dms_work` quando possível."
        )

    st.dataframe(tabela, use_container_width=True, height=520, hide_index=True)

    dms_eff = dms_work if isinstance(dms_work, pd.DataFrame) else pd.DataFrame()
    dms_expl = dms_eff
    if isinstance(dms_eff, pd.DataFrame) and not dms_eff.empty:
        dms_expl, _ = filter_dms_to_reference_month(
            dms_eff, column_map=cm, reference_month=5, reference_year=ref_year
        )

    merged14, _ = get_merged_aggregate_for_audits(
        df, cm, dms_work=dms_work, censo_work=censo_work, granularity=GRANULARITY_ESTABLISHMENT, **kw
    )
    merged_r, _ = get_merged_aggregate_for_audits(
        df, cm, dms_work=dms_work, censo_work=censo_work, granularity=GRANULARITY_ROOT, **kw
    )
    expl = build_quantidade_exploration(
        dms_expl,
        cm,
        merged_by_cnpj14=merged14,
        merged_by_root=merged_r,
        ref_month_meta=ref_month_meta,
    )

    with st.expander("Exploração descritiva — campo **QUANTIDADE** (DMS)", expanded=False):
        st.caption(
            "Sem interpretação automática sobre o significado de **QUANTIDADE**. "
            "Estatísticas abaixo usam o **mesmo recorte mensal** da agregação quando há coluna de competência."
        )
        if expl.get("referencia_mensal_meta") is not None:
            st.markdown("##### Competência mensal (fallback por CNPJ)")
            st.dataframe(expl["referencia_mensal_meta"], use_container_width=True, hide_index=True)
        if expl.get("distribuicao_linha_dms") is not None:
            st.markdown("##### Distribuição (linhas após filtro mensal)")
            st.dataframe(expl["distribuicao_linha_dms"], use_container_width=True, hide_index=True)
        if expl.get("top_altos") is not None:
            st.markdown("##### Maiores valores observados (amostra)")
            st.dataframe(expl["top_altos"], use_container_width=True, hide_index=True)
        if expl.get("top_baixos_nao_zero") is not None:
            st.markdown("##### Menores valores não nulos (amostra)")
            st.dataframe(expl["top_baixos_nao_zero"], use_container_width=True, hide_index=True)
        if expl.get("por_cnpj14_proporcao") is not None:
            st.markdown("##### Proporção QUANTIDADE_agregada ÷ QT_MAT_BAS (por CNPJ 14)")
            st.dataframe(expl["por_cnpj14_proporcao"].head(300), use_container_width=True, hide_index=True)
        if expl.get("por_raiz_resumo") is not None:
            st.markdown("##### Somas por raiz e razão QUANT_SUM ÷ MAT_BAS_SUM")
            st.dataframe(expl["por_raiz_resumo"].head(300), use_container_width=True, hide_index=True)

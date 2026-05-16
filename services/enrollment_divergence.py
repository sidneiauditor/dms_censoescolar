"""
Divergências operacionais entre matrículas **Censo** (`QT_MAT_BAS`) e **DMS** (`QUANTIDADE`).

Etapa **8.1** — agregação por **CNPJ 14**; Etapa **8.2** — raiz do CNPJ (8 dígitos) e filtro mensal DMS (maio).

Sem Streamlit — ver :mod:`ui.operacional_dashboard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from services.census_semantics import CENSUS_CANONICAL_FIELDS, physical_qt_mat_bas_column
from services.cnpj_aggregation import (
    AGG_BASE,
    AGG_ISS,
    AGG_MC,
    AGG_QTY,
    AGG_RAZAO,
    CNPJ_NORM_COL_CENSO,
    CNPJ_NORM_COL_DMS,
    INTERNAL_CNPJ,
    aggregate_census_by_cnpj,
    aggregate_dms_by_cnpj,
    filter_censo_for_fiscal_panel,
    filter_dms_to_reference_month,
    merge_aggregates_by_cnpj,
)
from services.cnpj_root_aggregation import (
    AGG_N_ESTAB_DMS,
    AGG_RAZAO_VARIANTES,
    CNPJ_RAIZ_COL,
    aggregate_census_by_cnpj_root,
    aggregate_dms_by_cnpj_root,
    merge_root_aggregates,
)
from services.dashboard_metrics import resolved_paths_for_dashboard
from services.inferred_mapping import pick_column
from utils.cnpj import normalize_cnpj_digits

Granularity = Literal["establishment", "economic_root"]

GRANULARITY_ESTABLISHMENT: Granularity = "establishment"
GRANULARITY_ROOT: Granularity = "economic_root"

COL_CHAVE_CNPJ = "CNPJ (14 dígitos)"
COL_CHAVE_RAIZ = "Raiz CNPJ (8 dígitos)"
COL_RAZAO = "Razão social"
COL_MC = "Matrículas Censo"
COL_MD = "Matrículas DMS"
COL_DIFF_ABS = "Diferença absoluta"
COL_DIFF_PCT = "Diferença percentual"
COL_VARIANTES = "Outras razões (variantes)"


def _resolve_census_co_entidade(censo_df: pd.DataFrame) -> str | None:
    hit = pick_column(
        [str(c) for c in censo_df.columns],
        ("CO_ENTIDADE", "COD_ESCOLA", "CODESCOLA", "INEP_ESCOLA"),
    )
    return hit if hit and hit in censo_df.columns else None


def _resolve_census_qt_bas_prefixed(consolidado_df: pd.DataFrame) -> str | None:
    target = CENSUS_CANONICAL_FIELDS["matriculas_total"].upper()
    for c in consolidado_df.columns:
        cs = str(c).strip()
        if not cs.upper().startswith("CENSO__"):
            continue
        phys = cs.split("__", 1)[1] if "__" in cs else ""
        if phys.upper() == target:
            return cs
    return None


def _fallback_merged_from_consolidado(
    consolidado_df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    granularity: Granularity,
    only_private_censo: bool = True,
    exclude_superior_puro: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = resolved_paths_for_dashboard(consolidado_df, column_map or {})
    cnpj_col = paths.get("cnpj_dms")
    qty_dm_col = paths.get("matriculas")
    raz_col = paths.get("razao")
    iss_col = paths.get("iss")
    censo_bas_pref = _resolve_census_qt_bas_prefixed(consolidado_df)

    if consolidado_df.empty or not cnpj_col or cnpj_col not in consolidado_df.columns:
        cols = (
            [CNPJ_RAIZ_COL, AGG_QTY, AGG_ISS, AGG_MC, AGG_BASE, AGG_RAZAO, AGG_N_ESTAB_DMS]
            if granularity == GRANULARITY_ROOT
            else [INTERNAL_CNPJ, AGG_QTY, AGG_ISS, AGG_MC, AGG_BASE, AGG_RAZAO]
        )
        msg = {"mode": "fallback_consolidado", "warn": "Coluna `dms__` CNPJ não resolvida no consolidado.", "granularity": granularity}
        return pd.DataFrame(columns=cols), msg

    norms = consolidado_df[cnpj_col].map(normalize_cnpj_digits)
    mask_ok = norms.notna()
    if not mask_ok.any():
        return pd.DataFrame(), {"mode": "fallback_consolidado", "warn": "CNPJ DMS ilegível no consolidado.", "granularity": granularity}

    stub = consolidado_df.loc[mask_ok].copy()
    stub["_k14"] = norms.loc[mask_ok].astype(str)

    censo_filter_meta: dict[str, Any] = {}
    if only_private_censo or exclude_superior_puro:
        stub, censo_filter_meta = filter_censo_for_fiscal_panel(
            stub,
            only_private=only_private_censo,
            exclude_superior_puro=exclude_superior_puro,
            keep_missing_dependencia=True,
            keep_missing_matriculas_bas=True,
        )

    qty_s = (
        pd.to_numeric(stub[qty_dm_col], errors="coerce").fillna(0.0)
        if qty_dm_col and qty_dm_col in stub.columns
        else pd.Series(0.0, index=stub.index)
    )
    iss_s = (
        pd.to_numeric(stub[iss_col], errors="coerce").fillna(0.0)
        if iss_col and iss_col in stub.columns
        else pd.Series(0.0, index=stub.index)
    )
    cen_bas = pd.to_numeric(stub[censo_bas_pref], errors="coerce") if censo_bas_pref else pd.Series(np.nan, index=stub.index)

    def _raz_first_nonempty(ser: pd.Series) -> str:
        for x in ser:
            t = str(x).strip()
            if t and t.lower() != "nan":
                return t
        return ""

    if granularity == GRANULARITY_ROOT:
        stub["_g"] = stub["_k14"].str.slice(0, 8)
    else:
        stub["_g"] = stub["_k14"]

    grp = stub.assign(_qty=qty_s, _iss=iss_s, _cen=cen_bas).groupby("_g", dropna=False, sort=False)
    qty_sum = grp["_qty"].sum()
    iss_sum = grp["_iss"].sum()
    cen_red = grp["_cen"].max()
    n_est = grp["_k14"].nunique() if granularity == GRANULARITY_ROOT else None

    if raz_col and raz_col in stub.columns:
        raz_g = grp[raz_col].apply(_raz_first_nonempty).reindex(qty_sum.index).fillna("")
    else:
        raz_g = pd.Series("", index=qty_sum.index)

    if granularity == GRANULARITY_ROOT:
        agg = pd.DataFrame(
            {
                CNPJ_RAIZ_COL: qty_sum.index.astype(str),
                AGG_QTY: qty_sum.to_numpy(dtype=float),
                AGG_ISS: iss_sum.reindex(qty_sum.index).to_numpy(dtype=float),
                AGG_MC: cen_red.reindex(qty_sum.index).to_numpy(dtype=float),
                AGG_BASE: np.zeros(len(qty_sum.index), dtype=float),
                AGG_RAZAO: raz_g.to_numpy(),
                AGG_RAZAO_VARIANTES: "",
                AGG_N_ESTAB_DMS: n_est.reindex(qty_sum.index).to_numpy(dtype=int) if n_est is not None else 0,
            }
        )
    else:
        agg = pd.DataFrame(
            {
                INTERNAL_CNPJ: qty_sum.index.astype(str),
                AGG_QTY: qty_sum.to_numpy(dtype=float),
                AGG_ISS: iss_sum.reindex(qty_sum.index).to_numpy(dtype=float),
                AGG_MC: cen_red.reindex(qty_sum.index).to_numpy(dtype=float),
                AGG_BASE: np.zeros(len(qty_sum.index), dtype=float),
                AGG_RAZAO: raz_g.to_numpy(),
            }
        )

    warn_txt = [
        (
            "Fallback consolidado por raiz: somas fiscais e max `censo__QT_MAT_BAS` por grupo de 8 dígitos."
            if granularity == GRANULARITY_ROOT
            else "Fluxo apenas com consolidado (CNPJ 14): max por CNPJ de `censo__QT_MAT_BAS` onde existir."
        ),
        "`Matrículas DMS`/`ISS`: prefixos `dms__`.",
    ]
    meta = {
        "mode": "fallback_consolidado",
        "warn": " ".join(warn_txt),
        "granularity": granularity,
        "only_private_censo": only_private_censo,
        "exclude_superior_puro": exclude_superior_puro,
        **censo_filter_meta,
    }
    return agg, meta


def merged_aggregate_internal(
    consolidado_df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None,
    censo_work: pd.DataFrame | None,
    granularity: Granularity = GRANULARITY_ESTABLISHMENT,
    reference_year: int | None = None,
    use_reference_month: bool = True,
    reference_month: int = 5,
    only_private_censo: bool = True,
    exclude_superior_puro: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cm = dict(column_map or {})
    use_frames = (
        isinstance(dms_work, pd.DataFrame)
        and isinstance(censo_work, pd.DataFrame)
        and not dms_work.empty
        and not censo_work.empty
        and CNPJ_NORM_COL_DMS in dms_work.columns
        and CNPJ_NORM_COL_CENSO in censo_work.columns
    )

    if use_frames:
        co_ent = _resolve_census_co_entidade(censo_work)
        qt_bas_phys = physical_qt_mat_bas_column(censo_work.columns)

        ref_month_meta: dict[str, Any] = {}
        dms_for_agg = dms_work
        if use_reference_month:
            dms_for_agg, ref_month_meta = filter_dms_to_reference_month(
                dms_work,
                column_map=cm,
                reference_month=reference_month,
                reference_year=reference_year,
            )

        censo_for_agg, censo_filter_meta = filter_censo_for_fiscal_panel(
            censo_work,
            only_private=only_private_censo,
            exclude_superior_puro=exclude_superior_puro,
        )
        n_privadas_ui = int(censo_filter_meta.get("n_privadas_censo", len(censo_for_agg)))

        if granularity == GRANULARITY_ROOT:
            c_agg = aggregate_census_by_cnpj_root(
                censo_for_agg,
                co_entidade_column=co_ent,
                only_private=False,
                exclude_superior_puro=False,
            )
            d_agg = aggregate_dms_by_cnpj_root(dms_for_agg, column_map=cm)
            merged = merge_root_aggregates(d_agg, c_agg, how="outer")
            return merged, {
                "mode": "aggregated_frames_root",
                "qt_mat_bas_column_physical": qt_bas_phys,
                "co_entidade_column_physical": co_ent,
                "granularity": granularity,
                "ref_month_meta": ref_month_meta,
                "only_private_censo": only_private_censo,
                "exclude_superior_puro": exclude_superior_puro,
                "n_privadas_censo": n_privadas_ui,
                **censo_filter_meta,
            }

        c_agg = aggregate_census_by_cnpj(
            censo_for_agg,
            co_entidade_column=co_ent,
            only_private=False,
            exclude_superior_puro=False,
        )
        d_agg, agg_ref_meta = aggregate_dms_by_cnpj(
            dms_for_agg,
            column_map=cm,
            use_reference_month=False,
        )
        if not ref_month_meta and agg_ref_meta:
            ref_month_meta = agg_ref_meta
        merged = merge_aggregates_by_cnpj(d_agg, c_agg, how="outer")
        return merged, {
            "mode": "aggregated_frames",
            "qt_mat_bas_column_physical": qt_bas_phys,
            "co_entidade_column_physical": co_ent,
            "granularity": granularity,
            "ref_month_meta": ref_month_meta,
            "only_private_censo": only_private_censo,
            "exclude_superior_puro": exclude_superior_puro,
            "n_privadas_censo": n_privadas_ui,
            **censo_filter_meta,
        }

    return _fallback_merged_from_consolidado(
        consolidado_df,
        cm,
        granularity=granularity,
        only_private_censo=only_private_censo,
        exclude_superior_puro=exclude_superior_puro,
    )


def _presentation_columns(merged: pd.DataFrame, *, granularity: Granularity) -> pd.DataFrame:
    if merged.empty:
        base_cols = [COL_CHAVE_CNPJ, COL_RAZAO, COL_MC, COL_MD, COL_DIFF_ABS, COL_DIFF_PCT]
        if granularity == GRANULARITY_ROOT:
            base_cols = [COL_CHAVE_RAIZ, COL_RAZAO, COL_VARIANTES, COL_MC, COL_MD, COL_DIFF_ABS, COL_DIFF_PCT]
        return pd.DataFrame(columns=base_cols)

    if granularity == GRANULARITY_ROOT:
        if CNPJ_RAIZ_COL not in merged.columns:
            return pd.DataFrame()
        key_s = merged[CNPJ_RAIZ_COL].astype(str)
        chave_label = COL_CHAVE_RAIZ
        var_s = merged.get(AGG_RAZAO_VARIANTES, pd.Series("", index=merged.index)).fillna("").astype(str)
    else:
        if INTERNAL_CNPJ not in merged.columns:
            return pd.DataFrame()
        key_s = merged[INTERNAL_CNPJ].astype(str)
        chave_label = COL_CHAVE_CNPJ
        var_s = None

    censo = pd.to_numeric(merged[AGG_MC], errors="coerce")
    dms = pd.to_numeric(merged[AGG_QTY], errors="coerce")
    raz = merged.get(AGG_RAZAO, pd.Series("", index=merged.index)).fillna("").astype(str)

    diff_abs = censo - dms
    cen_nv = censo.to_numpy(dtype=float)
    den_ok = np.isfinite(cen_nv) & (cen_nv != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_pct = (cen_nv - dms.to_numpy(dtype=float)) / cen_nv * 100.0
    diff_pct = np.where(den_ok, raw_pct, np.nan)

    data: dict[str, Any] = {
        chave_label: key_s.values,
        COL_RAZAO: raz.values,
        COL_MC: censo.values,
        COL_MD: dms.values,
        COL_DIFF_ABS: diff_abs.values,
        COL_DIFF_PCT: diff_pct,
    }
    if granularity == GRANULARITY_ROOT and var_s is not None:
        cols_order = [chave_label, COL_RAZAO, COL_VARIANTES, COL_MC, COL_MD, COL_DIFF_ABS, COL_DIFF_PCT]
        data[COL_VARIANTES] = var_s.values
        work = pd.DataFrame(data)[cols_order]
    else:
        cols_order = [chave_label, COL_RAZAO, COL_MC, COL_MD, COL_DIFF_ABS, COL_DIFF_PCT]
        work = pd.DataFrame(data)[cols_order]

    sort_key = work[COL_DIFF_PCT].abs()
    return (
        work.assign(_sort=sort_key)
        .sort_values("_sort", ascending=False, na_position="last")
        .drop(columns="_sort")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class EnrollmentDivergenceKpis:
    granularity: str
    match_exato: int
    total_divergencias: int
    total_iss: float
    """Unidades na linha de análise: contribuintes (14) ou grupos económicos (raiz)."""
    unidades_analise: int
    total_estabelecimentos: int
    """Soma dos CNPJ 14 distintos no universo DMS (fallback: igual a ``unidades_analise`` no nível estabelecimento)."""
    media_filiais_por_raiz: float | None
    """Válido só em ``economic_root``: média simples (#estabelecimentos/#grupos)."""


def compute_enrollment_kpis_from_merged(merged: pd.DataFrame, *, granularity: Granularity) -> EnrollmentDivergenceKpis:
    if merged.empty:
        return EnrollmentDivergenceKpis(
            granularity=granularity,
            match_exato=0,
            total_divergencias=0,
            total_iss=0.0,
            unidades_analise=0,
            total_estabelecimentos=0,
            media_filiais_por_raiz=None,
        )

    key_ok = INTERNAL_CNPJ in merged.columns if granularity == GRANULARITY_ESTABLISHMENT else CNPJ_RAIZ_COL in merged.columns
    if not key_ok:
        return EnrollmentDivergenceKpis(
            granularity=granularity,
            match_exato=0,
            total_divergencias=0,
            total_iss=0.0,
            unidades_analise=0,
            total_estabelecimentos=0,
            media_filiais_por_raiz=None,
        )

    censo = pd.to_numeric(merged[AGG_MC], errors="coerce")
    dms = pd.to_numeric(merged[AGG_QTY], errors="coerce")
    comparable = censo.notna() & dms.notna()

    matched = comparable & censo.eq(dms)
    diverged = comparable & ~censo.eq(dms)

    total_iss = float(pd.to_numeric(merged[AGG_ISS], errors="coerce").fillna(0).sum()) if AGG_ISS in merged.columns else 0.0

    n_unidades = len(merged.index)

    if granularity == GRANULARITY_ROOT:
        if AGG_N_ESTAB_DMS in merged.columns:
            total_estab = int(pd.to_numeric(merged[AGG_N_ESTAB_DMS], errors="coerce").fillna(0).sum())
        else:
            total_estab = n_unidades
        media_f = float(total_estab / n_unidades) if n_unidades else None
        return EnrollmentDivergenceKpis(
            granularity=granularity,
            match_exato=int(matched.sum()),
            total_divergencias=int(diverged.sum()),
            total_iss=total_iss,
            unidades_analise=n_unidades,
            total_estabelecimentos=total_estab,
            media_filiais_por_raiz=media_f,
        )

    n_est = merged[INTERNAL_CNPJ].astype(str).nunique()
    return EnrollmentDivergenceKpis(
        granularity=granularity,
        match_exato=int(matched.sum()),
        total_divergencias=int(diverged.sum()),
        total_iss=total_iss,
        unidades_analise=n_est,
        total_estabelecimentos=n_est,
        media_filiais_por_raiz=None,
    )


def compute_enrollment_kpis(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None = None,
    censo_work: pd.DataFrame | None = None,
    granularity: Granularity = GRANULARITY_ESTABLISHMENT,
    reference_year: int | None = None,
    use_reference_month: bool = True,
    reference_month: int = 5,
    only_private_censo: bool = True,
    exclude_superior_puro: bool = True,
) -> EnrollmentDivergenceKpis:
    merged, _meta = merged_aggregate_internal(
        df,
        column_map,
        dms_work=dms_work,
        censo_work=censo_work,
        granularity=granularity,
        reference_year=reference_year,
        use_reference_month=use_reference_month,
        reference_month=reference_month,
        only_private_censo=only_private_censo,
        exclude_superior_puro=exclude_superior_puro,
    )
    return compute_enrollment_kpis_from_merged(merged, granularity=granularity)


def build_enrollment_divergence_table(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None = None,
    censo_work: pd.DataFrame | None = None,
    granularity: Granularity = GRANULARITY_ESTABLISHMENT,
    reference_year: int | None = None,
    use_reference_month: bool = True,
    reference_month: int = 5,
    only_private_censo: bool = True,
    exclude_superior_puro: bool = True,
) -> pd.DataFrame:
    merged, _meta = merged_aggregate_internal(
        df,
        column_map,
        dms_work=dms_work,
        censo_work=censo_work,
        granularity=granularity,
        reference_year=reference_year,
        use_reference_month=use_reference_month,
        reference_month=reference_month,
        only_private_censo=only_private_censo,
        exclude_superior_puro=exclude_superior_puro,
    )
    return _presentation_columns(merged, granularity=granularity)


def get_merged_aggregate_for_audits(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None = None,
    censo_work: pd.DataFrame | None = None,
    granularity: Granularity = GRANULARITY_ESTABLISHMENT,
    reference_year: int | None = None,
    use_reference_month: bool = True,
    reference_month: int = 5,
    only_private_censo: bool = True,
    exclude_superior_puro: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expõe o quadro agregado interno para exploratórios (ex.: QUANTIDADE) sem recalcular merges."""

    return merged_aggregate_internal(
        df,
        column_map,
        dms_work=dms_work,
        censo_work=censo_work,
        granularity=granularity,
        reference_year=reference_year,
        use_reference_month=use_reference_month,
        reference_month=reference_month,
        only_private_censo=only_private_censo,
        exclude_superior_puro=exclude_superior_puro,
    )


def describe_column_bindings(
    consolidado_df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None = None,
    censo_work: pd.DataFrame | None = None,
    granularity: Granularity = GRANULARITY_ESTABLISHMENT,
    reference_year: int | None = None,
    use_reference_month: bool = True,
    reference_month: int = 5,
    only_private_censo: bool = True,
    exclude_superior_puro: bool = True,
) -> dict[str, str | None]:
    cm = dict(column_map or {})
    paths_cons = resolved_paths_for_dashboard(consolidado_df, cm)
    qt_bas_phys = physical_qt_mat_bas_column(censo_work.columns) if isinstance(censo_work, pd.DataFrame) else None
    _, meta = merged_aggregate_internal(
        consolidado_df,
        cm,
        dms_work=dms_work,
        censo_work=censo_work,
        granularity=granularity,
        reference_year=reference_year,
        use_reference_month=use_reference_month,
        reference_month=reference_month,
        only_private_censo=only_private_censo,
        exclude_superior_puro=exclude_superior_puro,
    )

    out: dict[str, str | None] = {
        "aggregation_mode": str(meta.get("mode")),
        "granularity": str(meta.get("granularity") or granularity),
        "cnpj_dms_prefixed_consolidado": paths_cons.get("cnpj_dms"),
        "iss_prefixed_consolidado": paths_cons.get("iss"),
        "qt_mat_bas_physical_censo_work": qt_bas_phys,
        "censo_prefixed_QT_MAT_BAS_consolidado": _resolve_census_qt_bas_prefixed(consolidado_df),
        "aggregation_note": None,
    }

    note = meta.get("warn")
    extra = ""
    if granularity == GRANULARITY_ROOT and meta.get("mode", "").startswith("aggregated_frames"):
        extra = "Agregação **por raiz (8 dígitos)** conforme Etapa 8.2 — totais de `QT_MAT_BAS` e `QUANTIDADE` somados por grupo económico."
    elif meta.get("mode") == "aggregated_frames" and qt_bas_phys:
        extra = (
            "`Matrículas Censo`: soma de **QT_MAT_BAS** (`"
            + str(qt_bas_phys)
            + "`) por CNPJ. `Matrículas DMS`: **QUANTIDADE** no mês de referência (maio ou fallback) por contribuinte."
        )
    ref_m = meta.get("ref_month_meta") or {}
    if isinstance(ref_m, dict) and ref_m.get("coluna_competencia_nao_encontrada"):
        extra = (extra + " " if extra else "") + "Competência DMS não encontrada — soma anual legada."
    chunks = [str(s) for s in (note, extra) if s and str(s).strip()]
    out["aggregation_note"] = " ".join(chunks).strip() or None

    return out

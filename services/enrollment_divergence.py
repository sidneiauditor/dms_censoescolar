"""
Divergência operacional entre matrículas **Censo** e **DMS** (Etapa 8.1).

Compara **contribuintes (CNPJ 14 dígitos)** já agregados — ``SUM(QT_MAT_BAS)`` no município
vs ``SUM(QUANTIDADE)`` na DMS. Sem Streamlit — ver :mod:`ui.operacional_dashboard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from services.census_semantics import CENSUS_CANONICAL_FIELDS, physical_qt_mat_bas_column
from services.cnpj_aggregation import (
    AGG_ISS,
    AGG_MC,
    AGG_QTY,
    AGG_RAZAO,
    CNPJ_NORM_COL_CENSO,
    CNPJ_NORM_COL_DMS,
    INTERNAL_CNPJ,
    aggregate_census_by_cnpj,
    aggregate_dms_by_cnpj,
    merge_aggregates_by_cnpj,
)
from services.dashboard_metrics import resolved_paths_for_dashboard
from services.inferred_mapping import pick_column
from utils.cnpj import normalize_cnpj_digits

COL_CNPJ = "CNPJ"
COL_RAZAO = "Razão social"
COL_MC = "Matrículas Censo"
COL_MD = "Matrículas DMS"
COL_DIFF_ABS = "Diferença absoluta"
COL_DIFF_PCT = "Diferença percentual"

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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Quando não existem frames de trabalho DMS/Censo, reduz repetidos por contribuinte.

    ``QUANTIDADE`` fiscal soma‑se por CNPJ; ``censo__QT_MAT_BAS`` repetido pelo merge
    linha‑a‑linha usa **max** por chave fiscal (anti‑inflação). Ver :func:`merged_aggregate_internal`.
    """

    paths = resolved_paths_for_dashboard(consolidado_df, column_map or {})
    cnpj_col = paths.get("cnpj_dms")
    qty_dm_col = paths.get("matriculas")
    raz_col = paths.get("razao")
    iss_col = paths.get("iss")
    censo_bas_pref = _resolve_census_qt_bas_prefixed(consolidado_df)

    if consolidado_df.empty or not cnpj_col or cnpj_col not in consolidado_df.columns:
        msg = {"mode": "fallback_consolidado", "warn": "Coluna `dms__` CNPJ não resolvida no consolidado."}
        return pd.DataFrame(columns=[INTERNAL_CNPJ, AGG_QTY, AGG_ISS, AGG_MC, AGG_BASE, AGG_RAZAO]), msg

    norms = consolidado_df[cnpj_col].map(normalize_cnpj_digits)
    mask_ok = norms.notna()
    if not mask_ok.any():
        return pd.DataFrame(), {"mode": "fallback_consolidado", "warn": "CNPJ DMS ilegível no consolidado."}

    stub = consolidado_df.loc[mask_ok].copy()
    stub["_k"] = norms.loc[mask_ok].astype(str)

    qty_s = pd.to_numeric(stub[qty_dm_col], errors="coerce").fillna(0.0) if qty_dm_col and qty_dm_col in stub.columns else pd.Series(0.0, index=stub.index)
    iss_s = pd.to_numeric(stub[iss_col], errors="coerce").fillna(0.0) if iss_col and iss_col in stub.columns else pd.Series(0.0, index=stub.index)
    cen_bas = pd.to_numeric(stub[censo_bas_pref], errors="coerce") if censo_bas_pref else pd.Series(np.nan, index=stub.index)

    def _raz_first_nonempty(ser: pd.Series) -> str:
        for x in ser:
            t = str(x).strip()
            if t and t.lower() != "nan":
                return t
        return ""

    grp = stub.assign(_qty=qty_s, _iss=iss_s, _cen=cen_bas).groupby("_k", dropna=False, sort=False)
    qty_sum = grp["_qty"].sum()
    iss_sum = grp["_iss"].sum()
    cen_red = grp["_cen"].max()
    if raz_col and raz_col in stub.columns:
        raz_g = grp[raz_col].apply(_raz_first_nonempty).reindex(qty_sum.index).fillna("")
    else:
        raz_g = pd.Series("", index=qty_sum.index)

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
    warn_txt = []
    warn_txt.append(
        "Fluxo apenas com consolidado: `Matrículas Censo` = max por CNPJ de `censo__QT_MAT_BAS`."
        if censo_bas_pref
        else "Sem `censo__QT_MAT_BAS` no consolidado; matrículas municipais ficam vazias no fallback."
    )
    warn_txt.append("`Matrículas DMS`/`ISS` vindos dos prefixos `dms__` do consolidado (soma fiscal por CNPJ).")
    meta = {"mode": "fallback_consolidado", "warn": " ".join(warn_txt)}
    return agg, meta


def merged_aggregate_internal(
    consolidado_df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None,
    censo_work: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cm = dict(column_map or {})

    if (
        isinstance(dms_work, pd.DataFrame)
        and isinstance(censo_work, pd.DataFrame)
        and not dms_work.empty
        and not censo_work.empty
        and CNPJ_NORM_COL_DMS in dms_work.columns
        and CNPJ_NORM_COL_CENSO in censo_work.columns
    ):
        co_ent = _resolve_census_co_entidade(censo_work)
        c_agg = aggregate_census_by_cnpj(censo_work, co_entidade_column=co_ent)
        d_agg = aggregate_dms_by_cnpj(dms_work, column_map=cm)
        merged = merge_aggregates_by_cnpj(d_agg, c_agg, how="outer")
        qt_bas_phys = physical_qt_mat_bas_column(censo_work.columns)
        return merged, {
            "mode": "aggregated_frames",
            "qt_mat_bas_column_physical": qt_bas_phys,
            "co_entidade_column_physical": co_ent,
        }

    fb, meta = _fallback_merged_from_consolidado(consolidado_df, cm)
    return fb, meta


def _presentation_columns(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty or INTERNAL_CNPJ not in merged.columns:
        return pd.DataFrame(columns=[COL_CNPJ, COL_RAZAO, COL_MC, COL_MD, COL_DIFF_ABS, COL_DIFF_PCT])

    censo = pd.to_numeric(merged[AGG_MC], errors="coerce")
    dms = pd.to_numeric(merged[AGG_QTY], errors="coerce")
    raz = merged.get(AGG_RAZAO, pd.Series("", index=merged.index)).fillna("").astype(str)

    diff_abs = censo - dms

    cen_nv = censo.to_numpy(dtype=float)
    den_ok = np.isfinite(cen_nv) & (cen_nv != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_pct = (cen_nv - dms.to_numpy(dtype=float)) / cen_nv * 100.0
    diff_pct = np.where(den_ok, raw_pct, np.nan)

    work = pd.DataFrame(
        {
            COL_CNPJ: merged[INTERNAL_CNPJ].astype(str).values,
            COL_RAZAO: raz.values,
            COL_MC: censo.values,
            COL_MD: dms.values,
            COL_DIFF_ABS: diff_abs.values,
            COL_DIFF_PCT: diff_pct,
        }
    )

    sort_key = work[COL_DIFF_PCT].abs()
    return (
        work.assign(_sort=sort_key)
        .sort_values("_sort", ascending=False, na_position="last")
        .drop(columns="_sort")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class EnrollmentDivergenceKpis:
    total_escolas: int
    """Contribuintes (CNPJ 14 dígitos distintos) na vista agregada."""

    match_exato: int
    """Contribuintes com totais municipal e fiscal coincidentes."""

    total_divergencias: int
    """Contribuintes com ambos os totais válidos numericamente mas diferentes."""

    total_iss: float
    """Soma dos ISS já agregados por CNPJ."""


def compute_enrollment_kpis_from_merged(merged: pd.DataFrame) -> EnrollmentDivergenceKpis:
    if merged.empty or INTERNAL_CNPJ not in merged.columns:
        return EnrollmentDivergenceKpis(0, 0, 0, 0.0)

    n_cnpjs = int(merged[INTERNAL_CNPJ].astype(str).nunique())

    censo = pd.to_numeric(merged[AGG_MC], errors="coerce")
    dms = pd.to_numeric(merged[AGG_QTY], errors="coerce")
    comparable = censo.notna() & dms.notna()

    matched = comparable & censo.eq(dms)
    diverged = comparable & ~censo.eq(dms)

    total_iss = float(pd.to_numeric(merged[AGG_ISS], errors="coerce").fillna(0).sum()) if AGG_ISS in merged.columns else 0.0

    return EnrollmentDivergenceKpis(
        total_escolas=n_cnpjs,
        match_exato=int(matched.sum()),
        total_divergencias=int(diverged.sum()),
        total_iss=total_iss,
    )


def compute_enrollment_kpis(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None = None,
    censo_work: pd.DataFrame | None = None,
) -> EnrollmentDivergenceKpis:
    merged, _meta = merged_aggregate_internal(df, column_map, dms_work=dms_work, censo_work=censo_work)
    return compute_enrollment_kpis_from_merged(merged)


def build_enrollment_divergence_table(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None = None,
    censo_work: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged, _meta = merged_aggregate_internal(df, column_map, dms_work=dms_work, censo_work=censo_work)
    return _presentation_columns(merged)


def describe_column_bindings(
    consolidado_df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    dms_work: pd.DataFrame | None = None,
    censo_work: pd.DataFrame | None = None,
) -> dict[str, str | None]:
    cm = dict(column_map or {})
    paths_cons = resolved_paths_for_dashboard(consolidado_df, cm)
    qt_bas_phys = physical_qt_mat_bas_column(censo_work.columns) if isinstance(censo_work, pd.DataFrame) else None
    _, meta = merged_aggregate_internal(consolidado_df, cm, dms_work=dms_work, censo_work=censo_work)

    out: dict[str, str | None] = {
        "aggregation_mode": str(meta.get("mode")),
        "cnpj_dms_prefixed_consolidado": paths_cons.get("cnpj_dms"),
        "iss_prefixed_consolidado": paths_cons.get("iss"),
        "qt_mat_bas_physical_censo_work": qt_bas_phys,
        "censo_prefixed_QT_MAT_BAS_consolidado": _resolve_census_qt_bas_prefixed(consolidado_df),
        "aggregation_note": None,
    }

    note = meta.get("warn")
    extra = ""
    if meta.get("mode") == "aggregated_frames" and qt_bas_phys:
        extra = (
            "`Matrículas Censo`: soma de **QT_MAT_BAS** (`"
            + str(qt_bas_phys)
            + "`) por CNPJ. `Matrículas DMS`: soma **QUANTIDADE** por contribuinte; **ISS**: soma de **VLIMPOSTO** já agregada."
        )
    chunks = [str(s) for s in (note, extra) if s and str(s).strip()]
    out["aggregation_note"] = " ".join(chunks).strip() or None

    return out
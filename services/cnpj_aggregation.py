"""
Agregação **por CNPJ** (Etapa 8.1) para alinhar granularidade fiscal (DMS) e pedagógica (Censo).

A linha-analítica do painel operacional compara apenas totais já agregados:

- ``SUM(DMS.QUANTIDADE)`` (e somas de ISS/base) por contribuinte;
- ``SUM(Censo.QT_MAT_BAS)`` por CNPJ (total oficial de Educação Básica no extracto municipal).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from services.census_semantics import physical_qt_mat_bas_column
from services.indicators import COL_BASE_CALC_ALIASES, COL_ISS_ALIASES
from services.inferred_mapping import propose_dms_mapping

LOG = logging.getLogger(__name__)

CNPJ_NORM_COL_CENSO = "__cnpj_norm_censo"
CNPJ_NORM_COL_DMS = "__cnpj_norm_dms"

INTERNAL_CNPJ = "_cnpj_norm14"
AGG_QTY = "_agg_quantidade"
AGG_ISS = "_agg_vlimposto"
AGG_BASE = "_agg_vlbase"
AGG_MC = "_agg_mat_censo_bas"
AGG_RAZAO = "_agg_razao"
AGG_N_ESCOLAS_CENSO = "_agg_n_escolas_censo"


def _valid_norm_mask(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip()
    return t.str.len().eq(14) & t.str.isdigit()


def _resolve_dms_quantity_column(dms_df: pd.DataFrame, column_map: dict[str, Any]) -> str | None:
    cm = column_map or {}
    pick = cm.get("dms_qtd")
    if isinstance(pick, str) and pick.strip() and pick in dms_df.columns:
        return pick
    prop = propose_dms_mapping([str(c) for c in dms_df.columns]).get("quantidade")
    if prop and prop in dms_df.columns:
        return prop
    return None


def _resolve_first_alias(dms_df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    cols_upper = {str(c).strip().upper(): str(c) for c in dms_df.columns}
    for a in aliases:
        u = a.strip().upper()
        if u in cols_upper:
            return cols_upper[u]
    return None


def aggregate_census_by_cnpj(
    censo_df: pd.DataFrame,
    *,
    col_cnpj_norm: str = CNPJ_NORM_COL_CENSO,
    co_entidade_column: str | None = None,
) -> pd.DataFrame:
    """
    Uma linha por CNPJ normalizado (14 dígitos): soma **apenas** ``QT_MAT_BAS``.

    ``co_entidade_column`` se fornecido permite contar escolas distintas vinculadas ao CNPJ.
    """

    if censo_df.empty or col_cnpj_norm not in censo_df.columns:
        return pd.DataFrame(
            columns=[INTERNAL_CNPJ, AGG_MC, AGG_N_ESCOLAS_CENSO],
        )

    qt_col = physical_qt_mat_bas_column(censo_df.columns)
    if not qt_col or qt_col not in censo_df.columns:
        LOG.warning(
            "aggregate_census_by_cnpj: coluna %s ausente no Censo — agregado vazio.",
            physical_qt_mat_bas_column.__name__,
        )
        return pd.DataFrame(columns=[INTERNAL_CNPJ, AGG_MC, AGG_N_ESCOLAS_CENSO])

    work = censo_df.loc[_valid_norm_mask(censo_df[col_cnpj_norm])].copy()
    if work.empty:
        return pd.DataFrame(columns=[INTERNAL_CNPJ, AGG_MC, AGG_N_ESCOLAS_CENSO])

    work[INTERNAL_CNPJ] = work[col_cnpj_norm].astype(str).str.strip()
    work["_qt_num"] = pd.to_numeric(work[qt_col], errors="coerce").fillna(0.0)

    co_col = None
    if co_entidade_column and co_entidade_column in work.columns:
        co_col = co_entidade_column

    g = work.groupby(INTERNAL_CNPJ, dropna=False, sort=False)
    agg_series = g["_qt_num"].sum()
    if co_col:
        n_esc = g[co_col].nunique()
    else:
        n_esc = g.size()

    out = pd.DataFrame(
        {
            INTERNAL_CNPJ: agg_series.index.astype(str),
            AGG_MC: agg_series.to_numpy(dtype=float),
            AGG_N_ESCOLAS_CENSO: n_esc.reindex(agg_series.index).to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def aggregate_dms_by_cnpj(
    dms_df: pd.DataFrame,
    *,
    col_cnpj_norm: str = CNPJ_NORM_COL_DMS,
    column_map: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Uma linha por CNPJ: soma ``QUANTIDADE``, ``VLIMPOSTO``, ``VLBASECALCULO`` (aliases em :mod:`services.indicators`).
    Inclui rótulo de razão social (primeiro valor não vazio por grupo).
    """

    if dms_df.empty or col_cnpj_norm not in dms_df.columns:
        return pd.DataFrame(
            columns=[INTERNAL_CNPJ, AGG_QTY, AGG_ISS, AGG_BASE, AGG_RAZAO],
        )

    qty_col = _resolve_dms_quantity_column(dms_df, dict(column_map or {}))
    iss_col = _resolve_first_alias(dms_df, COL_ISS_ALIASES)
    base_col = _resolve_first_alias(dms_df, COL_BASE_CALC_ALIASES)
    dm_prop = propose_dms_mapping([str(c) for c in dms_df.columns])
    raz_phys = dm_prop.get("razao_social")
    raz_col = None
    if raz_phys and raz_phys in dms_df.columns:
        raz_col = raz_phys

    work = dms_df.loc[_valid_norm_mask(dms_df[col_cnpj_norm])].copy()
    if work.empty:
        return pd.DataFrame(columns=[INTERNAL_CNPJ, AGG_QTY, AGG_ISS, AGG_BASE, AGG_RAZAO])

    work[INTERNAL_CNPJ] = work[col_cnpj_norm].astype(str).str.strip()

    qty_num = pd.to_numeric(work[qty_col], errors="coerce") if qty_col else pd.Series(np.nan, index=work.index)
    iss_num = pd.to_numeric(work[iss_col], errors="coerce") if iss_col else pd.Series(0.0, index=work.index)
    base_num = pd.to_numeric(work[base_col], errors="coerce") if base_col else pd.Series(0.0, index=work.index)

    work["_qty"] = qty_num.fillna(0.0)
    work["_iss"] = iss_num.fillna(0.0)
    work["_base"] = base_num.fillna(0.0)

    gb = work.groupby(INTERNAL_CNPJ, dropna=False, sort=False)
    agg_qty = gb["_qty"].sum()
    agg_iss = gb["_iss"].sum()
    agg_base = gb["_base"].sum()

    raz_series: pd.Series
    if raz_col:
        def _first_nonempty(ser: pd.Series) -> str:
            for x in ser:
                t = str(x).strip()
                if t and t.lower() != "nan":
                    return t
            return ""

        raz_series = gb[raz_col].apply(_first_nonempty)
        raz_series = raz_series.reindex(agg_qty.index).fillna("")
    else:
        raz_series = pd.Series("", index=agg_qty.index, dtype=str)

    out = pd.DataFrame(
        {
            INTERNAL_CNPJ: agg_qty.index.astype(str),
            AGG_QTY: agg_qty.to_numpy(dtype=float),
            AGG_ISS: agg_iss.reindex(agg_qty.index).to_numpy(dtype=float),
            AGG_BASE: agg_base.reindex(agg_qty.index).to_numpy(dtype=float),
            AGG_RAZAO: raz_series.to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def merge_aggregates_by_cnpj(
    dms_agg: pd.DataFrame,
    censo_agg: pd.DataFrame,
    *,
    how: str = "outer",
) -> pd.DataFrame:
    """Junta agregados DMS e Censo pela chave interna de 14 dígitos."""

    empty_out_cols = [
        INTERNAL_CNPJ,
        AGG_RAZAO,
        AGG_QTY,
        AGG_MC,
        AGG_ISS,
        AGG_BASE,
        AGG_N_ESCOLAS_CENSO,
    ]

    if dms_agg.empty and censo_agg.empty:
        return pd.DataFrame(columns=empty_out_cols)

    left = dms_agg if not dms_agg.empty else pd.DataFrame(columns=[INTERNAL_CNPJ])
    right = censo_agg if not censo_agg.empty else pd.DataFrame(columns=[INTERNAL_CNPJ])

    merged = pd.merge(left, right, on=INTERNAL_CNPJ, how=how, suffixes=("", "_dupdrop"))
    drop_dup = [c for c in merged.columns if c.endswith("_dupdrop")]
    if drop_dup:
        merged = merged.drop(columns=drop_dup)

    for col in empty_out_cols:
        if col not in merged.columns:
            merged[col] = np.nan if col != AGG_RAZAO else ""

    merged[AGG_RAZAO] = merged[AGG_RAZAO].fillna("").astype(str)
    merged[AGG_QTY] = pd.to_numeric(merged[AGG_QTY], errors="coerce").fillna(0.0)
    merged[AGG_MC] = pd.to_numeric(merged[AGG_MC], errors="coerce")
    merged[AGG_ISS] = pd.to_numeric(merged[AGG_ISS], errors="coerce").fillna(0.0)
    merged[AGG_BASE] = pd.to_numeric(merged[AGG_BASE], errors="coerce").fillna(0.0)
    merged[AGG_N_ESCOLAS_CENSO] = pd.to_numeric(merged[AGG_N_ESCOLAS_CENSO], errors="coerce")
    return merged

"""
Agregação **por raiz do CNPJ** (primeiros **8 dígitos**) — Etapa 8.2.

Útil quando vários **estabelecimentos** (CNPJ 14) pertencem à mesma matriz económica /
rede municipal: soma tributárias e pedagógicas ficam distribuídas por **filiais** com CNPJs distintos.

Chave oficial ``__cnpj_raiz`` = ``normalize_cnpj_14[:8]`` (só válido quando a normalização
produziu 14 dígitos).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from services.census_semantics import physical_qt_mat_bas_column
from services.cnpj_aggregation import (
    AGG_BASE,
    AGG_ISS,
    AGG_MC,
    AGG_QTY,
    AGG_RAZAO,
    AGG_N_ESCOLAS_CENSO,
    CNPJ_NORM_COL_CENSO,
    CNPJ_NORM_COL_DMS,
    INTERNAL_CNPJ,
    filter_censo_for_fiscal_panel,
    _resolve_dms_quantity_column,
    _resolve_first_alias,
    _valid_norm_mask,
)
from services.indicators import COL_BASE_CALC_ALIASES, COL_ISS_ALIASES
from services.inferred_mapping import propose_dms_mapping

LOG = logging.getLogger(__name__)

CNPJ_RAIZ_COL = "__cnpj_raiz"

AGG_N_ESTAB_DMS = "_agg_n_estabelecimentos_dms"
AGG_N_ESTAB_CENSO = "_agg_n_estabelecimentos_censo"
AGG_RAZAO_VARIANTES = "_agg_razao_variantes"


def dataframe_with_cnpj_raiz(df: pd.DataFrame, col_norm: str) -> pd.DataFrame:
    """Anexa ``__cnpj_raiz`` (8 dígitos) a partir de ``__cnpj_norm_*`` com 14 dígitos."""

    out = df.copy()
    if col_norm not in out.columns:
        out[CNPJ_RAIZ_COL] = ""
        return out
    s = out[col_norm].astype(str).str.strip()
    ok = s.str.len().eq(14) & s.str.isdigit()
    out[CNPJ_RAIZ_COL] = ""
    out.loc[ok, CNPJ_RAIZ_COL] = s.loc[ok].str.slice(0, 8)
    return out


def _normalize_razao_token(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _principal_e_variantes_razao(ser: pd.Series) -> tuple[str, str]:
    """
    Razão principal = forma mais **frequente** (após normalização leve de espaços/caixa);
    ``variantes`` = demais textos distintos (representante original), separados por `` | ``.
    """

    raw = [str(x).strip() for x in ser.tolist() if str(x).strip() and str(x).lower() not in {"nan", "none", "<na>"}]
    if not raw:
        return "", ""

    norm_to_canonical: dict[str, str] = {}
    for r in raw:
        k = _normalize_razao_token(r)
        if k not in norm_to_canonical:
            norm_to_canonical[k] = r

    keys = [_normalize_razao_token(r) for r in raw]
    cnt = Counter(keys)
    top_key, _ = cnt.most_common(1)[0]
    principal = norm_to_canonical.get(top_key, "")
    outros_k = sorted({k for k in norm_to_canonical.keys() if k != top_key})
    variantes = " | ".join(norm_to_canonical[k] for k in outros_k)
    return principal, variantes


def aggregate_census_by_cnpj_root(
    censo_df: pd.DataFrame,
    *,
    col_cnpj_norm: str = CNPJ_NORM_COL_CENSO,
    co_entidade_column: str | None = None,
    only_private: bool = False,
    exclude_superior_puro: bool = False,
) -> pd.DataFrame:
    """
    Por raiz: soma ``QT_MAT_BAS``, conta escolas (`CO_ENTIDADE` ínico) e CNPJs 14 distintos.
    """

    if censo_df.empty or col_cnpj_norm not in censo_df.columns:
        return pd.DataFrame(columns=[CNPJ_RAIZ_COL, AGG_MC, AGG_N_ESCOLAS_CENSO, AGG_N_ESTAB_CENSO])

    if only_private or exclude_superior_puro:
        censo_df, _ = filter_censo_for_fiscal_panel(
            censo_df,
            only_private=only_private,
            exclude_superior_puro=exclude_superior_puro,
        )

    qt_col = physical_qt_mat_bas_column(censo_df.columns)
    if not qt_col or qt_col not in censo_df.columns:
        LOG.warning("aggregate_census_by_cnpj_root: sem QT_MAT_BAS físico.")
        return pd.DataFrame(columns=[CNPJ_RAIZ_COL, AGG_MC, AGG_N_ESCOLAS_CENSO, AGG_N_ESTAB_CENSO])

    work = censo_df.loc[_valid_norm_mask(censo_df[col_cnpj_norm])].copy()
    if work.empty:
        return pd.DataFrame(columns=[CNPJ_RAIZ_COL, AGG_MC, AGG_N_ESCOLAS_CENSO, AGG_N_ESTAB_CENSO])

    work[INTERNAL_CNPJ] = work[col_cnpj_norm].astype(str).str.strip()
    work[CNPJ_RAIZ_COL] = work[INTERNAL_CNPJ].str.slice(0, 8)
    work["_qt"] = pd.to_numeric(work[qt_col], errors="coerce").fillna(0.0)

    co_col = co_entidade_column if co_entidade_column and co_entidade_column in work.columns else None

    g = work.groupby(CNPJ_RAIZ_COL, dropna=False, sort=False)
    mat_sum = g["_qt"].sum()
    n_esc = g[co_col].nunique() if co_col else g.size()
    n_cnpj = g[INTERNAL_CNPJ].nunique()

    return pd.DataFrame(
        {
            CNPJ_RAIZ_COL: mat_sum.index.astype(str),
            AGG_MC: mat_sum.to_numpy(dtype=float),
            AGG_N_ESCOLAS_CENSO: n_esc.reindex(mat_sum.index).to_numpy(),
            AGG_N_ESTAB_CENSO: n_cnpj.reindex(mat_sum.index).to_numpy(),
        }
    ).reset_index(drop=True)


def aggregate_dms_by_cnpj_root(
    dms_df: pd.DataFrame,
    *,
    col_cnpj_norm: str = CNPJ_NORM_COL_DMS,
    column_map: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Por raiz: soma QUANTIDADE / ISS / base, nº de CNPJ 14 distintos, razão principal + variantes."""

    if dms_df.empty or col_cnpj_norm not in dms_df.columns:
        return pd.DataFrame(
            columns=[
                CNPJ_RAIZ_COL,
                AGG_QTY,
                AGG_ISS,
                AGG_BASE,
                AGG_RAZAO,
                AGG_RAZAO_VARIANTES,
                AGG_N_ESTAB_DMS,
            ]
        )

    qty_col = _resolve_dms_quantity_column(dms_df, dict(column_map or {}))
    iss_col = _resolve_first_alias(dms_df, COL_ISS_ALIASES)
    base_col = _resolve_first_alias(dms_df, COL_BASE_CALC_ALIASES)
    dm_prop = propose_dms_mapping([str(c) for c in dms_df.columns])
    raz_phys = dm_prop.get("razao_social")
    raz_col = raz_phys if raz_phys and raz_phys in dms_df.columns else None

    work = dms_df.loc[_valid_norm_mask(dms_df[col_cnpj_norm])].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                CNPJ_RAIZ_COL,
                AGG_QTY,
                AGG_ISS,
                AGG_BASE,
                AGG_RAZAO,
                AGG_RAZAO_VARIANTES,
                AGG_N_ESTAB_DMS,
            ]
        )

    work[INTERNAL_CNPJ] = work[col_cnpj_norm].astype(str).str.strip()
    work[CNPJ_RAIZ_COL] = work[INTERNAL_CNPJ].str.slice(0, 8)

    work["_qty"] = (
        pd.to_numeric(work[qty_col], errors="coerce").fillna(0.0) if qty_col else pd.Series(0.0, index=work.index)
    )
    work["_iss"] = pd.to_numeric(work[iss_col], errors="coerce").fillna(0.0) if iss_col else pd.Series(0.0, index=work.index)
    work["_base"] = pd.to_numeric(work[base_col], errors="coerce").fillna(0.0) if base_col else pd.Series(0.0, index=work.index)

    gb = work.groupby(CNPJ_RAIZ_COL, dropna=False, sort=False)
    agg_qty = gb["_qty"].sum()
    agg_iss = gb["_iss"].sum()
    agg_base = gb["_base"].sum()
    n_est = gb[INTERNAL_CNPJ].nunique()

    raz_principal_list: list[str] = []
    raz_var_list: list[str] = []
    if raz_col:
        for root_key in agg_qty.index.astype(str):
            sub = work.loc[work[CNPJ_RAIZ_COL].astype(str) == root_key, raz_col]
            p, v = _principal_e_variantes_razao(sub)
            raz_principal_list.append(p)
            raz_var_list.append(v)
        raz_principal = pd.Series(raz_principal_list, index=agg_qty.index)
        raz_var = pd.Series(raz_var_list, index=agg_qty.index)
    else:
        raz_principal = pd.Series("", index=agg_qty.index)
        raz_var = pd.Series("", index=agg_qty.index)

    return pd.DataFrame(
        {
            CNPJ_RAIZ_COL: agg_qty.index.astype(str),
            AGG_QTY: agg_qty.to_numpy(dtype=float),
            AGG_ISS: agg_iss.reindex(agg_qty.index).to_numpy(dtype=float),
            AGG_BASE: agg_base.reindex(agg_qty.index).to_numpy(dtype=float),
            AGG_RAZAO: raz_principal.to_numpy(),
            AGG_RAZAO_VARIANTES: raz_var.to_numpy(),
            AGG_N_ESTAB_DMS: n_est.reindex(agg_qty.index).to_numpy(dtype=int),
        }
    ).reset_index(drop=True)


def merge_root_aggregates(
    dms_root: pd.DataFrame,
    censo_root: pd.DataFrame,
    *,
    how: str = "outer",
) -> pd.DataFrame:
    cols = [
        CNPJ_RAIZ_COL,
        AGG_RAZAO,
        AGG_RAZAO_VARIANTES,
        AGG_QTY,
        AGG_MC,
        AGG_ISS,
        AGG_BASE,
        AGG_N_ESCOLAS_CENSO,
        AGG_N_ESTAB_DMS,
        AGG_N_ESTAB_CENSO,
    ]

    if dms_root.empty and censo_root.empty:
        return pd.DataFrame(columns=cols)

    left = dms_root if not dms_root.empty else pd.DataFrame(columns=[CNPJ_RAIZ_COL])
    right = censo_root if not censo_root.empty else pd.DataFrame(columns=[CNPJ_RAIZ_COL])

    merged = pd.merge(left, right, on=CNPJ_RAIZ_COL, how=how, suffixes=("", "_dupdrop"))
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_dupdrop")], errors="ignore")

    for c in cols:
        if c not in merged.columns:
            merged[c] = np.nan if c not in (AGG_RAZAO, AGG_RAZAO_VARIANTES) else ""

    merged[AGG_RAZAO] = merged[AGG_RAZAO].fillna("").astype(str)
    merged[AGG_RAZAO_VARIANTES] = merged[AGG_RAZAO_VARIANTES].fillna("").astype(str)
    merged[AGG_QTY] = pd.to_numeric(merged[AGG_QTY], errors="coerce").fillna(0.0)
    merged[AGG_MC] = pd.to_numeric(merged[AGG_MC], errors="coerce")
    merged[AGG_ISS] = pd.to_numeric(merged[AGG_ISS], errors="coerce").fillna(0.0)
    merged[AGG_BASE] = pd.to_numeric(merged[AGG_BASE], errors="coerce").fillna(0.0)
    for c in (AGG_N_ESCOLAS_CENSO, AGG_N_ESTAB_DMS, AGG_N_ESTAB_CENSO):
        merged[c] = pd.to_numeric(merged.get(c), errors="coerce")
    return merged

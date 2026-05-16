"""
Exploração descritiva do campo **QUANTIDADE** na DMS (Etapa 8.2).

Sem juízos automáticos sobre o significado económico do campo — apenas estatísticas e
cruzes proporcionais com ``QT_MAT_BAS`` quando o comparativo por CNPJ ou raiz estiver disponível.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.cnpj_aggregation import (
    AGG_MC,
    AGG_QTY,
    CNPJ_NORM_COL_DMS,
    INTERNAL_CNPJ,
    _resolve_dms_quantity_column,
    _valid_norm_mask,
)


def build_quantidade_exploration(
    dms_df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    merged_by_cnpj14: pd.DataFrame | None = None,
    merged_by_root: pd.DataFrame | None = None,
    ref_month_meta: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame | None]:
    """
    Devolve tabelas prontas para ``st.dataframe`` — sem texto conclusivo.

    Chaves esperadas pela UI:

    - ``distribuicao_linha_dms`` — descritivos dos lançamentos (linha-a-linha) na base DMS;
    - ``top_altos``, ``top_baixos_nao_zero`` — extremos nos lançamentos;
    - ``por_cnpj14_proporcao`` — apenas se ``merged_by_cnpj14`` trouxer ``_agg_quantidade`` / ``_agg_mat_censo_bas``;
    - ``por_raiz_resumo`` — média e soma de QUANTIDADE / ``QT_MAT_BAS`` por raiz se ``merged_by_root`` existir.
    """

    out: dict[str, pd.DataFrame | None] = {
        "distribuicao_linha_dms": None,
        "top_altos": None,
        "top_baixos_nao_zero": None,
        "por_cnpj14_proporcao": None,
        "por_raiz_resumo": None,
        "referencia_mensal_meta": None,
    }

    if ref_month_meta:
        fb = ref_month_meta.get("fallback_detail")
        if isinstance(fb, pd.DataFrame) and not fb.empty:
            out["referencia_mensal_meta"] = fb.copy()
        elif ref_month_meta.get("coluna_competencia_nao_encontrada"):
            out["referencia_mensal_meta"] = pd.DataFrame(
                [{"aviso": "Coluna de competência não encontrada — exploração sobre todas as linhas DMS."}]
            )

    if dms_df.empty or CNPJ_NORM_COL_DMS not in dms_df.columns:
        return out

    qty_col = _resolve_dms_quantity_column(dms_df, dict(column_map or {}))
    if not qty_col or qty_col not in dms_df.columns:
        return out

    work = dms_df.loc[_valid_norm_mask(dms_df[CNPJ_NORM_COL_DMS])].copy()
    if work.empty:
        return out

    qty = pd.to_numeric(work[qty_col], errors="coerce")
    cnpj = work[CNPJ_NORM_COL_DMS].astype(str).str.strip()

    desc = qty.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame().T.rename(index={qty.name: "QUANTIDADE"})
    desc["non_null_linhas"] = int(qty.notna().sum())
    desc["zeros_linhas"] = int((qty.fillna(0) == 0).sum())
    out["distribuicao_linha_dms"] = desc.reset_index(drop=True)

    tbl = pd.DataFrame({"CNPJ_14_norm": cnpj.values, "QUANTIDADE": qty.values})
    top_n = min(15, len(tbl.index))
    if top_n:
        ranked = tbl.sort_values("QUANTIDADE", ascending=False).head(top_n)
        out["top_altos"] = ranked.reset_index(drop=True)
        nz = tbl.loc[tbl["QUANTIDADE"].notna() & (tbl["QUANTIDADE"] != 0)]
        if not nz.empty:
            out["top_baixos_nao_zero"] = nz.sort_values("QUANTIDADE", ascending=True).head(top_n).reset_index(drop=True)

    if (
        merged_by_cnpj14 is not None
        and not merged_by_cnpj14.empty
        and AGG_QTY in merged_by_cnpj14.columns
        and AGG_MC in merged_by_cnpj14.columns
    ):
        m = merged_by_cnpj14[[INTERNAL_CNPJ, AGG_QTY, AGG_MC]].copy()
        m["QUANTIDADE_agregada"] = pd.to_numeric(m[AGG_QTY], errors="coerce")
        m["QT_MAT_BAS_agregada"] = pd.to_numeric(m[AGG_MC], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            m["QUANTIDADE_sobre_QT_MAT_BAS"] = np.where(
                m["QT_MAT_BAS_agregada"].notna() & (m["QT_MAT_BAS_agregada"] > 0),
                m["QUANTIDADE_agregada"] / m["QT_MAT_BAS_agregada"],
                np.nan,
            )
        out["por_cnpj14_proporcao"] = m.drop(columns=[AGG_QTY, AGG_MC]).rename(
            columns={INTERNAL_CNPJ: "CNPJ_14_norm"}
        )

    if merged_by_root is not None and not merged_by_root.empty and AGG_QTY in merged_by_root.columns:
        from services.cnpj_root_aggregation import CNPJ_RAIZ_COL

        if CNPJ_RAIZ_COL in merged_by_root.columns and AGG_MC in merged_by_root.columns:
            r = merged_by_root[[CNPJ_RAIZ_COL, AGG_QTY, AGG_MC]].copy()
            r = r.rename(columns={CNPJ_RAIZ_COL: "raiz_CNPJ_8"})
            r["sum_QUANTIDADE"] = pd.to_numeric(r[AGG_QTY], errors="coerce")
            r["sum_QT_MAT_BAS"] = pd.to_numeric(r[AGG_MC], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                r["razao_QUANT_over_MAT_BAS"] = np.where(
                    r["sum_QT_MAT_BAS"].notna() & (r["sum_QT_MAT_BAS"] > 0),
                    r["sum_QUANTIDADE"] / r["sum_QT_MAT_BAS"],
                    np.nan,
                )
            out["por_raiz_resumo"] = r[
                ["raiz_CNPJ_8", "sum_QUANTIDADE", "sum_QT_MAT_BAS", "razao_QUANT_over_MAT_BAS"]
            ]

    return out

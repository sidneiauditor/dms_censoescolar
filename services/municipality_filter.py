"""
Filtragem municipal na tabela **bruta** de Escola antes do merge (volume menor).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from services.census_consolidator import normalize_co_entidade

LOG = logging.getLogger(__name__)


def _sanitize_code(value: Any) -> str:
    """Remove artefactos típicos CSV/Excel."""

    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0") and text.replace(".0", "").isdigit():
        text = text[:-2]
    return text


def filter_escola_by_municipality(
    df_escola: pd.DataFrame,
    phys_map_escola: dict[str, str],
    uf_escolha: str,
    municipio_codigo: str | None,
) -> tuple[pd.DataFrame, dict[str, str | int]]:
    """
    Mantém apenas linhas onde ``SG_UF`` e ``CO_MUNICIPIO`` coincidem com a escolha.

    Returns
    -------
    dataframe filtrado, estatísticas (linhas antes/depois)
    """

    n0 = len(df_escola.index)
    stats: dict[str, str | int] = {"antes": n0, "depois": n0}

    sg = phys_map_escola.get("SG_UF")
    co_mun = phys_map_escola.get("CO_MUNICIPIO")
    if sg is None or co_mun is None:
        stats["motivo"] = "Colunas físicas SG_UF/CO_MUNICIPIO não mapeadas — filtro ignorado."
        LOG.warning(stats["motivo"])
        return df_escola.copy(), stats

    if sg not in df_escola.columns or co_mun not in df_escola.columns:
        stats["motivo"] = "Colunas de localização não existem neste ficheiro."
        LOG.warning(stats["motivo"])
        return df_escola.copy(), stats

    mask = (
        df_escola[sg]
        .astype(str)
        .str.strip()
        .str.upper()
        == uf_escolha.strip().upper()
    )
    if municipio_codigo:
        mun_series = df_escola[co_mun].map(_sanitize_code)
        mun_norm = mun_series.astype(str).str.strip()
        code_target = municipio_codigo.strip()
        mask &= mun_norm == code_target

    out = df_escola.loc[mask].copy()
    stats["depois"] = len(out.index)
    stats["motivo"] = ""
    LOG.info(
        "Filtro municipal: UF=%s código_mun=%s | %s → %s linhas",
        uf_escolha,
        municipio_codigo,
        n0,
        stats["depois"],
    )
    return out, stats


def restrict_matricula_to_entidades(
    df_mat: pd.DataFrame,
    phys_co_matricula: str,
    entidades: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Evita empilhar todas as linhas da matricula nacional quando já temos o recorte."""

    co_key = normalize_co_entidade(df_mat[phys_co_matricula])
    sub = df_mat.loc[co_key.isin(entidades)].copy()
    return sub, {"antes_mat": len(df_mat.index), "depois_mat": len(sub.index)}

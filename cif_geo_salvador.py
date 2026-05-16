"""
Contexto geográfico pré-definido: Salvador · BA · IBGE 2927408 (sem Streamlit).

Usado pelo modo Operacional quando o utilizador não escolhe manualmente UF/município.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _sanitize_codigo_ibge(cell: object) -> str:
    texto = "" if pd.isna(cell) else str(cell).strip()
    if texto.endswith(".0") and texto.replace(".0", "").isdigit():
        texto = texto[:-2]
    return texto


UF_SALVADOR_CONST = "BA"
IBGE_SALVADOR = "2927408"
ROTULO_SALVADOR_CURTO = "Salvador"


def construir_geo_context_salvador(
    df_escola: pd.DataFrame,
    map_geo: dict[str, str],
    exercise: int,
) -> dict[str, Any]:
    """
    Produz estrutura espelho de ``render_etapa0_contexto_municipal`` para filtro só em Salvador.

    Se UF/município não reconhecíveis no ficheiro, ``filtro_impossivel_geo`` fica verdadeiro.
    """

    out: dict[str, Any] = {
        "exercise": int(exercise),
        "uf": None,
        "mun_code": None,
        "mun_label": "",
        "skip_geo": False,
        "filtro_ativo": False,
        "filtro_impossivel_geo": False,
    }
    uf_phys = map_geo.get("SG_UF")
    co_phys = map_geo.get("CO_MUNICIPIO")
    no_phys = map_geo.get("NO_MUNICIPIO")

    if not uf_phys or not co_phys:
        out["filtro_impossivel_geo"] = True
        return out
    if uf_phys not in df_escola.columns or co_phys not in df_escola.columns:
        out["filtro_impossivel_geo"] = True
        return out

    sub_ba_mask = df_escola[uf_phys].astype(str).str.strip().str.upper() == UF_SALVADOR_CONST.upper()
    if not bool(sub_ba_mask.any()):
        out["filtro_impossivel_geo"] = True
        return out

    cod_series = df_escola.loc[sub_ba_mask, co_phys].map(_sanitize_codigo_ibge)
    if IBGE_SALVADOR not in set(cod_series.astype(str)):
        out["filtro_impossivel_geo"] = True
        return out

    nome_escola = ROTULO_SALVADOR_CURTO
    if no_phys and no_phys in df_escola.columns:
        pick = df_escola.loc[
            sub_ba_mask & (df_escola[co_phys].map(_sanitize_codigo_ibge) == IBGE_SALVADOR),
            no_phys,
        ]
        if not pick.empty and pd.notna(pick.iloc[0]):
            n = str(pick.iloc[0]).strip()
            if n:
                nome_escola = n

    out["uf"] = UF_SALVADOR_CONST
    out["mun_code"] = IBGE_SALVADOR
    out["mun_label"] = f"{nome_escola} — código IBGE {IBGE_SALVADOR}"
    out["filtro_ativo"] = True
    return out

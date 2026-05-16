"""
Indicadores económico‑educacionais sobre o consolidado DMS × Censo.

Arquitetura:
- Entrada: um ``DataFrame`` já pré‑fundido (prefixos ``dms__`` e ``censo__`` vindos da Etapa 3) e opcionalmente
  ``column_map`` da Etapa 2 para localizar a coluna física de **matrículas** no lado Censo.
- Descoberta de colunas fiscal/pedagógicas por **lista de aliases** estáveis (+ ``pick_column`` de
  ``inferred_mapping``), sem depender de ordem nem de nomes mágicos fora desses aliases.
- Cada indicador é ``numerador / denominador``, com denominador = matrículas do Censo alinhadas por linha
  ao registo fiscal DMS na mesma linha consolidada (pós‑merge determinístico / texto opcional).
- Pós‑processamento único (:func:`_safe_numeric_ratio`) garante comportamento definido ante ``NaN``,
  zero, e infinitos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from services.inferred_mapping import pick_column

LOG = logging.getLogger(__name__)

COL_ISS_ALIASES: tuple[str, ...] = (
    "VLIMPOSTO",
    "VL_IMPOSTO",
    "VAL_IMP",
    "VALOR_ISS",
    "VL_ISS",
)
COL_MENSALIDADE_ALIASES: tuple[str, ...] = (
    "VLMENSALIDADE",
    "VL_MENSALIDADE",
    "VALOR_MENSALIDADE",
    "VL_MENS",
    "MENSALIDADE",
)
COL_BASE_CALC_ALIASES: tuple[str, ...] = (
    "VLBASECALCULO",
    "VL_BASE_CALCULO",
    "VL_BASECALC",
    "BASE_CALCULO",
    "BCALCULO",
)
COL_MATRICULAS_ALIASES: tuple[str, ...] = (
    "QT_MAT_BAS",
    "matriculas",
    "MATRICULAS",
    "TOTAL_MATRICULAS",
    "QT_TOTAL_MATRICULAS",
)

COL_ISS_PM = "iss_por_matricula"
COL_MSG_PM = "mensalidade_por_aluno"
COL_BASE_PM = "base_calculo_por_aluno"


@dataclass(frozen=True)
class IndicatorSummaryStats:
    """Estatísticas descritivas (apenas valores finitos de cada indicador)."""

    nome: str
    media: float | None
    mediana: float | None
    maximo: float | None
    minimo: float | None
    n_validos: int


@dataclass
class FiscalBasicIndicatorReport:
    """Traço útil para UI e logs: colunas físicas efectivas usadas e métricas resumidas."""

    colunas_resolvidas: dict[str, str | None] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)
    resumos: list[IndicatorSummaryStats] = field(default_factory=list)


def normalize_key(name: str) -> str:
    """Chave simples upper/underscore para cruzar com nomes físicos no consolidado."""

    return str(name).strip().upper().replace(" ", "_")


def _lista_colunas_prefixed(df: pd.DataFrame, prefix: str) -> list[str]:
    p = prefix + "__"
    return [c for c in df.columns.map(str) if c.startswith(p)]


def _resolve_prefixed_physical(
    df: pd.DataFrame,
    *,
    prefix: str,
    candidates: tuple[str, ...],
    prefer_physical: str | None = None,
) -> str | None:
    prefixed = _lista_colunas_prefixed(df, prefix)
    phys_list = []
    pref_map: dict[str, str] = {}
    for pc in prefixed:
        phys = pc.split("__", 1)[1]
        phys_list.append(phys)
        pref_map[normalize_key(phys)] = pc

    if prefer_physical and isinstance(prefer_physical, str) and prefer_physical.strip():
        key = normalize_key(prefer_physical)
        if key in pref_map:
            return pref_map[key]

    raw_hit = pick_column(phys_list, candidates)
    if raw_hit:
        nk = normalize_key(raw_hit)
        return pref_map.get(nk)
    return None


def _safe_numeric_ratio(numerator: pd.Series, denominator: pd.Series, *, label: str) -> pd.Series:
    """Divide `numerador/denominador`; ignora denominador não positivo ou inválido; inf → NaN."""

    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        quot = num / den
    bad_den = den.isna() | (den <= 0)
    quot = quot.mask(bad_den, np.nan)
    quot = quot.replace([np.inf, -np.inf], np.nan)
    qv = quot.to_numpy(dtype="float64", copy=True)
    not_nan = quot.notna().to_numpy(dtype=bool)
    non_finite_ok = np.isfinite(qv) | ~not_nan
    if not np.all(non_finite_ok):
        LOG.warning("Indicador %s: valores não finitos residuais → NaN.", label)
        quot = pd.Series(np.where(not_nan & np.isfinite(qv), qv, np.nan), index=quot.index, dtype="float64")
    return quot


def summarize_series_stats(nome_indicador: str, serie: pd.Series) -> IndicatorSummaryStats:
    s = pd.to_numeric(serie, errors="coerce")
    vals = s.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(vals.index)
    if n == 0:
        return IndicatorSummaryStats(nome=nome_indicador, media=None, mediana=None, maximo=None, minimo=None, n_validos=0)
    arr = vals.to_numpy(dtype=float)
    return IndicatorSummaryStats(
        nome=nome_indicador,
        media=float(np.mean(arr)),
        mediana=float(np.median(arr)),
        maximo=float(np.max(arr)),
        minimo=float(np.min(arr)),
        n_validos=int(n),
    )


def add_basic_fiscal_indicators(
    consolidated: pd.DataFrame,
    column_map: dict[str, Any],
) -> tuple[pd.DataFrame, FiscalBasicIndicatorReport]:
    """
    Injeta nas linhas consolidadas os indicadores 6.1 (rótulos estáveis COL_ISS_PM etc.).

    Preserva o ``DataFrame`` original (cópia). Se faltar denominador ou numerador resolve,
    indicadores ficam NaN com aviso no relatório.
    """

    report = FiscalBasicIndicatorReport()
    out = consolidated.copy()

    sentinel = "-- Selecionar coluna --"
    censo_mat_physical = column_map.get("censo_mat")
    if censo_mat_physical == sentinel or (isinstance(censo_mat_physical, str) and not censo_mat_physical.strip()):
        censo_mat_physical = None

    col_matricula = _resolve_prefixed_physical(
        out,
        prefix="censo",
        candidates=COL_MATRICULAS_ALIASES,
        prefer_physical=censo_mat_physical if isinstance(censo_mat_physical, str) else None,
    )
    report.colunas_resolvidas["matriculas_denominador"] = col_matricula

    col_iss = _resolve_prefixed_physical(out, prefix="dms", candidates=COL_ISS_ALIASES)
    col_mens = _resolve_prefixed_physical(out, prefix="dms", candidates=COL_MENSALIDADE_ALIASES)
    col_base = _resolve_prefixed_physical(out, prefix="dms", candidates=COL_BASE_CALC_ALIASES)
    report.colunas_resolvidas["VLIMPOSTO_numerador"] = col_iss
    report.colunas_resolvidas["VLMENSALIDADE_numerador"] = col_mens
    report.colunas_resolvidas["VLBASECALCULO_numerador"] = col_base

    if col_matricula is None:
        report.avisos.append(
            "Não foi encontrada coluna de matrículas no consolidado (`censo__…`). Verifique merge e campo `matriculas` no Censo."
        )

    denom = out[col_matricula] if col_matricula and col_matricula in out.columns else pd.Series(np.nan, index=out.index)

    if col_iss:
        out[COL_ISS_PM] = _safe_numeric_ratio(out[col_iss], denom, label=COL_ISS_PM)
        report.resumos.append(summarize_series_stats(COL_ISS_PM, out[COL_ISS_PM]))
    else:
        report.avisos.append("Coluna VLIMPOSTO (ISS) não localizada sob prefixo dms__. Indicador com NaN.")
        out[COL_ISS_PM] = pd.Series(np.nan, index=out.index, dtype=float)
        report.resumos.append(summarize_series_stats(COL_ISS_PM, out[COL_ISS_PM]))

    if col_mens:
        out[COL_MSG_PM] = _safe_numeric_ratio(out[col_mens], denom, label=COL_MSG_PM)
        report.resumos.append(summarize_series_stats(COL_MSG_PM, out[COL_MSG_PM]))
    else:
        report.avisos.append("Coluna VLMENSALIDADE não localizada sob prefixo dms__. Indicador com NaN.")
        out[COL_MSG_PM] = pd.Series(np.nan, index=out.index, dtype=float)
        report.resumos.append(summarize_series_stats(COL_MSG_PM, out[COL_MSG_PM]))

    if col_base:
        out[COL_BASE_PM] = _safe_numeric_ratio(out[col_base], denom, label=COL_BASE_PM)
        report.resumos.append(summarize_series_stats(COL_BASE_PM, out[COL_BASE_PM]))
    else:
        report.avisos.append("Coluna VLBASECALCULO não localizada sob prefixo dms__. Indicador com NaN.")
        out[COL_BASE_PM] = pd.Series(np.nan, index=out.index, dtype=float)
        report.resumos.append(summarize_series_stats(COL_BASE_PM, out[COL_BASE_PM]))

    LOG.info(
        "Indicadores 6.1: matriculas=%s iss_src=%s mens_src=%s base_src=%s",
        col_matricula,
        col_iss,
        col_mens,
        col_base,
    )
    return out, report


def refresh_consolidado_with_indicators(df: pd.DataFrame, column_map: dict[str, Any]) -> tuple[pd.DataFrame, FiscalBasicIndicatorReport]:
    """Atalho explícito: cópia + indicadores — útil após merges parciais sem duplicar lógica na UI."""

    return add_basic_fiscal_indicators(df, column_map)

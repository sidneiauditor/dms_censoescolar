"""
Divergência operacional entre matrículas **Censo** e **DMS** no consolidado (pós-merge).

Este módulo concentra **apenas dados** — sem Streamlit:

- Descoberta de colunas físicas com prefixos ``censo__`` / ``dms__`` herdando os mesmos aliases
  usados no painel técnico.
- Métricas de linha: ``censo − dms``, percentual sobre o Censo.
- KPIs mínimos agregados (totais sobre o ``DataFrame`` completo ou subconjunto filtrado no futuro).

A camada UI chama apenas :func:`compute_enrollment_kpis` e :func:`build_enrollment_divergence_table`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from services.dashboard_metrics import iss_series, resolved_paths_for_dashboard
from services.indicators import COL_MATRICULAS_ALIASES

# Campos típicos de número de alunos só na D municipal (nem sempre igual ao vocabulário do Censo).
DMS_ALUNOS_ALIASES: tuple[str, ...] = (
    "QUANTIDADE",
    "QTDE",
    "QTD_ALUNOS",
    "QTD",
    "TOTAL",
    "NU_QUANTIDADE",
)

SELECT_SENTINEL = "-- Selecionar coluna --"


def normalize_key(name: str) -> str:
    return str(name).strip().upper().replace(" ", "_")


def _lista_colunas_prefixed(df: pd.DataFrame, prefix: str) -> list[str]:
    p = prefix + "__"
    return [c for c in df.columns.map(str) if c.startswith(p)]


def _resolve_prefixed(
    df: pd.DataFrame,
    *,
    prefix: str,
    candidates: tuple[str, ...],
    prefer_physical: str | None,
) -> str | None:
    prefixed = _lista_colunas_prefixed(df, prefix)
    pref_map: dict[str, str] = {}
    phys_list: list[str] = []
    for pc in prefixed:
        phys = pc.split("__", 1)[1]
        phys_list.append(phys)
        pref_map[normalize_key(phys)] = pc
    if prefer_physical and isinstance(prefer_physical, str) and prefer_physical.strip():
        pk = normalize_key(prefer_physical.strip())
        if pk in pref_map:
            return pref_map[pk]
    from services.inferred_mapping import pick_column

    hit = pick_column(phys_list, candidates)
    if hit:
        return pref_map.get(normalize_key(hit))
    return None


def _prefer_from_map(column_map: dict[str, Any], key: str) -> str | None:
    v = column_map.get(key)
    if not isinstance(v, str) or not v.strip() or v == SELECT_SENTINEL:
        return None
    return v.strip()


def resolve_matricula_censo_column(df: pd.DataFrame, column_map: dict[str, Any]) -> str | None:
    return _resolve_prefixed(
        df,
        prefix="censo",
        candidates=COL_MATRICULAS_ALIASES,
        prefer_physical=_prefer_from_map(column_map, "censo_mat"),
    )


def resolve_matricula_dms_column(df: pd.DataFrame, column_map: dict[str, Any]) -> str | None:
    """Matrícula / quantidade no lado fiscal (prefixo ``dms__``)."""

    candidates = tuple(dict.fromkeys([*COL_MATRICULAS_ALIASES, *DMS_ALUNOS_ALIASES]))
    return _resolve_prefixed(
        df,
        prefix="dms",
        candidates=candidates,
        prefer_physical=_prefer_from_map(column_map, "dms_qtd"),
    )


COL_CNPJ = "CNPJ"
COL_RAZAO = "Razão social"
COL_MC = "Matrículas Censo"
COL_MD = "Matrículas DMS"
COL_DIFF_ABS = "Diferença absoluta"
COL_DIFF_PCT = "Diferença percentual"


@dataclass(frozen=True)
class EnrollmentDivergenceKpis:
    total_escolas: int
    """CNPJs não vazios distintos nas linhas consolidadas."""

    match_exato: int
    """Linhas onde Censo e DMS são numéricos finitos e iguais (``mat_censo == mat_dms``)."""

    total_divergencias: int
    """Linhas comparáveis (ambos os lados numéricos finitos) onde ``mat_censo != mat_dms``."""

    total_iss: float
    """Soma de ISS (:func:`dashboard_metrics.iss_series`) lado ``dms__``."""


def _distinct_dms_escolas(df: pd.DataFrame, cnpj_col: str | None) -> int:
    if not cnpj_col or cnpj_col not in df.columns:
        return 0
    s = df[cnpj_col].astype(str).str.strip()
    return int(s[(s != "") & (s.str.casefold() != "nan")].nunique())


def compute_enrollment_kpis(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    matricula_censo_col: str | None = None,
    matricula_dms_col: str | None = None,
    iss_col: str | None = None,
) -> EnrollmentDivergenceKpis:
    """
    KPI apresentados no topo do modo operacional mínimo.

    ``match_exato`` e ``total_divergências`` só consideram linhas **comparáveis**
    (duas contagens podem converter-se para float finito); linhas só com uma base
    ficam no consolidado mas **não entram** nesses dois contadores.
    """

    if df.empty:
        return EnrollmentDivergenceKpis(0, 0, 0, 0.0)

    cm = dict(column_map or {})
    paths = resolved_paths_for_dashboard(df, cm)
    mc = matricula_censo_col if matricula_censo_col else resolve_matricula_censo_column(df, cm)
    md = matricula_dms_col if matricula_dms_col else resolve_matricula_dms_column(df, cm)
    icol = paths.get("iss") if iss_col is None else iss_col

    c_npj = paths.get("cnpj_dms")
    total_escolas = _distinct_dms_escolas(df, c_npj)

    if not mc or mc not in df.columns or not md or md not in df.columns:
        total_iss = float(iss_series(df, icol).fillna(0).sum())
        return EnrollmentDivergenceKpis(total_escolas, 0, 0, total_iss)

    censo = pd.to_numeric(df[mc], errors="coerce")
    dms = pd.to_numeric(df[md], errors="coerce")
    comparable = censo.notna() & dms.notna()

    matched = comparable & censo.eq(dms)
    diverged = comparable & ~censo.eq(dms)

    total_iss = float(iss_series(df, icol).fillna(0).sum())

    return EnrollmentDivergenceKpis(
        total_escolas=int(total_escolas),
        match_exato=int(matched.sum()),
        total_divergencias=int(diverged.sum()),
        total_iss=total_iss,
    )


def build_enrollment_divergence_table(
    df: pd.DataFrame,
    column_map: dict[str, Any],
) -> pd.DataFrame:
    """
    Linhas ordenadas por ``|dif. %|`` descendente (:func:`pandas.DataFrame.sort_values`).

    Percentual: ``((censo - dms) / censo) * 100``; censo ilegível ou ``censo == 0`` ⇒ ``NaN``.
    """

    cm = dict(column_map or {})
    paths = resolved_paths_for_dashboard(df, cm)
    col_cnpj = paths.get("cnpj_dms")
    col_razao = paths.get("razao")
    col_mc = resolve_matricula_censo_column(df, cm)
    col_md = resolve_matricula_dms_column(df, cm)

    if df.empty:
        return pd.DataFrame(
            columns=[COL_CNPJ, COL_RAZAO, COL_MC, COL_MD, COL_DIFF_ABS, COL_DIFF_PCT]
        )

    if col_cnpj and col_cnpj in df.columns:
        serie_cnpj = df[col_cnpj].astype(str)
    else:
        serie_cnpj = pd.Series("", index=df.index, dtype=str)

    if col_razao and col_razao in df.columns:
        serie_razao = df[col_razao].astype(str)
    else:
        serie_razao = pd.Series("", index=df.index, dtype=str)

    if col_mc and col_mc in df.columns:
        censo = pd.to_numeric(df[col_mc], errors="coerce")
    else:
        censo = pd.Series(np.nan, index=df.index, dtype=float)

    if col_md and col_md in df.columns:
        dms = pd.to_numeric(df[col_md], errors="coerce")
    else:
        dms = pd.Series(np.nan, index=df.index, dtype=float)

    diff_abs = censo - dms

    censo_nv = censo.to_numpy(dtype=float)
    censo_den_ok = np.isfinite(censo_nv) & (censo_nv != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_pct = (censo_nv - dms.to_numpy(dtype=float)) / censo_nv * 100.0
    diff_pct = np.where(censo_den_ok, raw_pct, np.nan)

    work = pd.DataFrame(
        {
            COL_CNPJ: serie_cnpj.values,
            COL_RAZAO: serie_razao.values,
            COL_MC: censo.values,
            COL_MD: dms.values,
            COL_DIFF_ABS: diff_abs.values,
            COL_DIFF_PCT: diff_pct,
        }
    )

    sort_key = work[COL_DIFF_PCT].abs()
    work = work.assign(_sort=sort_key).sort_values("_sort", ascending=False, na_position="last").drop(columns="_sort")

    return work.reset_index(drop=True)


def describe_column_bindings(df: pd.DataFrame, column_map: dict[str, Any]) -> dict[str, str | None]:
    """Debug legível para a UI (“qual coluna física está a ser usada”)."""

    cm = dict(column_map or {})
    paths = resolved_paths_for_dashboard(df, cm)
    return {
        "cnpj_dms_prefixed": paths.get("cnpj_dms"),
        "razao_dms_prefixed": paths.get("razao"),
        "iss_prefixed": paths.get("iss"),
        "matriculas_censo_prefixed": resolve_matricula_censo_column(df, cm),
        "matriculas_dms_prefixed": resolve_matricula_dms_column(df, cm),
    }

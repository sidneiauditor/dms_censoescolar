"""
Agregações, rankings e séries para o painel operacional (Etapa 6.2).

Separação de responsabilidades:
- :mod:`services.indicators` — cálculo **por linha** (ratios fiscais / matrícula).
- Este módulo — **agregações**, rankings, contagem de divergências, preparação de dados para gráficos
  e tabela operacional, **sem** dependência de Streamlit.
- ``app.py`` — filtros interactivos, keys de widget e renderização.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.cnpj_merge import (
    CNPJ_INVALIDO_DMS,
    MATCH_CNPJ_EXATO,
    MATCH_MULTIPLAS_ESCOLAS,
    SEM_CORRESP_CNPJ,
)
from services.inferred_mapping import pick_column
from services.indicators import (
    COL_BASE_PM,
    COL_ISS_PM,
    COL_ISS_ALIASES,
    COL_MATRICULAS_ALIASES,
    COL_MSG_PM,
    COL_MENSALIDADE_ALIASES,
)

STATUS_COL = "match_status_principal"

RAZAO_DMS_ALIASES: tuple[str, ...] = (
    "NMRAZAOSOCIAL",
    "RAZAOSOCIAL",
    "NM_RAZAO_SOCIAL",
    "RAZAO_SOCIAL",
    "NOME_CONTRIBUINTE",
)
CNPJ_DMS_ALIASES: tuple[str, ...] = ("CNPJ", "NU_CNPJ", "DOCUMENTO", "NU_DOCUMENTO")

DEPENDENCIA_ALIASES: tuple[str, ...] = (
    "TP_DEPENDENCIA",
    "DEPENDENCIA",
    "DEPENDENCIA_ADMINISTRATIVA",
)

MatriculaFaixa = Literal["todas", "zero", "1_50", "51_200", "201_mais"]


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
    prefer_physical: str | None = None,
) -> str | None:
    prefixed = _lista_colunas_prefixed(df, prefix)
    phys_list: list[str] = []
    pref_map: dict[str, str] = {}
    for pc in prefixed:
        phys = pc.split("__", 1)[1]
        phys_list.append(phys)
        pref_map[normalize_key(phys)] = pc
    if prefer_physical and isinstance(prefer_physical, str) and prefer_physical.strip():
        pk = normalize_key(prefer_physical)
        if pk in pref_map:
            return pref_map[pk]
    hit = pick_column(phys_list, candidates)
    if hit:
        return pref_map.get(normalize_key(hit))
    return None


def resolved_paths_for_dashboard(
    df: pd.DataFrame,
    column_map: dict[str, Any],
) -> dict[str, str | None]:
    """Colunas físicas no consolidado (nomes já com prefixo) usadas pelo painel."""

    censo_mat_pref = column_map.get("censo_mat")
    if not isinstance(censo_mat_pref, str) or not censo_mat_pref.strip():
        censo_mat_pref = None

    paths = {
        "matriculas": _resolve_prefixed(
            df, prefix="censo", candidates=COL_MATRICULAS_ALIASES, prefer_physical=censo_mat_pref
        ),
        "iss": _resolve_prefixed(df, prefix="dms", candidates=COL_ISS_ALIASES),
        "mensalidade": _resolve_prefixed(df, prefix="dms", candidates=COL_MENSALIDADE_ALIASES),
        "razao": _resolve_prefixed(df, prefix="dms", candidates=RAZAO_DMS_ALIASES),
        "cnpj_dms": _resolve_prefixed(df, prefix="dms", candidates=CNPJ_DMS_ALIASES),
        "dependencia": _resolve_prefixed(df, prefix="censo", candidates=DEPENDENCIA_ALIASES),
    }
    return paths


def matricula_series(df: pd.DataFrame, mat_col: str | None) -> pd.Series:
    if not mat_col or mat_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[mat_col], errors="coerce")


def iss_series(df: pd.DataFrame, iss_col: str | None) -> pd.Series:
    if not iss_col or iss_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[iss_col], errors="coerce")


def apply_matricula_faixa_mask(
    df: pd.DataFrame,
    mat_col: str | None,
    faixa: MatriculaFaixa,
) -> pd.Series:
    if faixa == "todas" or not mat_col or mat_col not in df.columns:
        return pd.Series(True, index=df.index)
    m = matricula_series(df, mat_col)
    if faixa == "zero":
        return m.fillna(0).eq(0)
    if faixa == "1_50":
        return m.ge(1) & m.le(50)
    if faixa == "51_200":
        return m.ge(51) & m.le(200)
    if faixa == "201_mais":
        return m.ge(201)
    return pd.Series(True, index=df.index)


def filter_operational_dataframe(
    df: pd.DataFrame,
    *,
    column_map: dict[str, Any],
    match_status: list[str] | None = None,
    dependencia: list[str] | None = None,
    matricula_faixa: MatriculaFaixa = "todas",
) -> pd.DataFrame:
    """Filtra linhas conforme painel (status, dependência, faixa de matrícula)."""

    if df.empty:
        return df.copy()

    out = df.copy()
    paths = resolved_paths_for_dashboard(out, column_map)
    mask = pd.Series(True, index=out.index)

    if match_status and STATUS_COL in out.columns:
        mask &= out[STATUS_COL].astype(str).isin(match_status)

    dep_col = paths.get("dependencia")
    if dependencia and dep_col and dep_col in out.columns:
        dep_s = out[dep_col].astype(str).str.strip()
        mask &= dep_s.isin({str(x).strip() for x in dependencia})

    mask &= apply_matricula_faixa_mask(out, paths.get("matriculas"), matricula_faixa)
    return out.loc[mask].copy()


@dataclass
class DashboardKpis:
    total_iss: float
    total_matriculas: float
    total_escolas: int
    total_match_exato: int
    total_sem_correspondencia: int
    linhas_filtradas: int


def compute_kpis(df: pd.DataFrame, column_map: dict[str, Any]) -> DashboardKpis:
    """KPIs sobre o subconjunto já filtrado."""

    paths = resolved_paths_for_dashboard(df, column_map)
    iss = iss_series(df, paths.get("iss"))
    mat = matricula_series(df, paths.get("matriculas"))
    cnpj_col = paths.get("cnpj_dms")

    total_iss = float(iss.fillna(0).sum())
    total_matriculas = float(mat.fillna(0).sum())

    if cnpj_col and cnpj_col in df.columns:
        s = df[cnpj_col].astype(str).str.strip()
        total_escolas = int(s[(s != "") & (s.str.lower() != "nan")].nunique())
    else:
        total_escolas = 0

    if STATUS_COL in df.columns:
        st = df[STATUS_COL].astype(str)
        total_match_exato = int(st.eq(MATCH_CNPJ_EXATO).sum())
        total_sem_correspondencia = int(st.eq(SEM_CORRESP_CNPJ).sum())
    else:
        total_match_exato = 0
        total_sem_correspondencia = 0

    return DashboardKpis(
        total_iss=total_iss,
        total_matriculas=total_matriculas,
        total_escolas=total_escolas,
        total_match_exato=total_match_exato,
        total_sem_correspondencia=total_sem_correspondencia,
        linhas_filtradas=len(df.index),
    )


@dataclass
class DivergenceCounts:
    sem_correspondencia: int
    multiplas_escolas: int
    sem_matricula: int
    cnpj_invalido: int


def compute_divergence_counts(df: pd.DataFrame, column_map: dict[str, Any]) -> DivergenceCounts:
    paths = resolved_paths_for_dashboard(df, column_map)
    mat = matricula_series(df, paths.get("matriculas"))
    sem_mat = mat.isna() | (mat.fillna(0) <= 0)

    if STATUS_COL not in df.columns:
        return DivergenceCounts(0, 0, int(sem_mat.sum()), 0)

    stv = df[STATUS_COL].astype(str)
    return DivergenceCounts(
        sem_correspondencia=int(stv.eq(SEM_CORRESP_CNPJ).sum()),
        multiplas_escolas=int(stv.eq(MATCH_MULTIPLAS_ESCOLAS).sum()),
        sem_matricula=int(sem_mat.sum()),
        cnpj_invalido=int(stv.eq(CNPJ_INVALIDO_DMS).sum()),
    )


@dataclass
class DashboardRankings:
    top_iss: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_iss_por_matricula: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_mensalidade_por_aluno: pd.DataFrame = field(default_factory=pd.DataFrame)


def _ranking_base_columns(
    df: pd.DataFrame,
    column_map: dict[str, Any],
) -> pd.DataFrame:
    paths = resolved_paths_for_dashboard(df, column_map)
    base = pd.DataFrame(index=df.index)
    if paths.get("cnpj_dms") and paths["cnpj_dms"] in df.columns:
        base["cnpj"] = df[paths["cnpj_dms"]]
    else:
        base["cnpj"] = ""
    if paths.get("razao") and paths["razao"] in df.columns:
        base["razao"] = df[paths["razao"]]
    else:
        base["razao"] = ""
    base[STATUS_COL] = df[STATUS_COL] if STATUS_COL in df.columns else ""
    if paths.get("iss") and paths["iss"] in df.columns:
        base["iss"] = iss_series(df, paths["iss"])
    else:
        base["iss"] = np.nan
    if paths.get("matriculas") and paths["matriculas"] in df.columns:
        base["matriculas"] = matricula_series(df, paths["matriculas"])
    else:
        base["matriculas"] = np.nan
    if COL_ISS_PM in df.columns:
        base[COL_ISS_PM] = pd.to_numeric(df[COL_ISS_PM], errors="coerce")
    else:
        base[COL_ISS_PM] = np.nan
    if COL_MSG_PM in df.columns:
        base[COL_MSG_PM] = pd.to_numeric(df[COL_MSG_PM], errors="coerce")
    else:
        base[COL_MSG_PM] = np.nan
    return base


def compute_rankings(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    top_n: int = 10,
) -> DashboardRankings:
    """Top N linhas por ISS bruto, ISS/matricula e mensalidade/aluno."""

    if df.empty:
        return DashboardRankings()

    work = _ranking_base_columns(df, column_map)
    ix_iss = work["iss"].fillna(-np.inf).nlargest(top_n).index
    ix_ratio = (
        work[COL_ISS_PM]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-np.inf)
        .nlargest(top_n)
        .index
    )
    ix_m = (
        work[COL_MSG_PM]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-np.inf)
        .nlargest(top_n)
        .index
    )

    cols_show = ["razao", "cnpj", "matriculas", "iss", STATUS_COL, COL_ISS_PM, COL_MSG_PM]

    def _take(idx: pd.Index) -> pd.DataFrame:
        return work.loc[idx, [c for c in cols_show if c in work.columns]].reset_index(drop=True)

    return DashboardRankings(
        top_iss=_take(ix_iss),
        top_iss_por_matricula=_take(ix_ratio),
        top_mensalidade_por_aluno=_take(ix_m),
    )


def build_operational_table(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    *,
    max_rows: int = 500,
) -> pd.DataFrame:
    """Tabela única para operação fiscal: razão, CNPJ, matrículas, ISS, status, indicadores."""

    if df.empty:
        return pd.DataFrame()

    paths = resolved_paths_for_dashboard(df, column_map)
    out = pd.DataFrame(
        {
            "razão_social": df[paths["razao"]] if paths.get("razao") in df.columns else "",
            "cnpj": df[paths["cnpj_dms"]] if paths.get("cnpj_dms") in df.columns else "",
            "matrículas": matricula_series(df, paths.get("matriculas")),
            "iss": iss_series(df, paths.get("iss")),
            "status_match": df[STATUS_COL] if STATUS_COL in df.columns else "",
        },
        index=df.index,
    )
    for c in (COL_ISS_PM, COL_MSG_PM, COL_BASE_PM):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = np.nan

    dep = paths.get("dependencia")
    if dep and dep in df.columns:
        out["dependência_administrativa"] = df[dep]

    return out.head(max_rows).reset_index(drop=True)


def match_status_distribution(df: pd.DataFrame) -> pd.Series:
    if df.empty or STATUS_COL not in df.columns:
        return pd.Series(dtype=float)
    return df[STATUS_COL].astype(str).value_counts()


def figure_donut_match_status(df: pd.DataFrame):
    vc = match_status_distribution(df)
    if vc.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sem dados de status", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    fig = go.Figure(
        data=[
            go.Pie(
                labels=vc.index.tolist(),
                values=vc.values.tolist(),
                hole=0.52,
                sort=False,
            )
        ]
    )
    fig.update_layout(
        title="Distribuição por status de match",
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def figure_bar_iss_by_status(df: pd.DataFrame, column_map: dict[str, Any]):
    paths = resolved_paths_for_dashboard(df, column_map)
    iss_col = paths.get("iss")
    if df.empty or STATUS_COL not in df.columns or not iss_col or iss_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Sem ISS ou status", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    tmp = pd.DataFrame({"status": df[STATUS_COL].astype(str), "_iss": iss_series(df, iss_col)})
    g = tmp.groupby("status", dropna=False)["_iss"].sum().reset_index()
    g.columns = ["status", "iss_total"]
    fig = px.bar(g, x="status", y="iss_total", title="ISS total por status de match")
    fig.update_layout(xaxis_title="Status", yaxis_title="ISS (soma)", margin=dict(t=50, b=80))
    return fig


def figure_scatter_matriculas_iss(df: pd.DataFrame, column_map: dict[str, Any]):
    paths = resolved_paths_for_dashboard(df, column_map)
    mcol, icol = paths.get("matriculas"), paths.get("iss")
    if (
        df.empty
        or not mcol
        or not icol
        or mcol not in df.columns
        or icol not in df.columns
    ):
        fig = go.Figure()
        fig.add_annotation(
            text="Matrículas ou ISS indisponíveis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig
    plot_df = pd.DataFrame(
        {
            "matrículas": matricula_series(df, mcol).clip(lower=0),
            "iss": iss_series(df, icol),
            "status": df[STATUS_COL].astype(str) if STATUS_COL in df.columns else "",
        }
    ).dropna(subset=["matrículas", "iss"], how="all")
    fig = px.scatter(
        plot_df,
        x="matrículas",
        y="iss",
        color="status" if plot_df["status"].nunique() > 1 else None,
        title="Matrículas × ISS (por linha consolidada)",
        opacity=0.65,
    )
    fig.update_layout(margin=dict(t=50, b=40))
    return fig

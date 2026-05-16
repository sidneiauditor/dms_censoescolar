"""
Consolidação interna do Censo: tabela Escola ⊕ Matrícula sobre ``CO_ENTIDADE``.

Independente do ano ou do nome do ficheiro — apenas do mapeamento lógico definido pelo utilizador.
"""

from __future__ import annotations

import logging

import pandas as pd

from domain.census_logical import CENSO_ESCOLA_FIELDS, CENSO_MATRICULA_FIELDS

LOG = logging.getLogger(__name__)


class CensusMergeError(Exception):
    """Erro recuperável na validação ou junção das tabelas do Censo."""


def normalize_co_entidade(series: pd.Series) -> pd.Series:
    """Uniformiza código escolar para junção (texto limpo, sem sufixo ``.0`` flutuante)."""

    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    lower = out.str.lower()
    out = out.mask(lower.isin({"nan", "none", "<na>", ""}), "")
    return out


def _validate_escola_mapping(mapping: dict[str, str]) -> None:
    for spec in CENSO_ESCOLA_FIELDS:
        if not spec.obrigatorio_escola:
            continue
        src = mapping.get(spec.key)
        if not src:
            raise CensusMergeError(
                f"Mapeamento incompleto na **Escola**: falta o campo obrigatório `{spec.key}`."
            )


def _validate_matricula_mapping(mapping: dict[str, str]) -> None:
    for spec in CENSO_MATRICULA_FIELDS:
        if not spec.obrigatorio_matricula:
            continue
        src = mapping.get(spec.key)
        if not src:
            raise CensusMergeError(
                f"Mapeamento incompleto na **Matrícula**: falta o campo obrigatório `{spec.key}`."
            )


def _project_logical_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Extrai só colunas mapeadas e renomeia para nomes lógicos estáveis."""

    chunks: dict[str, pd.Series] = {}
    for logical, physical in mapping.items():
        if not physical or physical not in df.columns:
            raise CensusMergeError(
                f"A coluna física `{physical}` para `{logical}` não existe neste ficheiro."
            )
        chunks[logical] = df[physical].astype(str)

    out = pd.DataFrame(chunks)
    if "CO_ENTIDADE" in out.columns:
        out["CO_ENTIDADE"] = normalize_co_entidade(out["CO_ENTIDADE"])
    return out


def consolidate_census_escolar(
    df_escola: pd.DataFrame,
    df_matricula: pd.DataFrame | None,
    map_escola: dict[str, str],
    map_matricula: dict[str, str],
    exercise_year: int,
    *,
    source_escola_label: str = "",
    source_matricula_label: str = "",
) -> pd.DataFrame:
    """
    Produz uma única base Censo com colunas lógicas.

    - Se só existir Escola → devolve apenas as colunas lógicas da escola (+ metadados).
    - Se existir Escola *e* Matrícula → ``merge`` externo por ``CO_ENTIDADE``.
    """

    _validate_escola_mapping(map_escola)
    esc_part = _project_logical_columns(df_escola, map_escola)

    dup_esc = esc_part["CO_ENTIDADE"].duplicated().sum()
    if dup_esc:
        LOG.warning(
            "%s valores duplicados de CO_ENTIDADE na base Escola — o merge pode gerar mais linhas.",
            dup_esc,
        )

    if df_matricula is None or df_matricula.empty:
        LOG.info(
            "Consolidação Censo só com Escola (%s linhas), exercício=%s.",
            len(esc_part.index),
            exercise_year,
        )
        out = esc_part.copy()
        out["censo_fonte_escola"] = source_escola_label
        out["censo_fonte_matricula"] = ""
        out["censo_exercicio"] = str(exercise_year)
        return out

    _validate_matricula_mapping(map_matricula)
    mat_part = _project_logical_columns(df_matricula, map_matricula)

    dup_mat = mat_part["CO_ENTIDADE"].duplicated().sum()
    if dup_mat:
        LOG.warning(
            "%s valores duplicados de CO_ENTIDADE na base Matrícula.",
            dup_mat,
        )

    merged = esc_part.merge(
        mat_part,
        on="CO_ENTIDADE",
        how="outer",
        suffixes=("_base_escola", "_base_matricula"),
    )

    merged["censo_fonte_escola"] = source_escola_label
    merged["censo_fonte_matricula"] = source_matricula_label
    merged["censo_exercicio"] = str(exercise_year)

    LOG.info(
        "Censo consolidado Escola⊕Matrícula: %s linhas (exercício %s).",
        len(merged.index),
        exercise_year,
    )

    return merged

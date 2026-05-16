"""
Consolidação interna do Censo: tabela Escola ⊕ Matrícula sobre ``CO_ENTIDADE``.

Independente do ano ou do nome do ficheiro — apenas do mapeamento lógico definido pelo utilizador.

**Matrícula EB (Etapa 8.1)** — quando o campo lógico ``matriculas`` aponta para a coluna física oficial
INEP ``QT_MAT_BAS``, esse nome é **preservado** no ``DataFrame`` consolidado como coluna ``QT_MAT_BAS``
(``Float64``/float), paralelamente a ``matriculas`` na mesma base numérica — para uso canónico pelo
painel municipal e pela agregação por CNPJ sem ambiguidade de renomeação.
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


def _project_logical_columns(
    df: pd.DataFrame,
    mapping: dict[str, str],
    *,
    numeric_logical: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Extrai colunas mapeadas e renomeia para nomes lógicos estáveis."""

    chunks: dict[str, pd.Series] = {}
    for logical, physical in mapping.items():
        if not physical or physical not in df.columns:
            raise CensusMergeError(
                f"A coluna física `{physical}` para `{logical}` não existe neste ficheiro."
            )
        if logical in numeric_logical:
            chunks[logical] = pd.to_numeric(df[physical], errors="coerce")
        else:
            chunks[logical] = df[physical].astype(str)

    out = pd.DataFrame(chunks)
    if "CO_ENTIDADE" in out.columns:
        out["CO_ENTIDADE"] = normalize_co_entidade(out["CO_ENTIDADE"])
    return out


QT_MAT_BAS_CANONICAL = "QT_MAT_BAS"


def _sync_qt_mat_bas_canonical(
    mat_part: pd.DataFrame,
    df_matricula_raw: pd.DataFrame,
    map_matricula: dict[str, str],
) -> pd.DataFrame:
    """
    Se ``matriculas`` ↔ coluna física cujo **nome** é ``QT_MAT_BAS`` (INEP, case-insensitive),
    expõe **``QT_MAT_BAS``** como coluna canónica numérica no mesmo conjunto de linhas.

    Não confunde outros totais (ex.: ``TOTAL_MATRICULAS``) com ``QT_MAT_BAS``: só espelha quando
    o cabeçalho físico já é ``QT_MAT_BAS``.
    """

    phys = map_matricula.get("matriculas")
    if (
        phys is None
        or phys not in df_matricula_raw.columns
        or str(phys).strip().upper() != QT_MAT_BAS_CANONICAL
        or QT_MAT_BAS_CANONICAL in mat_part.columns
    ):
        return mat_part

    # Mesma série já usada logicamente como `matriculas`, mas com nome oficial INEP preservado.
    out = mat_part.copy()
    out[QT_MAT_BAS_CANONICAL] = pd.to_numeric(df_matricula_raw[phys], errors="coerce")

    nonzero = out[QT_MAT_BAS_CANONICAL].notna() & out[QT_MAT_BAS_CANONICAL].ne(0)
    LOG.debug(
        "Preservação `%s`: %s valores numéricos não nulos/not zero / %s linhas.",
        QT_MAT_BAS_CANONICAL,
        int(nonzero.sum()),
        len(out.index),
    )

    return out


def diagnose_matricula_qt_mat_bas(df_matricula: pd.DataFrame) -> dict[str, object]:
    """Métricas leves sobre a coluna física ``QT_MAT_BAS``, se existir no extracto."""

    cols_norm = {str(c).strip().upper(): str(c) for c in df_matricula.columns}
    phys = cols_norm.get(QT_MAT_BAS_CANONICAL)
    if phys is None or phys not in df_matricula.columns:
        return {
            "qt_mat_bas_presente": False,
            "physical_header": None,
            "dtype": None,
            "n_non_numeric_coerced_na": None,
            "n_valid_numeric": None,
        }

    ser = pd.to_numeric(df_matricula[phys], errors="coerce")
    raw_obj = df_matricula[phys]
    n_coerced = int(raw_obj.notna().sum() - ser.notna().sum())
    n_valid = int(ser.notna().sum())

    return {
        "qt_mat_bas_presente": True,
        "physical_header": phys,
        "dtype": str(df_matricula[phys].dtype),
        "n_non_numeric_coerced_na": n_coerced,
        "n_valid_numeric": n_valid,
    }


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

    qdiag = diagnose_matricula_qt_mat_bas(df_matricula)
    if qdiag.get("qt_mat_bas_presente"):
        LOG.info(
            "Matrícula INEP: `%s` presente (dtype bruto=%s, valores numéricos válidos após coerção=%s, "
            "células não numéricas→NaN=%s).",
            qdiag.get("physical_header"),
            qdiag.get("dtype"),
            qdiag.get("n_valid_numeric"),
            qdiag.get("n_non_numeric_coerced_na"),
        )
    else:
        LOG.info(
            "Matrícula: coluna física **%s** não encontrada — agregação 8.1 só terá `QT_MAT_BAS` "
            "se mapear `matriculas` para esse cabeçalho.",
            QT_MAT_BAS_CANONICAL,
        )

    mat_part = _project_logical_columns(
        df_matricula,
        map_matricula,
        numeric_logical=frozenset({"matriculas"}),
    )
    mat_part = _sync_qt_mat_bas_canonical(mat_part, df_matricula, map_matricula)

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

"""
Semântica oficial de matrículas do **Censo Escolar INEP** (Etapa 8.1).

- **Um único total consolidado** para cruzamento fiscal: ``QT_MAT_BAS`` (Educação Básica).
- Subcampos hierárquicos (ex.: creches, médio) **não são somados** entre si nem com o pai:
  fazem parte de uma árvore; somar ramos lado a lado produz dupla contagem.
"""

from __future__ import annotations

from typing import Any, FrozenSet

# Nome físico esperado nos microdados conforme metadados INEP públicos / dicionários.
CENSUS_CANONICAL_FIELDS: dict[str, str] = {
    "matriculas_total": "QT_MAT_BAS",
    "educacao_infantil": "QT_MAT_INF",
    "creche": "QT_MAT_INF_CRE",
    "fundamental": "QT_MAT_FUND",
    "medio": "QT_MAT_MED",
}

# Metadados: filho → pai lógico. ``QT_MAT_BAS`` é raiz nominal (pai ``None``).
CENSUS_MATRICULA_HIERARCHY: dict[str, dict[str, Any]] = {
    "QT_MAT_INF_CRE": {"parent": "QT_MAT_INF", "is_leaf": True},
    "QT_MAT_PRE": {"parent": "QT_MAT_INF", "is_leaf": True},
    "QT_MAT_INF": {"parent": "QT_MAT_BAS", "is_leaf": False},
    "QT_MAT_FUND": {"parent": "QT_MAT_BAS", "is_leaf": False},
    "QT_MAT_MED": {"parent": "QT_MAT_BAS", "is_leaf": False},
    "QT_MAT_BAS": {"parent": None, "is_leaf": False},
}

# Campos hierárquicos que não devem ser somados ao mesmo nível nem entre si como «totais extra».
CENSUS_HORIZONTAL_NON_ADDITIVE_FIELDS: FrozenSet[str] = frozenset(
    {"QT_MAT_INF", "QT_MAT_INF_CRE", "QT_MAT_PRE", "QT_MAT_FUND", "QT_MAT_MED"}
)


def physical_qt_mat_bas_column(columns: Any) -> str | None:
    """
    Resolve o cabeçalho físico **exclusivamente** como ``QT_MAT_BAS``
    (case-insensitive sobre o iterável de nomes de colunas).
    """

    canonical = CENSUS_CANONICAL_FIELDS["matriculas_total"]
    nk = canonical.strip().upper()
    if hasattr(columns, "tolist"):
        names = [str(x) for x in columns.tolist()]
    else:
        names = [str(x) for x in list(columns)]
    for c in names:
        if c.strip().upper() == nk:
            return c
    return None


def is_horizontal_sum_forbidden(column_name: str) -> bool:
    """Retorna ``True`` se o campo faz parte dos subconjuntos que não devem ser somados entre si."""

    short = str(column_name).strip().upper()
    return short in CENSUS_HORIZONTAL_NON_ADDITIVE_FIELDS

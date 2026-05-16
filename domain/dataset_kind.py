"""Tipos de base reconhecidos pela aplicação (genéricos por exercício)."""

from __future__ import annotations

from enum import Enum


class DatasetKind(str, Enum):
    """Identificador estável — não depende do nome do ficheiro nem do ano."""

    DMS_EDUCACAO = "dms_educacao"
    CENSO_ESCOLA = "censo_escola"
    CENSO_MATRICULA = "censo_matricula"


LABEL_PT: dict[DatasetKind, str] = {
    DatasetKind.DMS_EDUCACAO: "DMS Educação",
    DatasetKind.CENSO_ESCOLA: "Censo Escola",
    DatasetKind.CENSO_MATRICULA: "Censo Matrícula",
}


def label(kind: DatasetKind) -> str:
    return LABEL_PT.get(kind, kind.value)

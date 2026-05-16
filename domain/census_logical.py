"""Campos lógicos internos — independentes dos nomes físicos das colunas INEP/export."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogicalFieldSpec:
    """Descrição de um papel semântico na consolidação do Censo."""

    key: str
    description_pt: str
    obrigatorio_escola: bool = False
    obrigatorio_matricula: bool = False


CENSO_ESCOLA_FIELDS: tuple[LogicalFieldSpec, ...] = (
    LogicalFieldSpec(
        "CO_ENTIDADE",
        "Identificador único da escola (ex.: ``CO_ENTIDADE`` microdados INEP).",
        obrigatorio_escola=True,
    ),
    LogicalFieldSpec(
        "NO_ENTIDADE",
        "Denominação da escola (ex.: ``NO_ENTIDADE``).",
    ),
    LogicalFieldSpec(
        "dependencia_administrativa",
        "Dependência administrativa (ex.: ``TP_DEPENDENCIA`` ou equivalente na sua base).",
    ),
    LogicalFieldSpec(
        "modalidade",
        "Modalidade ou indicador equivalente conforme o layout do exercício.",
    ),
    LogicalFieldSpec(
        "CNPJ",
        "CNPJ para cruzamento fiscal, se existir na tabela escola (ex.: ``NU_CNPJ_ESCOLA_PRIVADA``).",
    ),
)

CENSO_MATRICULA_FIELDS: tuple[LogicalFieldSpec, ...] = (
    LogicalFieldSpec(
        "CO_ENTIDADE",
        "Mesmo identificador da escola que na tabela escola.",
        obrigatorio_matricula=True,
    ),
    LogicalFieldSpec(
        "matriculas",
        "Quantidade ou campo agregado de matrículas (ex.: ``QT_MAT_BAS`` ou total declarado).",
        obrigatorio_matricula=True,
    ),
)

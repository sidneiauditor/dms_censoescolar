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
    LogicalFieldSpec(
        "SG_UF",
        "Sigla da UF para filtro municipal (ex.: ``SG_UF`` nos microdados INEP). Opcional.",
    ),
    LogicalFieldSpec(
        "CO_MUNICIPIO",
        "Código IBGE do município (ex.: ``CO_MUNICIPIO``). Opcional — necessário para filtrar só escolas municipais.",
    ),
    LogicalFieldSpec(
        "NO_MUNICIPIO",
        "Nome do município no Censo (ex.: ``NO_MUNICIPIO``). Opcional — útil quando não há só código.",
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
        "Total de matrículas (EB): mapear para **QT_MAT_BAS** nos microdados INEP quando existir; "
        "o consolidador preserva também o nome oficial ``QT_MAT_BAS`` na base final.",
        obrigatorio_matricula=True,
    ),
)

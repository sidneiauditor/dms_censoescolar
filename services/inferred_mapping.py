"""
Inferência de colunas físicas típicas (INEP/microdados) sem depender do nome do ficheiro.

Comparações **case-insensitive** com normalização simples dos identificadores.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

LOG = logging.getLogger(__name__)


def normalize_identifier(column_name: object) -> str:
    """Chave comparable: maiúsculas, trim, espaços → ``_``, remove ``_`` repetidos."""

    raw = str(column_name).strip()
    cleaned = raw.upper().replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned


def pick_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    """
    Escolhe a primeira coluna cujo nome normalizado coincide com algum alias.
    Mantém o nome originalmente presente na base.
    """
    index = {normalize_identifier(c): str(c).strip() for c in columns}
    for cand in candidates:
        key = normalize_identifier(cand)
        if key in index:
            return index[key]
    return None


# Aliases ordenados por prioridade (mais específicos primeiro quando aplicável).

ESCOLA_ALIASES: dict[str, tuple[str, ...]] = {
    "CO_ENTIDADE": ("CO_ENTIDADE", "COD_ESCOLA", "CODESCOLA", "INEP_ESCOLA"),
    "NO_ENTIDADE": ("NO_ENTIDADE", "NOME_ENTIDADE", "NOME_ESCOLA"),
    "dependencia_administrativa": ("TP_DEPENDENCIA", "DEPENDENCIA", "DEPENDENCIA_ADMINISTRATIVA"),
    "modalidade": ("TIPO_ESCOLARIZACAO", "MODALIDADE", "DESC_MODALIDADE"),
    "CNPJ": ("NU_CNPJ_ESCOLA_PRIVADA", "CNPJ_ESCOLA", "CNPJ", "NU_CNPJ_MANTENEDORA"),
    "SG_UF": ("SG_UF", "SIGLA_UF", "UF"),
    "CO_MUNICIPIO": ("CO_MUNICIPIO", "COD_MUNICIPIO", "MUNICIPIO_CODIGO", "COD_IBGE"),
    "NO_MUNICIPIO": ("NO_MUNICIPIO", "NOME_MUNICIPIO", "MUNICIPIO_NOME"),
}

MATRICULA_ALIASES: dict[str, tuple[str, ...]] = {
    "CO_ENTIDADE": ("CO_ENTIDADE", "COD_ESCOLA", "INEP_ESCOLA"),
    "matriculas": ("QT_MAT_BAS", "MATRICULAS", "TOTAL_MATRICULAS", "QT_TOTAL_MATRICULAS"),
}

DMS_ALIASES: dict[str, tuple[str, ...]] = {
    "CNPJ": ("CNPJ", "NU_CNPJ", "DOCUMENTO"),
    "razao_social": ("NMRAZAOSOCIAL", "RAZAOSOCIAL", "NM_RAZAO_SOCIAL"),
    "quantidade": ("QUANTIDADE", "QTDE", "QTD_ALUNOS", "QTD", "TOTAL"),
}


def propose_escola_mapping(columns: list[str]) -> dict[str, str]:
    proposed: dict[str, str] = {}
    cols = [str(c) for c in columns]
    for logical, aliases in ESCOLA_ALIASES.items():
        hit = pick_column(cols, aliases)
        if hit:
            proposed[logical] = hit
            LOG.debug("Auto-map Escola: %s ← %s", logical, hit)
    return proposed


def propose_matricula_mapping(columns: list[str]) -> dict[str, str]:
    proposed: dict[str, str] = {}
    cols = [str(c) for c in columns]
    for logical, aliases in MATRICULA_ALIASES.items():
        hit = pick_column(cols, aliases)
        if hit:
            proposed[logical] = hit
            LOG.debug("Auto-map Matrícula: %s ← %s", logical, hit)
    return proposed


def propose_dms_mapping(columns: list[str]) -> dict[str, str]:
    proposed: dict[str, str] = {}
    cols = [str(c) for c in columns]
    for logical, aliases in DMS_ALIASES.items():
        hit = pick_column(cols, aliases)
        if hit:
            proposed[logical] = hit
            LOG.debug("Auto-map DMS: %s ← %s", logical, hit)
    return proposed

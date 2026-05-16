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
    "competencia": ("DTCOMPETENCIA", "DT_COMPETENCIA", "COMPETENCIA", "MES_COMPETENCIA"),
}


# Ordem própria do modo **operacional** (export DMS-Educação com cabeçalhos como ``NUCNPJ``).
DMS_OPERACIONAL_CNPJ_PRIORITY_ALIASES: tuple[str, ...] = (
    "NUCNPJ",
    "NU_CNPJ",
    "CNPJ",
    "CPF_CNPJ",
    "CNPJCPF",
)


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


def infer_dms_cnpj_column_operacional(columns: Iterable[str]) -> tuple[str | None, str]:
    """
    Resolver CNPJ da DMS apenas no fluxo **operacional**.

    Percorre ``DMS_OPERACIONAL_CNPJ_PRIORITY_ALIASES`` na ordem; compara contra cabeçalhos físicos usando
    :func:`normalize_identifier` para tolerar espaços/caixa variantes sem alterar ``propose_dms_mapping`` dos demais fluxos.

    Devolve ``(nome_da_coluna_encontrado, etiqueta_metodo_logs)``.
    """

    cols_str = [str(c).strip() for c in columns if str(c).strip()]
    index = {normalize_identifier(c): c for c in cols_str}
    for alias in DMS_OPERACIONAL_CNPJ_PRIORITY_ALIASES:
        key = normalize_identifier(alias)
        if key in index:
            resolved = index[key]
            method = (
                f"modo_operacional_prioridade(alias_normalizado={key!s}, primeiro_match_na_ordem_dms_export)"
            )
            LOG.debug("infer_dms_cnpj_column_operacional: %s → %s (%s)", alias, resolved, method)
            return resolved, method
    return None, ""


CONSOLIDADO_CNPJ_MERGE_FALLBACK_PRIORITY: tuple[str, ...] = (
    "CNPJ",
    "CNPJ_base_escola",
    "CNPJ_base_matricula",
)


def resolve_census_cnpj_physical_column(columns: Iterable[str]) -> str | None:
    """
    Localiza uma coluna de CNPJ **após consolidar Escola⊕Matrícula**.

    O alias INEP habitual costuma ficar igual a ``CNPJ``; quando ambas as tabelas têm coluna física chamada assim,
    ``census_consolidator`` pode produzir ``CNPJ_base_escola`` / ``CNPJ_base_matricula``.
    """

    cols = [str(c) for c in columns]

    inferred = propose_escola_mapping(list(cols)).get("CNPJ")
    if inferred and inferred in cols:
        LOG.debug("CNPJ censo resolver: usar proposta `%s`.", inferred)
        return inferred

    for cand in CONSOLIDADO_CNPJ_MERGE_FALLBACK_PRIORITY:
        if cand in cols:
            LOG.debug("CNPJ censo resolver: fallback prioritário `%s`.", cand)
            return cand

    # Colunas tipo ``CNPJ_base_escola`` que escaparam aos casos acima:
    nk_cnpj = normalize_identifier("CNPJ")
    prefix_hits = [c for c in cols if normalize_identifier(c).startswith(f"{nk_cnpj}_")]

    def _prio_fallback(name: str) -> tuple[int, str]:
        nid = normalize_identifier(name)
        if nid == "CNPJ_BASE_ESCOLA":
            order = 0
        elif nid == "CNPJ_BASE_MATRICULA":
            order = 1
        else:
            order = 9
        return (order, name)

    if prefix_hits:
        prefix_hits = sorted(prefix_hits, key=_prio_fallback)
        pick = prefix_hits[0]
        LOG.warning("CNPJ censo resolver: heurística de sufixo `%s`.", pick)
        return pick

    return None

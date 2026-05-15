"""
Normalização e validação de CNPJ (apenas dígitos, dígito verificador).

Compatível com entradas com máscara (pontos, barra, hífen, espaços).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


def only_digits(value: object) -> str:
    """Extrai apenas os dígitos de um valor textual ou numérico."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return "".join(ch for ch in text if ch.isdigit())


def normalize_cnpj_digits(value: object) -> str | None:
    """
    Produz uma string com 14 dígitos ou None se vazio ou inválido pela regra simples:

    - vazio ou sem dígitos → ``None`` (tratado como *vazio* na contagem);
    - mais de 14 dígitos numéricos → ``None`` (inválido: ambiguidade);
    - caso contrário ``zfill`` à esquerda até 14 posições.
    """
    d = only_digits(value)
    if not d:
        return None
    if len(d) > 14:
        logger.debug("CNPJ descartado: mais de 14 dígitos extraídos")
        return None
    padded = d.zfill(14)
    if len(padded) != 14:
        return None
    return padded


def cnpj_checksum_ok(d14: str) -> bool:
    """Verifica dígitos verificadores do CNPJ (Cadastro Nacional de PJ)."""
    if len(d14) != 14 or not d14.isdigit():
        return False
    if d14 == "00000000000000":
        return False
    digits = [int(c) for c in d14]
    w1 = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
    s1 = sum(w * n for w, n in zip(w1, digits[:12]))
    r1 = 11 - (s1 % 11)
    dv1 = 0 if r1 >= 10 else r1
    if dv1 != digits[12]:
        return False
    w2 = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
    s2 = sum(w * n for w, n in zip(w2, digits[:13]))
    r2 = 11 - (s2 % 11)
    dv2 = 0 if r2 >= 10 else r2
    return dv2 == digits[13]


@dataclass(frozen=True)
class CNPJClassification:
    """Contagens agregadas de uma coluna relativamente ao CNPJ."""

    empty: int
    invalid_format_or_checksum: int
    valid_checksum: int
    total: int


def classify_cnpj_cell(value: object) -> str:
    """
    Devolve uma etiqueta por célula: ``empty``, ``invalid`` ou ``valid``.
    """
    normalized = normalize_cnpj_digits(value)
    if normalized is None:
        raw_digits = only_digits(value)
        if not raw_digits:
            return "empty"
        return "invalid"
    if not cnpj_checksum_ok(normalized):
        return "invalid"
    return "valid"


def summarize_cnpj_column(series: pd.Series) -> CNPJClassification:
    """Estatísticas de validade para uma coluna (uma linha = um valor)."""
    labels = series.map(classify_cnpj_cell)
    empty = int((labels == "empty").sum())
    invalid = int((labels == "invalid").sum())
    valid_c = int((labels == "valid").sum())
    total = len(labels.index)
    return CNPJClassification(
        empty=empty,
        invalid_format_or_checksum=invalid,
        valid_checksum=valid_c,
        total=total,
    )


def add_normalized_cnpj_column(
    df: pd.DataFrame, source_col: str, out_col: str
) -> pd.DataFrame:
    """Adiciona coluna apenas com dígitos normalizados (14 dígitos) onde aplicável; vazio caso contrário."""

    def _norm(v: object) -> str:
        n = normalize_cnpj_digits(v)
        return n if n is not None else ""

    result = df.copy()
    result[out_col] = result[source_col].map(_norm)
    return result

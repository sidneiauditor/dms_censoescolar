"""
Normalização textual para matching razão social × nome de escola.

Passos: remover acentos (Unicode NFKD), sufixos societários comuns,
caracteres não alfanuméricos (mantendo letras Unicode após stripping),
maiúsculas e espaços compactados.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Suífijos frequentes em PJ brasileiras (palavras isoladas).
_RE_LTDA = re.compile(r"\bLTDA\.?\b", re.IGNORECASE)
_RE_SA = re.compile(r"\bS\s*/\s*A\.?\b", re.IGNORECASE)
_RE_ME = re.compile(r"\bM\.?\s*E\.?\b|\bME\b", re.IGNORECASE)


def strip_accents_keep_unicode_letters(text: str) -> str:
    """
    Remove marcas diacríticas (ex.: Á → A). Mantém letras base Unicode (Ç permanece Ç).
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def normalize_match_text(raw: object) -> str:
    """
    Texto preparado para RapidFuzz (comparável entre DMS e Censo).

    Se ``raw`` for vazio ou NA, devolve string vazia.
    """
    if raw is None:
        return ""
    if isinstance(raw, float) and pd.isna(raw):
        return ""
    text = str(raw).strip()
    if not text or text.upper() == "NAN":
        return ""

    text = strip_accents_keep_unicode_letters(text)
    text = text.upper()

    text = _RE_LTDA.sub(" ", text)
    text = _RE_SA.sub(" ", text)
    text = _RE_ME.sub(" ", text)

    # Mantém letras/dígitos (Unicode \w inclui Ç etc.) e converte resto em espaço.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_series(series: pd.Series) -> pd.Series:
    """Aplica ``normalize_match_text`` a cada célula."""

    return series.map(normalize_match_text)

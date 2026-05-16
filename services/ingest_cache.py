"""
Compatibilidade com código mais antigo — delega para ``services.table_loader``.

Novos desenvolvimentos devem usar ``load_dataset_bundle`` com ``DatasetKind``.
"""

from __future__ import annotations

from typing import Any

from domain.dataset_kind import DatasetKind
from services.table_loader import load_dataset_bundle


def load_dms_cached(raw: bytes, filename: str) -> tuple[Any, dict[str, Any]]:
    bundle = load_dataset_bundle(DatasetKind.DMS_EDUCACAO.value, raw, filename)
    return bundle["dataframe"], bundle.get("meta") or {}


def load_censo_cached(raw: bytes, filename: str):
    bundle = load_dataset_bundle(DatasetKind.CENSO_ESCOLA.value, raw, filename)
    return bundle["dataframe"]

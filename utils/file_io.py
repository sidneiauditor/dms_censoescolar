"""
Leitura de ficheiros XLSX e CSV para o upload Streamlit.

Mantém a lógica fora do app.py para reutilização nas próximas etapas.
"""

from __future__ import annotations

import io
import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Erro quando o tipo ou conteúdo do ficheiro não é aceite."""


def validate_upload_name(filename: str) -> Literal["csv", "xlsx"]:
    """
    Aceita apenas .csv e .xlsx (minúsculas ou maiúsculas no sufixo).
    Retorna o tipo inferido para roteamento do parser.
    """
    if not filename or not isinstance(filename, str):
        raise FileValidationError("Nome de ficheiro inválido ou vazio.")
    lower = filename.strip().lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xlsx"):
        return "xlsx"
    raise FileValidationError(
        f"Extensão não suportada. Utilize CSV ou XLSX. Recebido: {filename!r}"
    )


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Tenta `;` antes de `,` e encodings comuns (microdados INEP: `;`)."""
    encodings = ("utf-8-sig", "utf-8", "latin-1", "cp1252")
    seps = (";", ",")
    last_error: Exception | None = None
    for encoding in encodings:
        for sep in seps:
            try:
                bio = io.BytesIO(raw)
                df = pd.read_csv(bio, sep=sep, encoding=encoding, dtype=str)
                logger.info(
                    "CSV lido com encoding=%s sep=%s linhas=%s colunas=%s",
                    encoding,
                    repr(sep),
                    len(df.index),
                    len(df.columns),
                )
                return df
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                continue
    raise FileValidationError(
        "Não foi possível ler o CSV com encodings/separadores usuais."
    ) from last_error


def _read_xlsx_bytes(raw: bytes) -> pd.DataFrame:
    """Lê primeira folha do Excel via openpyxl (engine obrigatório no pandas)."""
    try:
        bio = io.BytesIO(raw)
        df = pd.read_excel(bio, engine="openpyxl", dtype=str)
        logger.info(
            "XLSX lido: linhas=%s colunas=%s",
            len(df.index),
            len(df.columns),
        )
        return df
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Falha ao ler XLSX")
        raise FileValidationError(
            "Não foi possível ler o ficheiro Excel. Confirme que é .xlsx válido "
            "(formato Office Open XML)."
        ) from exc


def load_dataframe(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    """
    Carrega CSV ou XLSX em DataFrame.

    Parameters
    ----------
    filename
        Nome original do ficheiro (para extensão).
    raw_bytes
        Conteúdo binário (ex.: file.getvalue() no Streamlit).
    """
    if not raw_bytes:
        raise FileValidationError("Ficheiro vazio.")
    kind = validate_upload_name(filename)
    if kind == "csv":
        return _read_csv_bytes(raw_bytes)
    return _read_xlsx_bytes(raw_bytes)

"""
Ingestão da DMS com deteção heurística da linha de cabeçalho e saneamento de colunas.

Motivação: relatórios exportados costumam trazer linhas de título, áreas em branco
e colunas geradas pelo Excel como ``Unnamed: N``.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any

import pandas as pd

from utils.file_io import FileValidationError, validate_upload_name

logger = logging.getLogger(__name__)


def _make_unique_columns(names: list[str]) -> list[str]:
    """Garante nomes únicos preservando o primeiro igual ("x", "x__2")."""
    counts: dict[str, int] = {}
    result: list[str] = []
    for label in names:
        base = label if label.strip() else "col_sem_nome"
        n = counts.get(base, 0) + 1
        counts[base] = n
        result.append(base if n == 1 else f"{base}__{n}")
    return result


def _strip_matrix(matrix: list[list[str]]) -> list[list[str]]:
    """Remove linhas inicial/final só com células vazias."""

    def row_empty(row: list[str]) -> bool:
        return all((c is None or str(c).strip() == "") for c in row)

    while matrix and row_empty(matrix[0]):
        matrix = matrix[1:]
    while matrix and row_empty(matrix[-1]):
        matrix = matrix[:-1]
    return matrix


def _normalize_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _matrix_from_excel_raw(raw: bytes) -> list[list[str]]:
    bio = io.BytesIO(raw)
    df = pd.read_excel(bio, engine="openpyxl", header=None, dtype=object)
    matrix: list[list[str]] = []
    for _, row in df.iterrows():
        matrix.append([_normalize_cell(v) for v in row.tolist()])
    return _strip_matrix(matrix)


def _guess_csv_sep(sample_lines: list[str]) -> str:
    """Escolhe ``;`` ou ``,`` com maior frequência média nas primeiras linhas."""
    nonempty = [
        ln for ln in sample_lines[:80] if ln.strip()
    ]
    if not nonempty:
        return ";"
    score_semi = 0
    score_comma = 0
    for ln in nonempty:
        score_semi += ln.count(";")
        quoted = False
        commas = 0
        for ch in ln:
            if ch == '"':
                quoted = not quoted
            elif ch == "," and not quoted:
                commas += 1
        score_comma += commas
    logger.info("CSV heurística: ; total=%s, separadores vírgula=%s", score_semi, score_comma)
    return ";" if score_semi >= score_comma else ","


def _decode_bytes(raw: bytes) -> tuple[str, str]:
    last_err: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text = raw.decode(encoding)
            logger.info("CSV decodificado com encoding=%s", encoding)
            return text, encoding
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    raise FileValidationError(
        "Não foi possível decodificar o CSV (tentados utf-8-sig, utf-8, latin-1, cp1252)."
    ) from last_err


def _matrix_from_csv_raw(raw: bytes) -> tuple[list[list[str]], str, str]:
    text, encoding = _decode_bytes(raw)
    lines = text.splitlines()
    nonempty_lines = [ln for ln in lines if ln.strip()]
    sep = _guess_csv_sep(nonempty_lines)
    reader = csv.reader(io.StringIO(text), delimiter=sep)
    matrix = [_strip_cells(row) for row in reader]
    matrix = _strip_matrix(matrix)
    return matrix, encoding, sep


def _strip_cells(row: list[str]) -> list[str]:
    out: list[str] = []
    for c in row:
        out.append(_normalize_cell(c))
    return out


def _score_header_candidate(row: list[str], idx: int) -> float:
    """
    Pontuação heurística: favorece linhas com várias células preenchidas
    relativamente curtas e penaliza uma única frase larga ("título" do relatório).
    """
    cells = [c for c in row if str(c).strip()]
    filled = len(cells)
    if filled <= 1:
        return min(-500.0 + filled, idx * -1.0)

    maxlen = max(len(str(c)) for c in cells)
    if maxlen > 180:
        return -350.0 + filled

    avg = sum(len(str(c)) for c in cells) / filled

    score = filled * 10.0 + 8.0
    score -= max(0.0, (maxlen - 60) * 0.05)
    if avg < 45:
        score += 12.0
    elif avg > 80:
        score -= 8.0

    numeric_like = sum(1 for c in cells if re.fullmatch(r"\d+[.,]?\d*", str(c)))
    if numeric_like == filled:
        score -= 20.0
    elif numeric_like == 0:
        score += 6.0
    score -= idx * 0.15
    return score


def detect_header_row_index(matrix: list[list[str]], max_scan: int = 45) -> int:
    """Índice 0‑based da linha escolhida como cabeçalho."""

    if not matrix:
        return 0
    span = min(max_scan, len(matrix))
    best_i = 0
    best_s = float("-inf")
    for i in range(span):
        s = _score_header_candidate(matrix[i], i)
        logger.debug(
            "Score cabeçalho candidato linha=%s score=%s amostra=%s",
            i,
            round(s, 2),
            matrix[i][:5],
        )
        if s > best_s:
            best_s = s
            best_i = i

    logger.info(
        "Cabeçalho DMS escolhido: linha %s (dentro das primeiras %s linhas não vazias).",
        best_i,
        span,
    )
    return best_i


def build_dataframe_from_matrix_with_header(matrix: list[list[str]], header_idx: int) -> pd.DataFrame:
    """Constrói DataFrame a partir da linha de cabeçalho."""

    data_rows = matrix[header_idx + 1 :]
    hdr_len = len(matrix[header_idx])
    max_cols = hdr_len
    for row in data_rows:
        max_cols = max(max_cols, len(row))

    header_cells = matrix[header_idx] + [""] * max(0, max_cols - hdr_len)
    raw_headers = [_clean_header_name(header_cells[j], j) for j in range(max_cols)]
    header = _make_unique_columns(raw_headers)

    rows: list[list[str]] = []
    for raw_row in data_rows:
        if not raw_row or all(not str(c).strip() for c in raw_row):
            continue
        rr = raw_row[:] + [""] * max(0, max_cols - len(raw_row))
        rr = rr[:max_cols]
        rows.append(rr)

    df = pd.DataFrame(rows, columns=pd.Index(header))
    df = df.replace("", pd.NA)
    return df


def _clean_header_name(name: object, col_index: int) -> str:
    s = _normalize_cell(name)
    if not s:
        return f"col_{col_index + 1}"
    lowered = str(s).lower().strip()
    if lowered.startswith("unnamed"):
        return f"col_{col_index + 1}"
    return s


def fix_unnamed_and_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uniformiza nomes ``Unnamed`` / repetidos / vazios; remove só colunas totalmente vazias."""

    cleaned = [_clean_header_name(str(df.columns[i]).strip(), i) for i in range(len(df.columns))]
    uniq = _make_unique_columns(cleaned)
    out = df.copy()
    out.columns = uniq
    out = out.dropna(axis=1, how="all")
    return out.fillna("").astype(str)


def load_dms_with_smart_header(filename: str, raw_bytes: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Lê CSV ou XLSX da DMS, ignora espaços antes da tabela, infere linha do cabeçalho.

    Parameters
    ----------
    filename
        Nome para inferir formato.
    raw_bytes
        Conteúdo binário upload.

    Returns
    -------
    DataFrame
        Dados já em ``str``.
    Meta
        Metadados de ingestão úteis para logs e auditoria manual.
    """
    if not raw_bytes:
        raise FileValidationError("Ficheiro vazio.")

    meta: dict[str, Any] = {"filename": filename}
    kind = validate_upload_name(filename)

    if kind == "csv":
        matrix, enc, sep = _matrix_from_csv_raw(raw_bytes)
        meta.update({"encoding": enc, "sep": sep})
    else:
        matrix = _matrix_from_excel_raw(raw_bytes)

    meta["rows_before_trim"] = len(matrix)
    matrix = _strip_matrix(matrix)
    meta["rows_after_trim"] = len(matrix)

    hdr = detect_header_row_index(matrix)
    meta["header_row_index"] = hdr
    df = build_dataframe_from_matrix_with_header(matrix, hdr)

    df = fix_unnamed_and_empty_columns(df)
    meta.update(
        {
            "n_rows": len(df.index),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
        }
    )
    logger.info(
        "DMS preparada: %s linhas, %s colunas; cabeçalho_linha=%s",
        meta["n_rows"],
        meta["n_cols"],
        hdr,
    )
    return df, meta

"""
Matching textual DMS × Censo via RapidFuzz com bloqueio por prefixo (bases grandes).

Cada linha DMS obtém o melhor candidato no Censo ≥ ``score_cutoff``, usando ``fuzz.WRatio``.
Fallback para lista completa do Censo quando o bucket por prefixo está vazio.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from utils.text_normalize import normalize_match_text

LOG = logging.getLogger(__name__)

BLOCK_PREFIX_LEN = 4
DEFAULT_BATCH_LOG_EVERY = 500


@dataclass(frozen=True)
class TextMatchSummary:
    linhas_dms: int
    linhas_censo: int
    encontrados: int
    sem_correspondencia: int
    score_min_usado: float
    tempo_segundos: float


def _blocking_key(norm: str) -> str:
    compact = "".join(ch for ch in norm if ch.isalnum())
    if not compact:
        return "__EMPTY__"
    return compact[:BLOCK_PREFIX_LEN].upper()


def _build_block_index(censo_norms: list[str]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, norm in enumerate(censo_norms):
        buckets[_blocking_key(norm)].append(idx)
    return dict(buckets)


def _pick_candidates(
    block_index: dict[str, list[int]],
    censo_count: int,
    norm_query: str,
) -> list[int]:
    key = _blocking_key(norm_query)
    cand = block_index.get(key)
    if cand:
        return cand
    LOG.debug(
        "Bucket vazio para prefixo=%s — fallback lista completa Censo (%s linhas).",
        key,
        censo_count,
    )
    return list(range(censo_count))


def run_textual_fuzzy_merge(
    dms_df: pd.DataFrame,
    censo_df: pd.DataFrame,
    col_dms_razao: str,
    col_censo_nome: str,
    score_cutoff: float,
    progress_callback: Callable[[float], None] | None = None,
    batch_log_every: int = DEFAULT_BATCH_LOG_EVERY,
) -> tuple[pd.DataFrame, TextMatchSummary]:
    """
    Produz um consolidado linha-a-linha da DMS com melhor linha do Censo por texto.

    Parameters
    ----------
    score_cutoff
        Pontuação mínima (0–100) para RapidFuzz ``WRatio``.
    """

    t0 = time.perf_counter()
    n_dms = len(dms_df.index)
    n_censo = len(censo_df.index)

    LOG.info(
        "Matching textual iniciado: DMS=%s linhas, Censo=%s linhas, cutoff=%s.",
        n_dms,
        n_censo,
        score_cutoff,
    )

    dms_norm = [normalize_match_text(v) for v in dms_df[col_dms_razao].tolist()]
    censo_norm = [normalize_match_text(v) for v in censo_df[col_censo_nome].tolist()]

    block_index = _build_block_index(censo_norm)
    LOG.info(
        "Índice de bloqueio criado: %s buckets distintos (prefix_len=%s).",
        len(block_index),
        BLOCK_PREFIX_LEN,
    )

    censo_records: list[dict[str, object]] = []
    scores: list[float | None] = [None] * n_dms
    statuses: list[str] = ["sem_correspondencia"] * n_dms
    censo_line_idx: list[Any | None] = [None] * n_dms
    censo_norm_match: list[str] = [""] * n_dms

    encontrados = 0
    amostras_log: list[str] = []

    for i in range(n_dms):
        query = dms_norm[i]
        if not query.strip():
            censo_records.append({f"censo__{c}": pd.NA for c in censo_df.columns})
        else:
            cand_idxs = _pick_candidates(block_index, n_censo, query)
            choices_sub = [censo_norm[j] for j in cand_idxs]

            match = process.extractOne(
                query,
                choices_sub,
                scorer=fuzz.WRatio,
                score_cutoff=float(score_cutoff),
            )

            if match is None:
                censo_records.append({f"censo__{c}": pd.NA for c in censo_df.columns})
            else:
                _choice_str, score, rel_idx = match
                censo_idx = cand_idxs[int(rel_idx)]
                encontrados += 1
                scores[i] = float(score)
                statuses[i] = "match_textual"
                censo_line_idx[i] = censo_df.index[censo_idx]
                censo_norm_match[i] = censo_norm[censo_idx]

                row_censo = censo_df.iloc[censo_idx]
                censo_records.append(
                    {f"censo__{str(c)}": row_censo[c] for c in censo_df.columns}
                )

                if len(amostras_log) < 5:
                    amostras_log.append(
                        f"linha_dms={i} score={score:.1f} | {query[:80]!r} => {censo_norm[censo_idx][:80]!r}"
                    )

        if progress_callback is not None:
            progress_callback((i + 1) / max(n_dms, 1))

        if batch_log_every and (i + 1) % batch_log_every == 0:
            LOG.info(
                "Matching textual progresso: %s / %s linhas DMS processadas.",
                i + 1,
                n_dms,
            )

    for linha in amostras_log:
        LOG.debug("Amostra match: %s", linha)

    censo_side = pd.DataFrame(censo_records)

    base = dms_df.copy().reset_index(drop=True)
    base.columns = [f"dms__{c}" for c in base.columns]

    consolidado = pd.concat(
        [
            base,
            censo_side.reset_index(drop=True),
            pd.Series(dms_norm, name="dms_texto_normalizado"),
            pd.Series(censo_norm_match, name="censo_texto_normalizado_match"),
            pd.Series(scores, name="similaridade_score"),
            pd.Series(statuses, name="match_status"),
            pd.Series(censo_line_idx, name="censo_indice_original"),
        ],
        axis=1,
    )

    elapsed = time.perf_counter() - t0
    summary = TextMatchSummary(
        linhas_dms=n_dms,
        linhas_censo=n_censo,
        encontrados=encontrados,
        sem_correspondencia=n_dms - encontrados,
        score_min_usado=float(score_cutoff),
        tempo_segundos=elapsed,
    )

    LOG.info(
        "Matching textual concluído em %.2fs: encontrados=%s sem_correspondência=%s.",
        elapsed,
        encontrados,
        summary.sem_correspondencia,
    )

    return consolidado, summary

"""
Merge determinístico DMS × Censo por CNPJ normalizado (14 dígitos).

O passe textual só deve correr sobre linhas onde a DMS não fornece CNPJ normalizável
(vazio ou classificado como inválido por ``classify_cnpj_cell``), evitando falsos positivos.

Estados típicos (``match_status_principal`` antes do texto):

- ``match_cnpj_exato`` — há exactamente uma escola municipal com esse CNPJ;
- ``sem_correspondencia_cnpj`` — CNPJ presente não existe nas escolas carregadas;
- ``multiplas_escolas_mesmo_cnpj`` — duplicação de CNPJ no lado Censo;
- ``sem_cnpj_utilizavel_dms`` — célula efetivamente vazia ou sem dígitos;
- ``cnpj_dms_invalido`` — dígitos incompatíveis com CNPJ 14 dígitos / checksum.

Após o texto complementar: ``match_textual_complementar`` ou ``sem_correspondencia_texto``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.cnpj import classify_cnpj_cell

LOG = logging.getLogger(__name__)

MATCH_CNPJ_EXATO = "match_cnpj_exato"
MATCH_MULTIPLAS_ESCOLAS = "multiplas_escolas_mesmo_cnpj"
SEM_CORRESP_CNPJ = "sem_correspondencia_cnpj"
SEM_CNPJ_DMS = "sem_cnpj_utilizavel_dms"
CNPJ_INVALIDO_DMS = "cnpj_dms_invalido"
MATCH_TEXTO_COMPLEMENTAR = "match_textual_complementar"
SEM_CORRESP_TEXTO = "sem_correspondencia_texto"

CONFIANCA_ALTA_CNPJ = "alta_conf_cnpj_exato"
CONFIANCA_DIVERGENCIA = "multiplicidade_cnpj_no_censo"
CONFIANCA_SEM_CHAVE_MERGE = "sem_chave_usavel"
CONFIANCA_MEDIA_TEXTO = "media_matching_textual"

MERGE_METODO_CNPJ = "cnpj_14_digitos"
MERGE_METODO_TEXTO = "texto_complementar"

ORDEM_COLUMN = "merge_linha_ordem"


@dataclass(frozen=True)
class CNPJDeterministicSummary:
    linhas_dms: int
    linhas_censo: int
    match_cnpj_exato: int
    multiplas_escolas_mesmo_cnpj: int
    sem_correspondencia_cnpj: int
    sem_cnpj_dms: int
    cnpj_dms_invalido: int
    chaves_com_multiplos_cnpj_no_censo: int
    tempo_segundos: float


@dataclass(frozen=True)
class ComplementaryTextSummary:
    linhas_elegiveis: int
    matches_texto: int
    sem_correspondencia: int
    score_cutoff_usado: float


def deterministic_merge_by_cnpj(
    dms_df: pd.DataFrame,
    censo_df: pd.DataFrame,
    *,
    col_dms_raw_cnpj: str,
    col_dms_norm: str = "__cnpj_norm_dms",
    col_censo_norm: str = "__cnpj_norm_censo",
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[pd.DataFrame, CNPJDeterministicSummary]:
    t0 = time.perf_counter()

    dm = dms_df.copy().reset_index(drop=True)
    cen = censo_df.copy().reset_index(drop=True)

    if col_dms_raw_cnpj not in dm.columns:
        raise ValueError(f"Coluna DMS esperada ausente: {col_dms_raw_cnpj!r}.")
    if col_dms_norm not in dm.columns:
        raise ValueError(f"Finalize a Etapa 2 — falta `{col_dms_norm}`.")

    n_dms = len(dm.index)
    n_censo = len(cen.index)
    dm[ORDEM_COLUMN] = np.arange(n_dms, dtype=int)

    norm_series = dm[col_dms_norm].astype(str).str.strip()
    chave_usavel = norm_series.str.len().eq(14) & norm_series.str.isdigit()
    lab_bruto = dm[col_dms_raw_cnpj].map(classify_cnpj_cell)

    censo_key_counts: dict[str, int] = {}
    chaves_dup_total = 0
    lookup_first = pd.DataFrame(columns=list(cen.columns), dtype=object)

    if col_censo_norm in cen.columns:
        nz = cen[col_censo_norm].astype(str).str.strip().ne("")
        cen_k_rows = cen.loc[nz].copy()
        if not cen_k_rows.empty:
            gs = cen_k_rows.groupby(col_censo_norm, dropna=False, sort=False)
            censo_key_counts = gs.size().to_dict()
            chaves_dup_total = int((gs.size() > 1).sum())
            lookup_first = gs.head(1).reset_index(drop=True).set_index(col_censo_norm)

    censo_wide: pd.DataFrame
    if lookup_first.empty or col_censo_norm not in cen.columns:
        censo_wide = pd.DataFrame(index=np.arange(n_dms), columns=list(cen.columns))
        censo_wide = censo_wide.astype(object)
        censo_wide.loc[:, :] = pd.NA
    else:
        censo_wide = lookup_first.reindex(norm_series.fillna("").tolist()).reset_index(drop=True)

    censo_records: list[dict[str, object]] = []
    for i in range(n_dms):
        row = censo_wide.iloc[i]
        censo_records.append({f"censo__{str(col)}": row[col] for col in cen.columns})

    statuses: list[str] = []
    confidences: list[str] = []
    match_counts_series: list[float] = []
    duplicate_label_series: list[object] = []

    for i in range(n_dms):
        lab = lab_bruto.iat[i]
        usable = bool(chave_usavel.iat[i])
        key = norm_series.iat[i]

        if not usable:
            statuses.append(SEM_CNPJ_DMS if lab == "empty" else CNPJ_INVALIDO_DMS)
            confidences.append(CONFIANCA_SEM_CHAVE_MERGE)
            match_counts_series.append(0.0)
            duplicate_label_series.append(pd.NA)
        else:
            mc = censo_key_counts.get(key, 0)
            match_counts_series.append(float(mc))
            if mc <= 0:
                statuses.append(SEM_CORRESP_CNPJ)
                confidences.append(CONFIANCA_SEM_CHAVE_MERGE)
                duplicate_label_series.append(pd.NA)
            elif mc == 1:
                statuses.append(MATCH_CNPJ_EXATO)
                confidences.append(CONFIANCA_ALTA_CNPJ)
                duplicate_label_series.append(pd.NA)
            else:
                statuses.append(MATCH_MULTIPLAS_ESCOLAS)
                confidences.append(CONFIANCA_DIVERGENCIA)
                duplicate_label_series.append(int(mc))

        if progress_callback is not None:
            progress_callback((i + 1) / max(n_dms, 1))

    base_pref = dm.add_prefix("dms__")

    placeholders = pd.DataFrame(
        {
            "similaridade_score": pd.Series([pd.NA] * n_dms),
            "dms_texto_normalizado": "",
            "censo_texto_normalizado_match": "",
            "censo_indice_original": pd.Series([pd.NA] * n_dms, dtype=object),
        },
    )

    merge_metodos: list[str] = []
    for stv in statuses:
        if str(stv) in {SEM_CNPJ_DMS, CNPJ_INVALIDO_DMS}:
            merge_metodos.append("")
        else:
            merge_metodos.append(MERGE_METODO_CNPJ)
    metodo_prim = pd.Series(merge_metodos, index=np.arange(n_dms), dtype=object)

    out = pd.concat(
        [
            base_pref.reset_index(drop=True),
            pd.DataFrame(censo_records),
            metodo_prim.rename("merge_metodo_primario"),
            pd.Series(statuses, name="match_status_principal"),
            pd.Series(confidences, name="merge_confianca"),
            pd.Series(match_counts_series, name="cnpj_censo_candidatos_mesmo_numero"),
            pd.Series(duplicate_label_series, name="censo_escolas_duplicate_count_para_chave"),
            placeholders.reset_index(drop=True),
        ],
        axis=1,
    )

    elapsed = time.perf_counter() - t0

    summary = CNPJDeterministicSummary(
        linhas_dms=n_dms,
        linhas_censo=n_censo,
        match_cnpj_exato=statuses.count(MATCH_CNPJ_EXATO),
        multiplas_escolas_mesmo_cnpj=statuses.count(MATCH_MULTIPLAS_ESCOLAS),
        sem_correspondencia_cnpj=statuses.count(SEM_CORRESP_CNPJ),
        sem_cnpj_dms=statuses.count(SEM_CNPJ_DMS),
        cnpj_dms_invalido=statuses.count(CNPJ_INVALIDO_DMS),
        chaves_com_multiplos_cnpj_no_censo=chaves_dup_total,
        tempo_segundos=elapsed,
    )

    LOG.info(
        "Determinístico CNPJ: exatos=%s sem_corr=%s multi=%s sem_cnpj=%s invalid_dms=%s | chaves censo repetidas=%s | %.3fs",
        summary.match_cnpj_exato,
        summary.sem_correspondencia_cnpj,
        summary.multiplas_escolas_mesmo_cnpj,
        summary.sem_cnpj_dms,
        summary.cnpj_dms_invalido,
        summary.chaves_com_multiplos_cnpj_no_censo,
        elapsed,
    )

    return out, summary


def merge_status_qualifies_textual_complement(status_principal: object) -> bool:
    return str(status_principal) in {SEM_CNPJ_DMS, CNPJ_INVALIDO_DMS}


def stitch_complementary_textual_into_base(
    base_df: pd.DataFrame,
    fuzzy_result: pd.DataFrame,
    *,
    score_cutoff_used: float,
) -> tuple[pd.DataFrame, ComplementaryTextSummary]:
    ord_col_pref = f"dms__{ORDEM_COLUMN}"

    if ord_col_pref not in fuzzy_result.columns:
        raise ValueError(f"Esperado {ord_col_pref!r} no resultado textual para alinhar as linhas.")
    if ord_col_pref not in base_df.columns:
        raise ValueError(f"Esperado {ord_col_pref!r} na base antes do passe textual.")

    updated = base_df.copy()
    base_ord_mask = pd.to_numeric(updated[ord_col_pref], errors="coerce")

    texto_matches = 0
    textual_sem_match = 0

    censo_pref_cols_fuzzy = [str(c) for c in fuzzy_result.columns if str(c).startswith("censo__")]

    for pos in range(len(fuzzy_result.index)):
        ordem_linha = pd.to_numeric(fuzzy_result[ord_col_pref].iloc[pos], errors="coerce")
        if pd.isna(ordem_linha):
            continue
        ordem_linha = int(ordem_linha)
        ridx_candidates = updated.index[(base_ord_mask == ordem_linha).to_numpy()].tolist()
        if not ridx_candidates:
            LOG.warning("Passe texto: orden %s não encontrada na base — ignorando.", ordem_linha)
            continue
        ridx = ridx_candidates[0]
        fz_row = fuzzy_result.iloc[pos]

        for cc in censo_pref_cols_fuzzy:
            if cc in fz_row.index and cc in updated.columns:
                updated.at[ridx, cc] = fz_row[cc]

        for extras in ("similaridade_score", "dms_texto_normalizado", "censo_texto_normalizado_match"):
            if extras in fz_row.index and extras in updated.columns:
                updated.at[ridx, extras] = fz_row[extras]

        raw_status_fz = str(fz_row.get("match_status", ""))
        if raw_status_fz == "match_textual":
            updated.at[ridx, "match_status_principal"] = MATCH_TEXTO_COMPLEMENTAR
            updated.at[ridx, "merge_confianca"] = CONFIANCA_MEDIA_TEXTO
            updated.at[ridx, "merge_metodo_primario"] = MERGE_METODO_TEXTO
            texto_matches += 1
        else:
            updated.at[ridx, "match_status_principal"] = SEM_CORRESP_TEXTO
            textual_sem_match += 1
            updated.at[ridx, "merge_confianca"] = CONFIANCA_SEM_CHAVE_MERGE
            updated.at[ridx, "merge_metodo_primario"] = MERGE_METODO_TEXTO

    n_eligible = len(fuzzy_result.index)
    cs = ComplementaryTextSummary(
        linhas_elegiveis=int(n_eligible),
        matches_texto=texto_matches,
        sem_correspondencia=textual_sem_match,
        score_cutoff_usado=float(score_cutoff_used),
    )
    return updated, cs


STATUSES_PRIMARIOS_CNPJ_ONLY = frozenset(
    {
        MATCH_CNPJ_EXATO,
        MATCH_MULTIPLAS_ESCOLAS,
        SEM_CORRESP_CNPJ,
        SEM_CNPJ_DMS,
        CNPJ_INVALIDO_DMS,
    }
)

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
from typing import Any

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


def compute_merge_debug_snapshot(
    dms_df: pd.DataFrame,
    censo_df: pd.DataFrame,
    *,
    col_dms_norm: str = "__cnpj_norm_dms",
    col_censo_norm: str = "__cnpj_norm_censo",
) -> dict[str, Any]:
    """
    Métricas para diagnóstico antes/durante falhas no merge: formas, CNPJs únicos (14 dígitos),
    contagem de chaves com mais do que uma linha (duplicidade no mesmo lado).
    """

    snap: dict[str, Any] = {
        "dms_shape": tuple(dms_df.shape),
        "censo_shape": tuple(censo_df.shape),
        "dms_col_norm_presente": col_dms_norm in dms_df.columns,
        "censo_col_norm_presente": col_censo_norm in censo_df.columns,
        "dms_cnpj_unicos_14_digitos": None,
        "dms_chaves_com_mais_de_uma_linha": None,
        "censo_cnpj_unicos_nao_vazio": None,
        "censo_chaves_com_mais_de_uma_linha": None,
    }
    if col_dms_norm in dms_df.columns:
        s = dms_df[col_dms_norm].astype(str).str.strip()
        ok = s.str.len().eq(14) & s.str.isdigit()
        keys = s.loc[ok]
        snap["dms_cnpj_unicos_14_digitos"] = int(keys.nunique())
        vc = keys.value_counts()
        snap["dms_chaves_com_mais_de_uma_linha"] = int((vc > 1).sum())
    if col_censo_norm in censo_df.columns:
        s = censo_df[col_censo_norm].astype(str).str.strip()
        nz = s.ne("")
        keys_c = s.loc[nz]
        snap["censo_cnpj_unicos_nao_vazio"] = int(keys_c.nunique())
        vc = keys_c.value_counts()
        snap["censo_chaves_com_mais_de_uma_linha"] = int((vc > 1).sum())
    return snap


def _log_merge_debug_snapshot(where: str, snap: dict[str, Any]) -> None:
    LOG.info(
        "%s — snapshot merge: dms_shape=%s censo_shape=%s %s",
        where,
        snap.get("dms_shape"),
        snap.get("censo_shape"),
        {k: v for k, v in snap.items() if k not in ("dms_shape", "censo_shape")},
    )


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

    def _phase_fail(phase: str, exc: BaseException) -> None:
        LOG.exception(
            "deterministic_merge_by_cnpj — exceção na fase %r (causa original preservada em __cause__)",
            phase,
        )

    snap0 = compute_merge_debug_snapshot(
        dms_df, censo_df, col_dms_norm=col_dms_norm, col_censo_norm=col_censo_norm
    )
    _log_merge_debug_snapshot("deterministic_merge_by_cnpj [entrada]", snap0)

    try:
        dm = dms_df.copy().reset_index(drop=True)
        cen = censo_df.copy().reset_index(drop=True)
    except Exception as e:
        _phase_fail("cópia_reset_index", e)
        raise RuntimeError(
            "deterministic_merge_by_cnpj falhou na fase: cópia_reset_index (DataFrame.copy / reset_index)"
        ) from e

    try:
        if col_dms_raw_cnpj not in dm.columns:
            raise ValueError(f"Coluna DMS esperada ausente: {col_dms_raw_cnpj!r}.")
        if col_dms_norm not in dm.columns:
            raise ValueError(f"Finalize a Etapa 2 — falta `{col_dms_norm}`.")
        if col_censo_norm not in cen.columns:
            raise ValueError(
                f"Censo de trabalho sem `{col_censo_norm}`. Esta coluna deve ser criada antes do merge "
                f"determinístico com `utils.cnpj.add_normalized_cnpj_column` sobre a coluna física CNPJ municipal."
            )

        n_dms = len(dm.index)
        n_censo = len(cen.index)
        dm[ORDEM_COLUMN] = np.arange(n_dms, dtype=int)

        norm_series = dm[col_dms_norm].astype(str).str.strip()
        chave_usavel = norm_series.str.len().eq(14) & norm_series.str.isdigit()
        lab_bruto = dm[col_dms_raw_cnpj].map(classify_cnpj_cell)
        LOG.info(
            "deterministic_merge_by_cnpj — após preparação: n_dms=%s n_censo=%s linhas_com_chave_dms_14d=%s",
            n_dms,
            n_censo,
            int(chave_usavel.sum()),
        )
    except Exception as e:
        _phase_fail("validação_preparação_classify", e)
        raise RuntimeError(
            "deterministic_merge_by_cnpj falhou na fase: validação_preparação_classify"
        ) from e

    censo_key_counts: dict[str, int] = {}
    chaves_dup_total = 0
    lookup_first = pd.DataFrame(columns=list(cen.columns), dtype=object)

    try:
        LOG.info("deterministic_merge_by_cnpj — antes filtro censo (chave normalizada não vazia)")
        if col_censo_norm in cen.columns:
            nz = cen[col_censo_norm].astype(str).str.strip().ne("")
            cen_k_rows = cen.loc[nz].copy()
            LOG.info(
                "deterministic_merge_by_cnpj — após filtro: linhas_censo_com_chave=%s de %s",
                len(cen_k_rows.index),
                len(cen.index),
            )
            if not cen_k_rows.empty:
                LOG.info("deterministic_merge_by_cnpj — antes groupby(%r)", col_censo_norm)
                gs = cen_k_rows.groupby(col_censo_norm, dropna=False, sort=False)
                censo_key_counts = gs.size().to_dict()
                chaves_dup_total = int((gs.size() > 1).sum())
                LOG.info(
                    "deterministic_merge_by_cnpj — após groupby: n_chaves_distintas=%s chaves_censo_com_mais_de_uma_linha=%s",
                    len(censo_key_counts),
                    chaves_dup_total,
                )
                lookup_first = gs.head(1).reset_index(drop=True).set_index(col_censo_norm)

        LOG.info("deterministic_merge_by_cnpj — antes alinhamento (lookup/reindex → censo_wide)")
        censo_wide: pd.DataFrame
        if lookup_first.empty or col_censo_norm not in cen.columns:
            censo_wide = pd.DataFrame(index=np.arange(n_dms), columns=list(cen.columns))
            censo_wide = censo_wide.astype(object)
            censo_wide.loc[:, :] = pd.NA
        else:
            # lookup_first foi criado com set_index(col_censo_norm): a chave está no índice, não nas colunas.
            # reindex(...) seguido de reset_index(drop=True) **eliminava** a chave — gerando KeyError ao pedir row[col_censo_norm].
            aligned = lookup_first.reindex(norm_series.fillna("").tolist())
            censo_wide = aligned.reset_index()
            censo_wide = censo_wide.reindex(columns=list(cen.columns))
    except Exception as e:
        _phase_fail("filtro_groupby_alinhamento", e)
        raise RuntimeError(
            "deterministic_merge_by_cnpj falhou na fase: filtro_groupby_alinhamento (filtro | groupby | reindex)"
        ) from e

    try:
        LOG.info(
            "deterministic_merge_by_cnpj — antes materialização censo_records (linhas DMS=%s)",
            n_dms,
        )
        LOG.info(
            "deterministic_merge_by_cnpj — materialização: colunas schema cen.columns=%s",
            list(map(str, cen.columns)),
        )
        LOG.info(
            "deterministic_merge_by_cnpj — materialização: colunas censo_wide.columns=%s",
            list(map(str, censo_wide.columns)),
        )
        if col_censo_norm not in censo_wide.columns:
            LOG.warning(
                "deterministic_merge_by_cnpj — após alinhamento ainda falta %r em censo_wide (usará pd.NA nas células).",
                col_censo_norm,
            )
        if n_dms > 0:
            zrow = censo_wide.iloc[0]
            LOG.info(
                "deterministic_merge_by_cnpj — exemplo linha 0 (Series.iloc[0]): row.index=%s",
                list(map(str, zrow.index)),
            )

        censo_records: list[dict[str, object]] = []
        for i in range(n_dms):
            row = censo_wide.iloc[i]
            censo_records.append(
                {f"censo__{str(col)}": row.get(col, pd.NA) for col in cen.columns},
            )
    except Exception as e:
        _phase_fail("materialização_censo_por_linha_dms", e)
        raise RuntimeError(
            "deterministic_merge_by_cnpj falhou na fase: materialização_censo_por_linha_dms"
        ) from e

    try:
        LOG.info("deterministic_merge_by_cnpj — antes classificação (status por linha DMS, n=%s)", n_dms)
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
    except Exception as e:
        _phase_fail("classificação_status", e)
        raise RuntimeError(
            "deterministic_merge_by_cnpj falhou na fase: classificação_status"
        ) from e

    try:
        LOG.info("deterministic_merge_by_cnpj — antes concat (montagem DataFrame final)")
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
    except Exception as e:
        _phase_fail("concat_montagem_saida", e)
        raise RuntimeError(
            "deterministic_merge_by_cnpj falhou na fase: concat_montagem_saida"
        ) from e

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

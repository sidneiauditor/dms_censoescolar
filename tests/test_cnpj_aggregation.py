"""Testes — filtro mensal DMS (Etapa 8.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from services.cnpj_aggregation import (
    AGG_MC,
    AGG_QTY,
    CNPJ_NORM_COL_CENSO,
    CNPJ_NORM_COL_DMS,
    aggregate_census_by_cnpj,
    aggregate_dms_by_cnpj,
    filter_censo_for_fiscal_panel,
    filter_dms_to_reference_month,
)


def _row(cnpj: str, comp: str, qty: float, tipo: str = "NORMAL", sit: str = "ATIVA") -> dict:
    return {
        CNPJ_NORM_COL_DMS: cnpj,
        "DTCOMPETENCIA": comp,
        "QUANTIDADE": qty,
        "TIPO": tipo,
        "SITUACAO": sit,
    }


def test_filter_uses_maio_when_available():
    df = pd.DataFrame(
        [
            _row("28580065001810", "2025-04-01", 10),
            _row("28580065001810", "2025-05-01", 100),
            _row("28580065001810", "2025-06-01", 200),
        ]
    )
    out, meta = filter_dms_to_reference_month(df, reference_year=2025, reference_month=5)
    assert len(out) == 1
    assert float(out["QUANTIDADE"].iloc[0]) == 100.0
    assert meta["cnpjs_com_maio"] == 1
    assert meta["cnpjs_com_fallback"] == 0


def test_filter_fallback_to_abril():
    df = pd.DataFrame(
        [
            _row("28580065004593", "2025-04-01", 50),
            _row("28580065004593", "2025-06-01", 999),
        ]
    )
    out, meta = filter_dms_to_reference_month(df, reference_year=2025, reference_month=5)
    assert len(out) == 1
    assert float(out["QUANTIDADE"].iloc[0]) == 50.0
    assert meta["cnpjs_com_maio"] == 0
    assert meta["cnpjs_com_fallback"] == 1
    fb = meta["fallback_detail"]
    assert fb.iloc[0]["mes_referencia_usado"] == 4


def test_filter_excludes_cnpj_without_valid_month():
    df = pd.DataFrame(
        [
            _row("28580065001900", "2025-06-01", 10),
        ]
    )
    out, meta = filter_dms_to_reference_month(df, reference_year=2025, reference_month=5)
    assert out.empty
    assert meta["cnpjs_sem_competencia"] == 1


def test_filter_without_competencia_column_returns_full_df():
    df = pd.DataFrame(
        {
            CNPJ_NORM_COL_DMS: ["28580065001810"],
            "QUANTIDADE": [42],
        }
    )
    out, meta = filter_dms_to_reference_month(df, reference_year=2025)
    assert len(out) == 1
    assert meta["coluna_competencia_nao_encontrada"] is True


def test_filter_skips_retificadora_and_cancelada():
    df = pd.DataFrame(
        [
            _row("28580065001810", "2025-05-01", 1, tipo="RETIFICADORA"),
            _row("28580065001810", "2025-05-01", 2, sit="CANCELADA"),
            _row("28580065001810", "2025-04-01", 40),
        ]
    )
    out, meta = filter_dms_to_reference_month(df, reference_year=2025, reference_month=5)
    assert len(out) == 1
    assert float(out["QUANTIDADE"].iloc[0]) == 40.0
    assert meta["cnpjs_com_fallback"] == 1


def test_aggregate_dms_monthly_vs_annual():
    df = pd.DataFrame(
        [
            _row("28580065001810", "2025-05-01", 100),
            _row("28580065001810", "2025-01-01", 10),
            _row("28580065001810", "2025-03-01", 10),
        ]
    )
    monthly, _ = aggregate_dms_by_cnpj(df, use_reference_month=True, reference_year=2025)
    annual, _ = aggregate_dms_by_cnpj(df, use_reference_month=False)
    assert float(monthly[AGG_QTY].iloc[0]) == 100.0
    assert float(annual[AGG_QTY].iloc[0]) == 120.0


def test_aggregate_census_only_private():
    df = pd.DataFrame(
        {
            CNPJ_NORM_COL_CENSO: ["11111111000191", "22222222000100", "33333333000177"],
            "dependencia_administrativa": [3, 4, 4],
            "QT_MAT_BAS": [100, 50, 0],
        }
    )
    all_agg = aggregate_census_by_cnpj(df, only_private=False, exclude_superior_puro=False)
    priv_agg = aggregate_census_by_cnpj(df, only_private=True, exclude_superior_puro=False)
    priv_eb = aggregate_census_by_cnpj(df, only_private=True, exclude_superior_puro=True)
    assert len(all_agg) == 3
    assert len(priv_agg) == 2
    assert len(priv_eb) == 1
    assert float(priv_eb[AGG_MC].iloc[0]) == 50.0


def test_filter_censo_logical_dependencia_column():
    df = pd.DataFrame(
        {
            "dependencia_administrativa": ["3", "4"],
            "QT_MAT_BAS": [10, 20],
        }
    )
    out, meta = filter_censo_for_fiscal_panel(df, only_private=True, exclude_superior_puro=False)
    assert len(out) == 1
    assert meta["n_publicas_excluidas"] == 1
    assert meta["dependencia_col"] == "dependencia_administrativa"

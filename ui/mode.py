"""Modo de uso: Operacional (CIF Salvador) vs Técnico (fluxo desenvolvedor)."""

from __future__ import annotations

import streamlit as st

SESSION_OPERACIONAL = "cif_is_modo_operacional"
AUTO_PROCESS_KEY = "cif_pref_auto_process_when_three_files"

MODO_OPERACIONAL_LABEL = "Operacional (Salvador — default)"
MODO_TECNICO_LABEL = "Técnico / diagnóstico completo"


def ensure_default_modo_operacional() -> None:
    if SESSION_OPERACIONAL not in st.session_state:
        st.session_state[SESSION_OPERACIONAL] = True


def is_modo_operacional() -> bool:
    ensure_default_modo_operacional()
    return bool(st.session_state[SESSION_OPERACIONAL])


def modo_sidebar_radio() -> bool:
    """Seleção na sidebar; devolve ``True`` se modo Operacional."""

    idx = 0 if is_modo_operacional() else 1
    choice = st.sidebar.radio(
        "Modo",
        options=[MODO_OPERACIONAL_LABEL, MODO_TECNICO_LABEL],
        index=idx,
        key="cif_modo_sidebar_widget",
        help="**Operacional** — Salvador BA/2927408, painel primeiro. **Técnico** — fluxo íntegro.",
    )
    sel = choice == MODO_OPERACIONAL_LABEL
    st.session_state[SESSION_OPERACIONAL] = sel
    return sel


def salvador_sidebar_contexto_ui(exercise_year: int) -> None:
    st.sidebar.caption(
        f"Território: **Salvador (BA)** · IBGE **2927408** · ano **{exercise_year}**."
    )


def marcar_preset_salvador_sessao() -> None:
    st.session_state["cif_contexto_preset"] = "salvador_ba_2927408"

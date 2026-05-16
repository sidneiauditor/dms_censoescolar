"""
Fluxo técnico completo (todas as etapas, diagnósticos e mapeamentos).

A orquestração das funções existentes continua em ``app.py`` para evitar ciclos de import
enquanto o monólito é migrado por fases. Este módulo concentra constantes e futuros
componentes só do modo **Técnico**.
"""

from __future__ import annotations

# Expander aberto por defeito no modo técnico (comportamento histórico).
EXPANDER_TECNICO_ABERTO_PADRAO = True

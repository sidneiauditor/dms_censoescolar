"""
App local Streamlit — cruzamento DMS-Educação × Censo Escolar.

Arquitetura (UX atual):
- **Etapa / Carregamento** — uploads DMS, Censo Escola e opcionalmente Matrícula (qualquer nome de ficheiro).
- **Modo simples** — pré-mapeamento automático pelos aliases INEP/export; Etapa 0 (exercício + UF +
  município) corta Escola antes do merge; menos opções na UI técnica.
- **Modo avançado** — mapeamento lógico explícito (todas as colunas) + opção de desativar filtro municipal.
- **Consolidação** — apenas escolas do município (por defeito) ⊕ Matrícula recortada ao mesmo conjunto de
  ``CO_ENTIDADE`` quando possível; colunas lógicas estáveis no resultado.
- **Etapa 2** — vínculo DMS ↔ Censo municipal já “pensado”: CNPJ, nome, matrículas onde existirem.
- **Etapa 3** — matching textual RapidFuzz entre DMS e Censo consolidado.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from domain.census_logical import (
    CENSO_ESCOLA_FIELDS,
    CENSO_MATRICULA_FIELDS,
    LogicalFieldSpec,
)
from domain.dataset_kind import DatasetKind, label as dataset_kind_label
from services.census_consolidator import (
    CensusMergeError,
    consolidate_census_escolar,
    normalize_co_entidade,
)
from services.inferred_mapping import propose_dms_mapping, propose_escola_mapping, propose_matricula_mapping
from services.municipality_filter import filter_escola_by_municipality, restrict_matricula_to_entidades
from services.table_loader import load_dataset_bundle, spinner_message
from services.text_fuzzy_merge import run_textual_fuzzy_merge
from utils.cnpj import add_normalized_cnpj_column, summarize_cnpj_column
from utils.file_io import FileValidationError

APP_DIR = Path(__file__).resolve().parent
SELECT_SENTINEL = "-- Selecionar coluna --"

UX_SIMPLES = "Simples (recomendado) — município + automático"
UX_AVANCADO = "Avançado — mapeamento manual e diagnósticos"

LOG = logging.getLogger(__name__)


def configure_logging() -> None:
    """Logs para ficheiro (DEBUG) e consola (INFO). Evita handlers duplicados em reruns Streamlit."""

    root = logging.getLogger()
    if root.handlers:
        return

    log_dir = APP_DIR / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)


def _friendly_file_error(where: str, err: BaseException) -> None:
    st.error(f"**{where}**: não foi possível utilizar este ficheiro.")
    if isinstance(err, FileValidationError):
        st.warning(str(err))
        LOG.warning("Validação falhou [%s]: %s", where, err)
        return
    st.warning(
        "Ocorreu um erro inesperado. Sugestões:\n\n"
        "- Confirme se o formato é CSV ou Excel (.xlsx) íntegro.\n\n"
        "- Para CSV grave em UTF‑8 ou Windows‑1252.\n\n"
        "- Experimente remover palavras‑passe de folha Excel."
    )
    LOG.exception("Erro ao processar [%s]", where)
    with st.expander("Detalhe técnico (equipa técnica)"):
        st.code(str(err))


def _preview_dataframe(title: str, df: pd.DataFrame, caption: str) -> None:
    st.markdown(f"**{title}** — {caption}")
    st.dataframe(df, use_container_width=True, height=320)


def column_options(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return [SELECT_SENTINEL]
    return [SELECT_SENTINEL] + [str(c) for c in df.columns]


def collect_logical_mapping(prefix_key: str, specs: tuple[LogicalFieldSpec, ...]) -> dict[str, str]:
    """Lê ``st.session_state`` preenchido pelos ``selectbox`` de mapeamento lógico."""

    result: dict[str, str] = {}
    for spec in specs:
        val = st.session_state.get(f"{prefix_key}_{spec.key}", SELECT_SENTINEL)
        if val != SELECT_SENTINEL:
            result[spec.key] = val
    return result


def resolve_escola_mapping(cols_escola: list[str], *, ui_simples: bool) -> dict[str, str]:
    """Junta inferência automática com overrides do modo avançado (widgets já renderizados)."""

    proposals = propose_escola_mapping(list(cols_escola))
    if ui_simples:
        return proposals

    merged: dict[str, str] = dict(proposals)
    widgets = collect_logical_mapping("logical_escola", CENSO_ESCOLA_FIELDS)
    for logical_key, physical in widgets.items():
        if physical and physical != SELECT_SENTINEL and physical in cols_escola:
            merged[logical_key] = physical
    return merged


def resolve_matricula_mapping(cols_mat: list[str] | None, *, ui_simples: bool) -> dict[str, str]:
    if not cols_mat:
        return {}
    proposals = propose_matricula_mapping(list(cols_mat))
    if ui_simples:
        return proposals
    merged: dict[str, str] = dict(proposals)
    widgets = collect_logical_mapping("logical_matricula", CENSO_MATRICULA_FIELDS)
    for logical_key, physical in widgets.items():
        if physical and physical != SELECT_SENTINEL and physical in cols_mat:
            merged[logical_key] = physical
    return merged


def _sanitize_municipio_codigo_cell(value: object) -> str:
    texto = "" if pd.isna(value) else str(value).strip()
    if texto.endswith(".0") and texto.replace(".0", "").isdigit():
        texto = texto[:-2]
    return texto


def render_etapa0_contexto_municipal(
    df_escola: pd.DataFrame,
    map_geo: dict[str, str],
    *,
    ui_simples: bool,
) -> dict[str, Any]:
    """
    Etapa 0 — exercício + UF + município. ``map_geo`` deve refletir o mapeamento **resolvido** actual
    (automático ou manual) para ``SG_UF`` / ``CO_MUNICIPIO`` / ``NO_MUNICIPIO``.
    """

    st.divider()
    st.header("Etapa 0 — Contexto municipal")
    if ui_simples:
        st.caption(
            "Indique o ano de referência e o **município**. Só ficam escolas dessa cidade no Censo "
            "**antes** da junção — processamento mais leve."
        )
    else:
        st.caption(
            "Igual ao modo simples, com opções extra: pode **desligar** o recorte territorial se o ficheiro "
            "já estiver pré-filtrado ou não usar colunas típicas INEP."
        )

    exercise_default = int(st.session_state.get("ctx_exercise_default", 2025))
    ex = st.number_input(
        "Exercício do Censo (ano de referência)",
        min_value=1996,
        max_value=2050,
        value=exercise_default,
        step=1,
        key="ctx_exercise_year",
        help='Aparece em ``censo_exercicio`` na base consolidada.',
    )
    st.session_state["ctx_exercise_default"] = int(ex)

    uf_phys = map_geo.get("SG_UF")
    co_phys = map_geo.get("CO_MUNICIPIO")
    no_phys = map_geo.get("NO_MUNICIPIO")

    skip_geo = False
    if not ui_simples:
        skip_geo = st.checkbox(
            "Não aplicar filtro municipal (usar todas as linhas da Escola carregada)",
            value=bool(st.session_state.get("ctx_skip_municipality", False)),
            key="ctx_skip_municipality",
        )

    out: dict[str, Any] = {
        "exercise": int(ex),
        "uf": None,
        "mun_code": None,
        "mun_label": "",
        "skip_geo": bool(skip_geo),
        "filtro_ativo": False,
        "filtro_impossivel_geo": False,
    }

    if skip_geo:
        st.info(
            "**Filtro municipal desativado.** Será usado o conjunto inteiro da tabela Escola em memória — "
            "único cenário válido quando o arquivo já está recortado ou não há UF/município."
        )
        return out

    if uf_phys is None or co_phys is None:
        out["filtro_impossivel_geo"] = True
        st.warning(
            "As colunas físicas típicas de **UF / município** não foram encontradas nem mapeadas.\n\n"
            "- Modo simples: confirme se o ficheiro é o microdados **Escola** INEP (com ``SG_UF`` / ``CO_MUNICIPIO``).\n"
            "- Modo avançado: associe explicitamente esses dois campos lógicos no mapeamento **ou** desative o filtro acima."
        )
        return out

    if uf_phys not in df_escola.columns or co_phys not in df_escola.columns:
        out["filtro_impossivel_geo"] = True
        st.error("As colunas de localização definidas pelo mapeamento **não existem** nesta tabela Escola.")
        return out

    ufs = df_escola[uf_phys].dropna().astype(str).str.strip().str.upper().unique()
    ufs_sorted = sorted(u for u in ufs if u)
    if not ufs_sorted:
        st.warning("Sem valores de UF na coluna física configurada.")
        out["filtro_impossivel_geo"] = True
        return out

    uf_sel = st.selectbox("UF", ufs_sorted, key="ctx_uf_select")
    out["uf"] = uf_sel

    sub_mask = df_escola[uf_phys].astype(str).str.strip().str.upper() == str(uf_sel).strip().upper()
    sub = df_escola.loc[sub_mask]
    use_name = bool(no_phys and no_phys in df_escola.columns)

    labels: list[str] = []
    label_to_code: dict[str, str] = {}
    for _, row in sub[[co_phys] + ([no_phys] if use_name else [])].drop_duplicates().iterrows():
        code_raw = row[co_phys]
        code = _sanitize_municipio_codigo_cell(code_raw)
        if use_name:
            nome = str(row[no_phys]).strip() if not pd.isna(row[no_phys]) else ""
            label = f"{nome} — código IBGE {code}" if nome else f"Município código IBGE {code}"
        else:
            label = f"Município código IBGE {code}"
        if label not in label_to_code:
            labels.append(label)
            label_to_code[label] = code

    labels = sorted(labels, key=lambda s: (label_to_code.get(s, ""), s))
    if not labels:
        st.warning("Nenhum código de município encontrado para a UF seleccionada neste ficheiro.")
        out["filtro_impossivel_geo"] = True
        return out

    chosen = st.selectbox("Município", labels, key="ctx_municipio_select")
    out["mun_code"] = label_to_code.get(chosen)
    out["mun_label"] = chosen
    out["filtro_ativo"] = True
    st.success(
        f"Vamos **filtrar o Censo** para apenas escolas em **{chosen.split(' — ')[0].strip()}** "
        f"({uf_sel}) antes de juntar com Matrícula."
    )
    return out


def append_censo_context_columns(df: pd.DataFrame, ctx: dict[str, Any]) -> pd.DataFrame:
    """Replica metadados de contexto em todas as linhas."""

    out = df.copy()
    if ctx.get("uf"):
        out["censo_ctx_UF"] = str(ctx["uf"])
    if ctx.get("mun_code") is not None:
        out["censo_ctx_municipio_codigo"] = str(ctx["mun_code"])
    if ctx.get("mun_label"):
        out["censo_ctx_municipio_rotulo_ui"] = str(ctx["mun_label"])
    filt = "sim" if ctx.get("filtro_ativo") else "nao"
    out["censo_ctx_filtro_municipal_aplicado"] = filt
    return out


def render_logical_mapper(
    titulo: str,
    df: pd.DataFrame,
    specs: tuple[LogicalFieldSpec, ...],
    prefix_key: str,
) -> None:
    st.markdown(f"#### {titulo}")
    opts = column_options(df)
    for spec in specs:
        obrig = "**obrigatório**" if getattr(spec, "obrigatorio_escola", False) or getattr(
            spec, "obrigatorio_matricula", False
        ) else "opcional"
        st.selectbox(
            f"`{spec.key}` ({obrig}) — {spec.description_pt}",
            opts,
            key=f"{prefix_key}_{spec.key}",
        )


def render_cnpj_stats_block(label: str, df: pd.DataFrame, col_name: str) -> None:
    """Mostra contagem vazios / inválidos / válidos (com dígitos verificadores)."""

    st.markdown(label)
    if col_name == SELECT_SENTINEL or col_name not in df.columns:
        st.info("Selecione primeiro a coluna de CNPJ.")
        return
    stats = summarize_cnpj_column(df[col_name])
    c_empty, c_inv, c_ok = st.columns(3)
    c_empty.metric("Vazio", f"{stats.empty:,}")
    c_inv.metric("Inválido formato/DV", f"{stats.invalid_format_or_checksum:,}")
    c_ok.metric("Válidos (DV ok)", f"{stats.valid_checksum:,}")
    pct = (
        round(100.0 * stats.valid_checksum / stats.total, 1)
        if stats.total
        else 0.0
    )
    st.caption(
        f"Total de registos avaliados: **{stats.total:,}**. "
        f"Percentagem com CNPJ aceite: **{pct} %**."
    )
    LOG.info(
        "CNPJ %s [%s]: vazio=%s inválido=%s válido=%s",
        label,
        col_name,
        stats.empty,
        stats.invalid_format_or_checksum,
        stats.valid_checksum,
    )


def run_etapa2_mapping(
    dms_df: pd.DataFrame,
    censo_df: pd.DataFrame,
    *,
    ui_simples: bool,
    up_dms_name: str,
    up_escola_name: str,
) -> None:
    """Seleção compacta das colunas de ligação DMS ↔ Censo + normalização de CNPJ."""

    st.divider()
    st.header("Etapa 2 — Ligação fiscal com o Censo municipal")
    if ui_simples:
        st.markdown(
            "O passo anterior já **nomeou logicamente** o CNPJ da escola, o nome público (`NO_ENTIDADE`) e as "
            "matrículas sempre que foram reconhecidos no ficheiro INEP ou no mapeamento avançado. "
            "Aqui concentramo-nos apenas em **alinhar os campos ao export da DMS**."
        )
    else:
        st.caption(
            "As colunas do Censo consolidado mantêm nomes estáveis (`CNPJ`, `NO_ENTIDADE`, `matriculas`, … — "
            "podem estar ausentes quando não mapeadas). Os CNPJs são normalizados e validados (DV)."
        )

    sig_uid = "|".join(
        (
            up_dms_name,
            up_escola_name,
            str(len(dms_df.index)),
            str(len(censo_df.index)),
            ",".join(map(str, list(censo_df.columns[:16]))),
        )
    )
    if st.session_state.get("_etapa2_payload_sig") != sig_uid:
        st.session_state["_etapa2_payload_sig"] = sig_uid
        for k in ("map_dms_cnpj", "map_dms_razao", "map_dms_qtd", "map_censo_cnpj", "map_censo_nome", "map_censo_mat"):
            st.session_state.pop(k, None)
        dm = propose_dms_mapping([str(c) for c in dms_df.columns])
        cols_d = set(map(str, dms_df.columns))
        cols_c = set(map(str, censo_df.columns))
        prop_d_cnpj = dm.get("CNPJ")
        if isinstance(prop_d_cnpj, str) and prop_d_cnpj in cols_d:
            st.session_state["map_dms_cnpj"] = prop_d_cnpj
        prop_r = dm.get("razao_social")
        if isinstance(prop_r, str) and prop_r in cols_d:
            st.session_state["map_dms_razao"] = prop_r
        prop_q = dm.get("quantidade")
        if isinstance(prop_q, str) and prop_q in cols_d:
            st.session_state["map_dms_qtd"] = prop_q
        if "CNPJ" in cols_c:
            st.session_state["map_censo_cnpj"] = "CNPJ"
        if "NO_ENTIDADE" in cols_c:
            st.session_state["map_censo_nome"] = "NO_ENTIDADE"
        if "matriculas" in cols_c:
            st.session_state["map_censo_mat"] = "matriculas"

    opts_d = column_options(dms_df)
    opts_c = column_options(censo_df)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("DMS-Educação")
        dms_cnpj = st.selectbox(
            "Coluna que contém o CNPJ jurídico",
            opts_d,
            key="map_dms_cnpj",
            help='Procuramos automaticamente por aliases comuns (``CNPJ``, ``NU_CNPJ``, …).',
        )
        dms_razao = st.selectbox(
            "Texto institucional (razão ou nome próximo)",
            opts_d,
            key="map_dms_razao",
            help='Usado **mais tarde** no matching textual quando o fiscal quiser conferir também por texto.',
        )
        dms_qtd = st.selectbox(
            "Quantidade relacionada ao serviço/alunos (opcional)",
            opts_d,
            key="map_dms_qtd",
        )

    with c2:
        st.subheader("Censo (já com colunas lógicas quando possível)")
        censo_cnpj = st.selectbox(
            "CNPJ institucional do Censo",
            opts_c,
            key="map_censo_cnpj",
        )
        censo_nome = st.selectbox(
            "Denominação pública reconhecível",
            opts_c,
            key="map_censo_nome",
        )
        censo_mat = st.selectbox(
            "Campo agregado de matrículas (opcional)",
            opts_c,
            key="map_censo_mat",
        )

    diag_exp = st.expander("Diagnóstico de CNPJ (modo avançado)", expanded=not ui_simples)
    with diag_exp:
        left, right = st.columns(2)
        with left:
            render_cnpj_stats_block("DMS", dms_df, dms_cnpj)
        with right:
            render_cnpj_stats_block("Censo municipal", censo_df, censo_cnpj)

    if ui_simples:
        with st.expander("Ajustar campos manualmente (se o ficheiro DMS tiver nomes atípicos)"):
            st.caption(
                "Só precisa de alterar alguma coisa se a deteção automática não bater com o layout real—por exemplo "
                "colunas internas com prefixos do sistema contabilístico."
            )

    if (
        dms_cnpj != SELECT_SENTINEL
        and censo_cnpj != SELECT_SENTINEL
        and dms_cnpj in dms_df.columns
        and censo_cnpj in censo_df.columns
    ):
        dms_work = add_normalized_cnpj_column(dms_df, dms_cnpj, "__cnpj_norm_dms")
        censo_work = add_normalized_cnpj_column(censo_df, censo_cnpj, "__cnpj_norm_censo")
        st.session_state["dms_work"] = dms_work
        st.session_state["censo_work"] = censo_work
        st.session_state["column_map"] = {
            "dms_cnpj": dms_cnpj,
            "dms_razao": dms_razao,
            "dms_qtd": dms_qtd,
            "censo_cnpj": censo_cnpj,
            "censo_nome": censo_nome,
            "censo_mat": censo_mat,
        }
        if ui_simples:
            st.success(
                "**Tudo certo.** Pré-processámos os CNPJs em memória — na Etapa 3 pode cruzar também por texto "
                "se desejar."
            )
        else:
            sample = dms_work[[dms_cnpj, "__cnpj_norm_dms"]].head(8)
            with st.expander("Pré-visualização de CNPJ normalizado (DMS — primeiras linhas)"):
                st.dataframe(sample, use_container_width=True, hide_index=True)
        LOG.info("Mapa da Etapa 2 persistido em session_state.")
    else:
        for key in ("dms_work", "censo_work", "column_map"):
            st.session_state.pop(key, None)
        if ui_simples:
            st.info("Escolha o **CNPJ** na base DMS e no Censo — é o único par obrigatório para continuar.")
        else:
            st.info(
                "Seleccione o par **CNPJ** nas duas bases para criar as colunas internas `__cnpj_norm_dms` "
                "e `__cnpj_norm_censo` antes do matching textual."
            )


def default_select_index(options: list[str], preferred: str | None) -> int:
    """Índice inicial do ``selectbox`` quando o nome preferido existe nas opções."""

    if preferred and preferred in options:
        return options.index(preferred)
    return 0


def run_etapa3_textual_merge(
    dms_work: pd.DataFrame,
    censo_work: pd.DataFrame,
    *,
    ux_simples: bool = False,
) -> None:
    """Matching textual + métricas + preview + export ``consolidado.xlsx``."""

    st.divider()
    st.header("Etapa 3 — Cruzamento textual (RapidFuzz)")
    if ux_simples:
        st.caption(
            "**Opcional.** Quando dois CNPJs não coincidem na DMS tentamos aproximar o nome registado pela "
            "escola usando RapidFuzz (WRatio)."
        )
    else:
        st.caption(
            "Para cada linha da DMS compara-se a razão social **normalizada** com os nomes "
            "do Censo consolidado (bloqueio por prefixo para bases grandes). "
            "Scorer: **WRatio**. Saída: **outputs/consolidado.xlsx**."
        )

    cm = st.session_state.get("column_map") or {}
    opts_dms = column_options(dms_work)
    opts_censo = column_options(censo_work)

    g1, g2 = st.columns(2)
    with g1:
        col_razao = st.selectbox(
            "NM razão social / texto (DMS)",
            opts_dms,
            index=default_select_index(opts_dms, cm.get("dms_razao")),
            key="etapa3_col_dms_razao",
        )
    with g2:
        col_nome = st.selectbox(
            "Nome da escola (Censo)",
            opts_censo,
            index=default_select_index(opts_censo, cm.get("censo_nome")),
            key="etapa3_col_censo_nome",
        )

    cutoff = st.radio(
        "Pontuação mínima RapidFuzz (0–100)",
        options=[70, 80, 90],
        index=1,
        horizontal=True,
        key="etapa3_cutoff",
    )

    if col_razao == SELECT_SENTINEL or col_nome == SELECT_SENTINEL:
        st.warning("Seleccione as duas colunas de texto para executar o matching.")
        return

    run_clicked = st.button("Executar matching textual", type="primary", key="btn_etapa3_run")

    if run_clicked:
        prog = st.progress(0)

        def _cb(progress: float) -> None:
            prog.progress(min(max(progress, 0.0), 1.0))

        try:
            consolidado, summary = run_textual_fuzzy_merge(
                dms_work,
                censo_work,
                col_dms_razao=col_razao,
                col_censo_nome=col_nome,
                score_cutoff=float(cutoff),
                progress_callback=_cb,
            )
        except Exception as exc:  # pylint: disable=broad-except
            prog.empty()
            st.error("Não foi possível concluir o matching textual.")
            LOG.exception("Etapa 3 — erro")
            with st.expander("Detalhe técnico"):
                st.code(str(exc))
            return

        prog.empty()

        st.session_state["consolidado_df"] = consolidado
        st.session_state["consolidado_summary"] = summary
        st.session_state["etapa3_run_signature"] = (col_razao, col_nome, int(cutoff))

        out_path = APP_DIR / "outputs" / "consolidado.xlsx"
        try:
            consolidado.to_excel(out_path, index=False, engine="openpyxl")
            LOG.info(
                "consolidado.xlsx gravado em %s (%s linhas, %s matches).",
                out_path,
                len(consolidado.index),
                summary.encontrados,
            )
            st.success(f"Consolidado gravado em `{out_path}`.")
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Gravação em disco falhou (permite mes assim descarregar): {exc}")
            LOG.exception("Falha ao gravar consolidado.xlsx")

    summary_obj = st.session_state.get("consolidado_summary")
    consolidado = st.session_state.get("consolidado_df")
    sig_stored = st.session_state.get("etapa3_run_signature")
    sig_now = (col_razao, col_nome, int(cutoff))

    if consolidado is None or summary_obj is None:
        return

    if sig_stored != sig_now:
        st.info(
            "**Parâmetros alterados** relativamente ao último resultado (colunas ou corte). "
            "Clique novamente em **Executar matching textual** para atualizar."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linhas DMS", f"{summary_obj.linhas_dms:,}")
    m2.metric("Matches encontrados", f"{summary_obj.encontrados:,}")
    m3.metric("Sem correspondência", f"{summary_obj.sem_correspondencia:,}")
    aderencia = (
        100.0 * summary_obj.encontrados / summary_obj.linhas_dms
        if summary_obj.linhas_dms
        else 0.0
    )
    m4.metric("Percentagem match", f"{aderencia:.1f} %")

    st.caption(
        f"Censo com **{summary_obj.linhas_censo:,}** registos · tempo **{summary_obj.tempo_segundos:.2f} s** · "
        f"corte **≥ {summary_obj.score_min_usado:.0f}**."
    )

    filt = st.radio(
        "Pré-visualização consolidado",
        ["Todos", "Só matches", "Só sem correspondência"],
        horizontal=True,
        key="etapa3_preview_filter",
    )
    view = consolidado
    if filt == "Só matches":
        view = consolidado.loc[consolidado["match_status"] == "match_textual"].copy()
    elif filt == "Só sem correspondência":
        view = consolidado.loc[consolidado["match_status"] == "sem_correspondencia"].copy()

    st.dataframe(view.head(200), use_container_width=True, height=420)

    buf = io.BytesIO()
    consolidado.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        label="Descarregar consolidado.xlsx",
        data=buf.getvalue(),
        file_name="consolidado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_consolidado",
    )


def _load_bundle(kind: DatasetKind, uploaded: Any, label_erro: str) -> dict[str, Any] | None:
    """Carrega upload para estrutura ``bundle`` ou devolve ``None`` com erro na UI."""

    if uploaded is None:
        return None
    raw = uploaded.getvalue()
    fname = uploaded.name
    try:
        with st.spinner(spinner_message(kind.value)):
            bundle = load_dataset_bundle(kind.value, raw, fname)
        LOG.info(
            "Carregado %s (%s): linhas=%s colunas=%s",
            dataset_kind_label(kind),
            fname,
            len(bundle["dataframe"].index),
            len(bundle["dataframe"].columns),
        )
        return bundle
    except (FileValidationError, ValueError) as exc:
        _friendly_file_error(label_erro, exc)
    except Exception as exc:  # pylint: disable=broad-except
        _friendly_file_error(label_erro, exc)
    return None


def main() -> None:
    configure_logging()
    st.set_page_config(
        page_title="DMS × Censo Escolar",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("DMS-Educação × Censo Escolar (contexto municipal)")
    st.caption(
        "**Modo simples** orienta pelo município e pré-mapeia colunas típicas de INEP. "
        "**Modo avançado** abre todas as ferramentas técnicas (mapeamentos manuais, diagnósticos detalhados)."
    )

    with st.sidebar:
        st.header("Ajuda rápida")
        st.markdown(
            "- Fluxo típico: **carregar Escola (+ Matrícula)** → definir UF/município → consolidar "
            "→ carregar **DMS** → normalizar **CNPJ** → matching texto.\n"
            "- Export **consolidado** na Etapa 3.\n"
            "- Logs: `outputs/app.log` · **⋮ → Clear cache** quando trocar ficheiros grandes."
        )
        APP_DIR.mkdir(parents=True, exist_ok=True)
        (APP_DIR / "uploads").mkdir(parents=True, exist_ok=True)
        (APP_DIR / "outputs").mkdir(parents=True, exist_ok=True)

    st.header("1. Carregar bases")
    st.caption(
        "Separe sempre **DMS** (fiscal), **Escola INEP/export** e, se disponível, a agregação de **Matrículas**. "
        "O nome dos ficheiros não importa — só precisamos de CSV/XLSX íntegro."
    )

    u1, u2, u3 = st.columns(3)
    with u1:
        up_dms = st.file_uploader(
            dataset_kind_label(DatasetKind.DMS_EDUCACAO),
            type=["csv", "xlsx"],
            key="upload_slot_dms",
            help="Exportação da DMS-Educação (CSV ou Excel).",
        )
    with u2:
        up_escola = st.file_uploader(
            dataset_kind_label(DatasetKind.CENSO_ESCOLA),
            type=["csv", "xlsx"],
            key="upload_slot_censo_escola",
            help="Tabela de escolas do Censo (qualquer exercício).",
        )
    with u3:
        up_mat = st.file_uploader(
            dataset_kind_label(DatasetKind.CENSO_MATRICULA),
            type=["csv", "xlsx"],
            key="upload_slot_censo_matricula",
            help="Opcional mas recomenda-se para pré-filtrar antes do merge nacional.",
        )

    ux_mode_choice = st.radio(
        "**Modo de trabalho**",
        [UX_SIMPLES, UX_AVANCADO],
        horizontal=False,
        key="ux_flow_mode_radio",
        help="O modo simples esconde mapeamentos técnicos até ser realmente preciso corrigi-los manualmente.",
    )
    ui_simples = ux_mode_choice == UX_SIMPLES

    dms_bundle = _load_bundle(DatasetKind.DMS_EDUCACAO, up_dms, "DMS-Educação")
    escola_bundle = _load_bundle(DatasetKind.CENSO_ESCOLA, up_escola, "Censo Escola")
    mat_bundle = _load_bundle(DatasetKind.CENSO_MATRICULA, up_mat, "Censo Matrícula")

    dms_df: pd.DataFrame | None = None
    dms_meta: dict[str, Any] = {}
    df_escola: pd.DataFrame | None = None
    df_mat: pd.DataFrame | None = None

    if dms_bundle:
        dms_df = dms_bundle["dataframe"]
        dms_meta = dms_bundle.get("meta") or {}
    if escola_bundle:
        df_escola = escola_bundle["dataframe"]
    if mat_bundle:
        df_mat = mat_bundle["dataframe"]

    pv1, pv2, pv3 = st.columns(3)
    preview_h = 260 if ui_simples else 320
    with pv1:
        st.subheader("Pré-visualização — DMS")
        if dms_df is None:
            st.info("Sem ficheiro.")
        else:
            st.success(f"`{up_dms.name}` · {len(dms_df):,} × {len(dms_df.columns)}")
            if dms_meta and not ui_simples:
                with st.expander("Meta cabeçalho export DMS"):
                    slim = {k: v for k, v in dms_meta.items() if k != "columns"}
                    st.json(slim)
            st.markdown("**Amostra** — primeiras linhas")
            st.dataframe(dms_df.head(14), use_container_width=True, height=preview_h)
    with pv2:
        st.subheader("Pré-visualização — Escola")
        if df_escola is None:
            st.info("Sem ficheiro.")
        else:
            st.success(f"`{up_escola.name}` · {len(df_escola):,} × {len(df_escola.columns)}")
            st.markdown("**Amostra** — antes do recorte municipal")
            st.dataframe(df_escola.head(14), use_container_width=True, height=preview_h)
    with pv3:
        st.subheader("Pré-visualização — Matrícula")
        if df_mat is None:
            st.info("Sem ficheiro (opcional).")
        else:
            st.success(f"`{up_mat.name}` · {len(df_mat):,} × {len(df_mat.columns)}")
            st.markdown("**Amostra**")
            st.dataframe(df_mat.head(14), use_container_width=True, height=preview_h)

    st.divider()
    st.header("2. Orientar Escola ▸ Matrícula ao município")
    cols_esc = list(map(str, df_escola.columns)) if df_escola is not None else []

    if not ui_simples and df_escola is not None:
        st.markdown("#### Mapeamento manual (somente modo avançado)")
        render_logical_mapper("Tabela Escola", df_escola, CENSO_ESCOLA_FIELDS, "logical_escola")

    if not ui_simples and df_mat is not None:
        render_logical_mapper("Tabela Matrícula", df_mat, CENSO_MATRICULA_FIELDS, "logical_matricula")

    if df_escola is None:
        st.warning("Precisamos do ficheiro **Escola** para continuar até à consolidação.")
        return

    resolved_escola_map = resolve_escola_mapping(cols_esc, ui_simples=ui_simples)

    co_auto = resolved_escola_map.get("CO_ENTIDADE")
    if co_auto:
        st.success(
            f"**Identificação automática.** A coluna **`{co_auto}`** ficou ligada ao papel lógico `CO_ENTIDADE` "
            "(código típico de escola / INEP)."
        )

    geo_ctx_snapshot = render_etapa0_contexto_municipal(
        df_escola,
        resolved_escola_map,
        ui_simples=ui_simples,
    )
    exercise_year_ctx = int(geo_ctx_snapshot["exercise"])

    if st.button("Consolidar Censo municipal (Escola ⊕ Matrícula)", type="primary", key="btn_consolidar_censo"):
        map_e_click = resolve_escola_mapping(cols_esc, ui_simples=ui_simples)
        map_m_click = resolve_matricula_mapping(list(map(str, df_mat.columns)) if df_mat is not None else None, ui_simples=ui_simples)

        df_esc_eff = df_escola.copy()
        df_mat_eff = df_mat.copy() if df_mat is not None else None

        filt_stats: dict[str, str | int] = {}
        if geo_ctx_snapshot.get("filtro_ativo") and geo_ctx_snapshot.get("uf") and geo_ctx_snapshot.get("mun_code"):
            df_esc_eff, filt_stats = filter_escola_by_municipality(
                df_esc_eff,
                map_e_click,
                uf_escolha=str(geo_ctx_snapshot["uf"]),
                municipio_codigo=str(geo_ctx_snapshot["mun_code"]),
            )
            antes = filt_stats.get("antes", "")
            depois = filt_stats.get("depois", "")
            motivo_txt = filt_stats.get("motivo", "")
            if motivo_txt:
                st.warning(f"Filtro municipal não aplicável: **{motivo_txt}**")
            else:
                st.info(f"Linhas Escola **antes ▸ depois** do filtro municipal: `{antes:,}` ▸ `{depois:,}`.")

        phys_co_esc = map_e_click.get("CO_ENTIDADE")
        if (
            isinstance(df_mat_eff, pd.DataFrame)
            and phys_co_esc is not None
            and phys_co_esc in df_esc_eff.columns
        ):
            subset_co = normalize_co_entidade(df_esc_eff[phys_co_esc])
            entidades = set(subset_co.tolist())
            entidades.discard("")
            phys_co_mat = map_m_click.get("CO_ENTIDADE")
            if (
                phys_co_mat
                and isinstance(phys_co_mat, str)
                and phys_co_mat in df_mat_eff.columns
                and entidades
            ):
                df_mat_eff_, mat_trim = restrict_matricula_to_entidades(df_mat_eff, phys_co_mat, entidades)
                df_mat_eff = df_mat_eff_
                st.success(
                    f"Matrículas também **recortadas** ao mesmo conjunto de escolas municipais (`{phys_co_mat}`): "
                    f"{mat_trim['antes_mat']:,} → {mat_trim['depois_mat']:,} linhas."
                )

        fn_esc = up_escola.name if up_escola else ""
        fn_mat = up_mat.name if up_mat else ""
        merged_out: pd.DataFrame | None = None
        try:
            merged_out = consolidate_census_escolar(
                df_esc_eff,
                df_mat_eff,
                map_e_click,
                map_m_click,
                int(exercise_year_ctx),
                source_escola_label=fn_esc,
                source_matricula_label=fn_mat or "",
            )
        except CensusMergeError as exc:
            st.error(str(exc))
            LOG.warning("Consolidação Censo recusada: %s", exc)
        except Exception as exc:  # pylint: disable=broad-except
            st.error("Erro inesperado na consolidação.")
            LOG.exception("merge censo")
            st.code(str(exc))
        else:
            if merged_out is not None:
                merged_final = append_censo_context_columns(merged_out, geo_ctx_snapshot)
                st.session_state["censo_consolidado_df"] = merged_final
                LOG.debug("Mapeamento escola aplicado: %s | matricula: %s", map_e_click, map_m_click)
                st.success(f"Censo consolidado (**{len(merged_final.index):,}** linhas**) com metadados de contexto.")

                stale_sig_anchor = tuple(
                    sorted(
                        {
                            ("uf_pick", geo_ctx_snapshot.get("uf")),
                            ("mun_pick", geo_ctx_snapshot.get("mun_code")),
                            ("filtro_geo", geo_ctx_snapshot.get("filtro_ativo")),
                            ("skip_geo_flag", geo_ctx_snapshot.get("skip_geo")),
                        }
                    )
                )
                st.session_state["censo_consolidado_signature"] = (
                    fn_esc,
                    fn_mat or "",
                    int(exercise_year_ctx),
                    bool(ui_simples),
                    tuple(sorted(map_e_click.items())),
                    tuple(sorted(map_m_click.items())),
                    stale_sig_anchor,
                )

    censo_consolidado = st.session_state.get("censo_consolidado_df")
    sig_store = st.session_state.get("censo_consolidado_signature")

    resolved_now = resolve_escola_mapping(list(map(str, df_escola.columns)), ui_simples=ui_simples)
    map_m_live = resolve_matricula_mapping(list(map(str, df_mat.columns)) if df_mat is not None else None, ui_simples=ui_simples)
    cur_ctx_stub = geo_ctx_snapshot
    stale_anchor_now = tuple(
        sorted(
            {
                ("uf_pick", cur_ctx_stub.get("uf")),
                ("mun_pick", cur_ctx_stub.get("mun_code")),
                ("filtro_geo", cur_ctx_stub.get("filtro_ativo")),
                ("skip_geo_flag", cur_ctx_stub.get("skip_geo")),
            }
        )
    )
    current_sig_attempt = (
        up_escola.name if up_escola else "",
        up_mat.name if up_mat else "",
        int(cur_ctx_stub["exercise"]),
        bool(ui_simples),
        tuple(sorted(resolved_now.items())),
        tuple(sorted(map_m_live.items())),
        stale_anchor_now,
    )

    if isinstance(censo_consolidado, pd.DataFrame):
        st.subheader("Base Censo consolidada disponível nesta sessão")
        meta_cols = [
            c
            for c in censo_consolidado.columns
            if str(c).startswith(("censo_", "censo_ctx_")) or str(c) == "censo_exercicio"
        ]
        if meta_cols:
            st.caption("Metadados colados à consolidação: " + ", ".join(f"`{c}`" for c in meta_cols[:11]))
        st.dataframe(censo_consolidado.head(30), use_container_width=True, height=360)
        if sig_store is not None and sig_store != current_sig_attempt:
            st.warning(
                "Os ficheiros, o modo UX, os mapeamentos **ou** o contexto municipal/exercício mudaram "
                "**desde a última consolidação bem-sucedida**. Clique novamente em **Consolidar** para manter "
                "a base alinhada com o formulário atual."
            )

    st.divider()
    st.header("3. Cruzamento com DMS")

    _maybe_continue_dms_etapas(dms_df, censo_consolidado, ui_simples, up_dms, up_escola)


def _maybe_continue_dms_etapas(
    dms_df: pd.DataFrame | None,
    censo_consolidado: object,
    ui_simples: bool,
    up_dms: Any,
    up_escola: Any,
) -> None:
    if not isinstance(censo_consolidado, pd.DataFrame):
        st.warning(
            "Após configurar UF/município e garantir reconhecimento das colunas, clique em **Consolidar** "
            "para gerar a base única usada pela DMS."
        )
        return

    if dms_df is None:
        st.info(
            "**Carregue a DMS‑Educação** para iniciar Etapa 2 (CNPJ normalizado em memória) e Etapa 3 (matching)."
        )
        return

    dms_nm = getattr(up_dms, "name", "?")
    escola_nm = getattr(up_escola, "name", "?") if up_escola else "?"
    run_etapa2_mapping(
        dms_df,
        censo_consolidado,
        ui_simples=ui_simples,
        up_dms_name=str(dms_nm),
        up_escola_name=str(escola_nm),
    )

    dms_work = st.session_state.get("dms_work")
    censo_work = st.session_state.get("censo_work")
    if isinstance(dms_work, pd.DataFrame) and isinstance(censo_work, pd.DataFrame):
        run_etapa3_textual_merge(dms_work, censo_work, ux_simples=ui_simples)
    else:
        if ui_simples:
            st.info("Assim que escolher o **CNPJ** na DMS e no Censo desbloqueia o matching texto.")
        else:
            st.warning("Complete Etapa 2 com o par **CNPJ** para habilitar Etapa 3.")


if __name__ == "__main__":
    main()

"""Área Streamlit dos carregamentos: técnico (3 uploads) ou operacional (multi-ficheiro + triagem)."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import streamlit as st

from domain.dataset_kind import DatasetKind, label as dataset_kind_label
from services.upload_batch_classifier import (
    ClassificationAttempt,
    OperationalBatchResolved,
    classify_uploaded_file as _svc_classify_uploaded_file,

    infer_dataset_kind as _infer_kind_on_normalized_cols,
    normalized_column_set,
    resolve_operational_upload_batch as _resolver_ops_interno_servico_principal,
)

_UI_LOG = logging.getLogger(__name__)


def classify_uploaded_file(uploaded_file: Any) -> ClassificationAttempt | None:
    return _svc_classify_uploaded_file(uploaded_file)


def infer_dataset_kind(
    columns: Iterable[str],

) -> tuple[DatasetKind | None, str, dict[DatasetKind, int]]:
    """

    Inferência apenas por texto dos cabeçalhos físicos já listados (**sem ler ficheiro**).


    Delega aos marcadores esperados pela CIF (ver ``services.upload_batch_classifier``).


    """

    return _infer_kind_on_normalized_cols(normalized_column_set(columns))



def resolve_operational_upload_batch(uploads: Iterable[Any]) -> OperationalBatchResolved:
    fich = tuple(uploads or ())
    _UI_LOG.debug("UI resolve batch: %s ficheiros", len(fich))
    return _resolver_ops_interno_servico_principal(fich)



def render_operacional_batch_upload(

) -> tuple[Any | None, Any | None, Any | None, OperationalBatchResolved]:






    """

    Substituindo triplo ``file_uploader`` operacional pela triagem única multi-seleccionada.



    Mantém resultado `(dms, escola, matricula, resultado_batch)` esperado pela orquestração em ``app``.
    """

    st.subheader("Carregar arquivos")






    st.caption(




        "**Lote único:** escolha de uma vez CSV/XLSx da **DMS**, **microdados Escola** e base **Matrícula** (**Ctrl+Seleccionar vários**, ou drag-and-drop)." 
    )


    multi_fich = st.file_uploader(
        "Carregar os três conjuntos obrigatórios do cruzamento CIF Salvador",
        type=["csv", "xlsx"],

        accept_multiple_files=True,


        key="cif_op_batch_uploader_primary_v02",


        help="Três artefactos diferentes **na mesma janela** — o programa descobre apenas pelas colunas técnicas habitualmente vistas nas exportações oficiais."


    )




    lote_carregado_tuple = tuple(multi_fich or ())






    resultado_stack = resolve_operational_upload_batch(lote_carregado_tuple)



















    for aviso_visual in resultado_stack.warnings:


        aviso_visual = aviso_visual.strip()


        if not aviso_visual:


            continue




        st.warning(aviso_visual)




        _UI_LOG.warning("oper_UPLOAD_ui_warn — %s", aviso_visual)

















    lbl_cart_op = {
        DatasetKind.DMS_EDUCACAO: "✅ **DMS Educação** identificada",
        DatasetKind.CENSO_ESCOLA: "✅ **Censo Escola** identificado",
        DatasetKind.CENSO_MATRICULA: "✅ **Censo Matrícula** identificado",
    }






    slot_tpl = (
        (DatasetKind.DMS_EDUCACAO, "DMS‑Educação (extrato fiscal)"),
        (DatasetKind.CENSO_ESCOLA, "Censo Escola"),

        (DatasetKind.CENSO_MATRICULA, "Censo Matrícula"),
    )


    tri_col = st.columns(3)



    for pos_col, tpl_desc in enumerate(slot_tpl):




        col_ctx = tri_col[pos_col]


        dk_enum_slot, texto_curto_human_slot = tpl_desc




        resultado_slot = resultado_stack.attempts_by_kind.get(dk_enum_slot)




        confl_ambig_here = dk_enum_slot in resultado_stack.ambiguous_kinds




        with col_ctx:




            if confl_ambig_here:






                st.error("⚠ **Dois ficheiros** competem pela mesma categoria estrutural — confirme o lote antes de processar.")


                continue






            if resultado_slot is not None:


                alfa, beta = resultado_slot.dataframe_ready.shape


                st.success(lbl_cart_op[dk_enum_slot])
                st.markdown(f"**Ficheiro:** `{resultado_slot.filename}`")
                st.caption(f"**Linhas × colunas:** {int(alfa):,} × {int(beta):,}")


                _UI_LOG.info(




                    "[identificação_operacional_visual_OK] papel=%s ficheiro=%r meta=%s",






                    dk_enum_slot.value,






                    resultado_slot.filename,




                    resultado_slot.method_heuristic,


                )


            else:


                st.info(


                    "⏳ **Pendentes** — espera arquivo com marcações suficientemente claras segundo o **cenário típico**."


                    + " ("


                    + texto_curto_human_slot


                    + ")."


                )




                _UI_LOG.debug("oper_UPLOAD_slot_SEM_classificação %s.", dk_enum_slot.value)






    ufDMS_WIDGET = resultado_stack.uploads.get(DatasetKind.DMS_EDUCACAO)



    uf_ESC_WIDGET = resultado_stack.uploads.get(DatasetKind.CENSO_ESCOLA)


    uf_MAT_WIDGET = resultado_stack.uploads.get(DatasetKind.CENSO_MATRICULA)




    return ufDMS_WIDGET, uf_ESC_WIDGET, uf_MAT_WIDGET, resultado_stack


def render_secao_carregar_arquivos(


    *,
    compacto_operacional: bool,

) -> tuple[Any | None, Any | None, Any | None]:





    """

    Fluxo técnico clássico com **dois uploads visíveis lado a lado**/três.



    **`compacto_operacional=True`:** levantado erro porque o Salvador compacto já não usa esse caminho.



    """

    if compacto_operacional:


        raise ValueError(
            "**Operação CIF‑Salvador compacta obriga apenas** `render_operacional_batch_upload` — não chame esta função com `compacto_operacional=True`."

        )




    st.header("1. Carregar bases")




    st.caption(




        "**DMS**: extrato próprio fiscal; **Escola**/ **Matricula** censo habitual INEP/export. Nomes são livres; validamos estruturas."


    )




    cA, cB, cZ = st.columns(3)



    with cA:




        dms_manual = st.file_uploader(
            dataset_kind_label(DatasetKind.DMS_EDUCACAO),
            type=["csv", "xlsx"],
            key="upload_slot_dms",
        )
    with cB:
        esc_manual = st.file_uploader(
            dataset_kind_label(DatasetKind.CENSO_ESCOLA),
            type=["csv", "xlsx"],
            key="upload_slot_censo_escola",
        )
    with cZ:
        mat_manual = st.file_uploader(
            dataset_kind_label(DatasetKind.CENSO_MATRICULA),
            type=["csv", "xlsx"],
            key="upload_slot_censo_matricula",
        )




    return dms_manual, esc_manual, mat_manual




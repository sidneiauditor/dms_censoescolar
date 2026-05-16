"""
Identificação automática em lote: DMS-Educação vs Censo Escola vs Censo Matrícula.

Somente modo operacional. Heurísticas baseadas nos cabeçalhos habitualmente esperados pela CIF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from domain.dataset_kind import DatasetKind
from domain.dataset_kind import label as dataset_kind_pt
from services.inferred_mapping import normalize_identifier
from services.table_loader import load_dataset_bundle

LOG = logging.getLogger(__name__)


_MARKERS_DMS: frozenset[str] = frozenset(
    normalize_identifier(x)
    for x in ("NUCNPJ", "VLIMPOSTO", "VLMENSALIDADE", "QUANTIDADE", "VLBASECALCULO")
)
_MARKERS_ESCOLA: frozenset[str] = frozenset(
    normalize_identifier(x) for x in ("NO_ENTIDADE", "CO_ENTIDADE", "TP_DEPENDENCIA", "SG_UF")
)
_MARKERS_MATRICULA: frozenset[str] = frozenset(
    normalize_identifier(x) for x in ("QT_MAT", "QT_MAT_BAS", "CO_ENTIDADE")
)

_MIN_HITS: dict[DatasetKind, int] = {
    DatasetKind.DMS_EDUCACAO: 2,
    DatasetKind.CENSO_ESCOLA: 3,
    DatasetKind.CENSO_MATRICULA: 2,
}


REQUIRED_BATCH_KINDS = frozenset(
    (
        DatasetKind.DMS_EDUCACAO,
        DatasetKind.CENSO_ESCOLA,
        DatasetKind.CENSO_MATRICULA,
    )
)


class ClassificationLoaderTag(str, Enum):
    PLAIN_CENSO = "plain_via_censo_escola_loader"
    DMS_SMART = "smart_dms_educacao_loader"


@dataclass(frozen=True)
class PeekVariant:

    tag: ClassificationLoaderTag
    dataframe: Any
    heuristic_loader: str


@dataclass
class ClassificationAttempt:

    uploaded: Any
    filename: str
    variant_tag: ClassificationLoaderTag
    inferred_kind: DatasetKind | None
    method_heuristic: str
    markers_hit: dict[DatasetKind, int]
    strength: float
    dataframe_ready: Any


@dataclass(frozen=True)
class OperationalBatchResolved:

    warnings: tuple[str, ...]
    uploads: dict[DatasetKind, Any]

    classifications: tuple[ClassificationAttempt | None, ...]
    ambiguous_kinds: frozenset[DatasetKind]
    unrecognized_files: frozenset[str]


    attempts_by_kind: dict[DatasetKind, ClassificationAttempt]


    def triple_ready(self) -> bool:


        """

        Trio completo apenas sem conflitos de «empate forte» dentro do mesmo tipo.

        """

        if self.ambiguous_kinds:




            return False



        return frozenset(self.uploads) == REQUIRED_BATCH_KINDS


def normalized_column_set(columns: Iterable[str]) -> frozenset[str]:
    return frozenset(normalize_identifier(c) for c in columns if str(c).strip())


def infer_dataset_kind(
    norm_cols: frozenset[str],
) -> tuple[DatasetKind | None, str, dict[DatasetKind, int]]:


    """

    Decide o tipo com um vencedor claro segundo acertos aos marcadores e mínimos por tipo."""

    hits = {


        DatasetKind.DMS_EDUCACAO: len(norm_cols & _MARKERS_DMS),


        DatasetKind.CENSO_ESCOLA: len(norm_cols & _MARKERS_ESCOLA),

        DatasetKind.CENSO_MATRICULA: len(norm_cols & _MARKERS_MATRICULA),
    }



    if not norm_cols:


        return None, "sem_colunas", hits






    ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0].value))


    best_k, best_v = ranked[0]

    second_v = ranked[1][1] if len(ranked) > 1 else -1




    if best_v <= 0:


        return None, "nenhum_marker_reconhecido", hits




    min_req = _MIN_HITS.get(best_k, 99)


    if best_v < min_req:


        return (


            None,


            (


                "abaixo_minimo_tipico_"


                f"{best_k.value}_precisa_{min_req}_obteve_{best_v}"


            ),

            hits,


        )




    if best_v <= second_v:


        return None, ("empate_ou_concorrencia_de_familias_marker", hits)


    rationale = (
        f"wins_best={best_k.value}|max={best_v}|second_max={second_v}|"
        f"dms_hit={hits[DatasetKind.DMS_EDUCACAO]}|"


        f"escola_hit={hits[DatasetKind.CENSO_ESCOLA]}|"


        f"mat_hit={hits[DatasetKind.CENSO_MATRICULA]}"






    )


    return best_k, rationale, hits


def _try_load_variant(
    *,
    raw: bytes,


    filename: str,

    tag: ClassificationLoaderTag,


) -> PeekVariant | None:


    dk = DatasetKind.DMS_EDUCACAO if tag == ClassificationLoaderTag.DMS_SMART else DatasetKind.CENSO_ESCOLA


    try:



        bundle = load_dataset_bundle(dk.value, raw, filename)

        return PeekVariant(


            tag=tag,





            dataframe=bundle["dataframe"],






            heuristic_loader=(






                "plain_escola_like_loader"


                if tag == ClassificationLoaderTag.PLAIN_CENSO






                else "dms_smart_loader",





            ),




        )


    except Exception as exc:



        LOG.info(




            "upload_batch_classifier.loader_rejeicao tag=%s ficheiro=%s exc=%s",






            tag.value,






            filename,






            exc.__class__.__name__,




        )


        return None


def classify_uploaded_file(uploaded_file: Any) -> ClassificationAttempt | None:


    """

    Corre duas cargas paralelas esperadas pela app (inteligência DMS + carregamento estilo censo).


    Mantém apenas a melhor segunda ``infer_dataset_kind``.
    """

    filename = getattr(uploaded_file, "name", "sem_nome") or "sem_nome"

    filename = str(filename)

    raw = getattr(uploaded_file, "getvalue", lambda: b"")()

    successes: list[PeekVariant] = []

    for tag in (ClassificationLoaderTag.DMS_SMART, ClassificationLoaderTag.PLAIN_CENSO):




        peek = _try_load_variant(raw=bytes(raw), filename=filename, tag=tag)


        if peek is not None:


            successes.append(peek)



    if not successes:


        LOG.warning("upload_batch_classifier: load_falhou_ficheiro=%r", filename)


        return None






    best_attempt: ClassificationAttempt | None = None



    for peek in successes:


        ncol = normalized_column_set(peek.dataframe.columns)

        kind_infer, heuristic_raw, mh = infer_dataset_kind(ncol)






        discriminador_esc = len(ncol & (_MARKERS_ESCOLA - _MARKERS_MATRICULA))


        discriminador_mat = len(ncol & (_MARKERS_MATRICULA - _MARKERS_ESCOLA))



        bias = discriminador_esc * 0.44 + discriminador_mat * 0.46






        cand_kind_final = kind_infer






        if kind_infer is None:


            base = float(max(mh.values(), default=-1.0))

            extra_dms_boost = mh.get(DatasetKind.DMS_EDUCACAO, 0) * 1.93

            total_strength_raw = float(base + extra_dms_boost + bias * 10.85)







            mh_note_parts = ";".join(f"{k_.value}->{mh[k_]}" for k_ in mh)


            mh_note_parts = mh_note_parts or "zero_hits_marker"






            heuristic_full = f"auto_rejeicao::{heuristic_raw}|{peek.heuristic_loader}|hit_map({mh_note_parts})"


        else:


            loader_bonus_pref = float(




                bool(




                    kind_infer == DatasetKind.DMS_EDUCACAO


                    and peek.tag == ClassificationLoaderTag.DMS_SMART





                )




                or bool(






                    kind_infer != DatasetKind.DMS_EDUCACAO


                    and peek.tag == ClassificationLoaderTag.PLAIN_CENSO






                )




            ) * 13.92






            total_strength_raw = float(kind_infer and mh[kind_infer] * 22.7 + loader_bonus_pref + bias)





            heuristic_full = (






                heuristic_raw




                + f"|LOADER={peek.tag.value}|motor={peek.heuristic_loader}"




            )





















        cand = ClassificationAttempt(






            uploaded=uploaded_file,





            filename=filename,







            variant_tag=peek.tag,







            inferred_kind=cand_kind_final,




            method_heuristic=heuristic_full,







            markers_hit=dict(mh),







            strength=float(total_strength_raw),




            dataframe_ready=peek.dataframe.copy(),





        )
























        LOG.info(






            (




                "upload_batch_classifier.triagem arquivo=%r inferido_kind=%s força_interna_real=%s "




                "|detalhe=%s marcadores_internos_hit=%s"




            ),

            filename,

            cand_kind_final.value if cand_kind_final else "SEM_TIPO_FINAL",




            cand.strength,

            heuristic_full,

            mh,






        )





        if best_attempt is None:

            best_attempt = cand

        elif cand.strength > best_attempt.strength:

            best_attempt = cand

    

    assert best_attempt is not None

    return best_attempt


def resolve_operational_upload_batch(
    uploads: Sequence[Any],

) -> OperationalBatchResolved:


    notices: list[str] = []

    pool: dict[DatasetKind, list[ClassificationAttempt]] = {
        DatasetKind.DMS_EDUCACAO: [],
        DatasetKind.CENSO_ESCOLA: [],

        DatasetKind.CENSO_MATRICULA: [],
    }



    classifications: list[ClassificationAttempt | None] = []

    unknowns: list[str] = []

    uploads_list = tuple(uploads or ())




    tally_name: dict[str, int] = {}



    for uploaded in uploads_list:

        fn = getattr(uploaded, "name", "?")

        tally_name.setdefault(str(fn), 0)




        tally_name[str(fn)] += 1




        cand = classify_uploaded_file(uploaded)

        classifications.append(cand)



        if cand is None:

            unknowns.append(str(fn))





            notices.append(f"⚠ Arquivo ilegível ou corrompido: `{fn}`.")






            LOG.warning("upload_batch_classifier.resolve: SKIP arquivo `%s` cand=None", fn)




            continue



        if cand.inferred_kind is None:






            unknowns.append(str(fn))




            notices.append(


                "⚠ Arquivo não identificado automaticamente pelo pipeline habitual do **CIF**"


                + f". (`{fn}`)"


            )




            LOG.warning(


                "upload_batch_classifier.arquivo_SEM_TIPO arquivo=%r metodo_interno=`%s`",


                fn,


                cand.method_heuristic,


            )


            continue






        pool[cand.inferred_kind].append(cand)




        LOG.info(




            "upload_batch_classifier.BUCKET tipo=%s ficheiro=%r",




            cand.inferred_kind.value,




            fn,


        )


    picks_upload: dict[DatasetKind, ClassificationAttempt] = {}

    ambiguous_marker: set[DatasetKind] = set()



    STRENG_EQUAL_EPS = 0.075






    for kind_slot, contenders in pool.items():

        if not contenders:

            continue




        contenders.sort(key=lambda cx: (-cx.strength, cx.filename))






        primo = contenders[0]






        segundo = contenders[1] if len(contenders) > 1 else None







        if segundo is not None and abs(segundo.strength - primo.strength) < STRENG_EQUAL_EPS:






            ambiguous_marker.add(kind_slot)




            human_lbl_kind = dataset_kind_pt(kind_slot)




            nomes_conflict = "` `".join(f"`{sx.filename}`" for sx in contenders[:4])






            notices.append((


                "⚠ **Dois** (ou mais) arquivos classificados como **"


                + human_lbl_kind


                + "** com resultados tropo «parelhos»: "




                + nomes_conflict




                + " — remover duplicidades ou usar modo técnico."




            ))




            LOG.warning(




                ("upload_batch_classifier.CONFLITO_empate_tipo `%s`: %s"),

                kind_slot.value,

                [(z.filename, z.strength, z.method_heuristic) for z in contenders[:5]],




            )


            continue






        picks_upload[kind_slot] = primo






        if len(contenders) > 1:




            notifies_multi = "` ` ".join("`" + (x.filename) + "`" for x in contenders[1:4])


            notices.append(


                ("⚠ Vários candidatos a **" + dataset_kind_pt(kind_slot) + "** ")


                + (f"a escolher — preferido maior confiança: `{primo.filename}` (ignor.{notifies_multi})")


            )


            LOG.warning(


                ("upload_batch_classifier.multiplos_escolhas_resolvido tipo=%s preferido=%r concorrentes=%r"),




                kind_slot.value,


                primo.filename,


                [w.filename for w in contenders],


            )




        else:






            LOG.info(


                ("upload_batch_classifier.seleção_ÚNICA `%s`: %s (força %.3f meta=%s )"),




                dataset_kind_pt(kind_slot),


                primo.filename,


                float(primo.strength),


                primo.method_heuristic,


            )


    if any(cnt > 1 for cnt in tally_name.values()):
        notices.append("⚠ Mais de uma cópia com o mesmo **nome original** aparece na lista.")



        LOG.warning(
            ("upload_batch_classifier.duplicacao_nomes_ficheiros: %s"),


            [n for (n, c) in tally_name.items() if c > 1],


        )


    upload_widget_map_final: dict[DatasetKind, Any] = {kk: vv.uploaded for (kk, vv) in picks_upload.items()}




    finalized = OperationalBatchResolved(




        warnings=tuple(dict.fromkeys(notices)),
        uploads=dict(upload_widget_map_final),




        classifications=tuple(classifications),






        ambiguous_kinds=frozenset(ambiguous_marker),




        unrecognized_files=frozenset(unknowns),




        attempts_by_kind=dict(picks_upload),




    )


    LOG.info(
        (
            "upload_batch_classifier.FINAL_BATCH triple_ok=%s"
            "|uploads_tipos_presentes=%s|avisos_internos_qty=%s|ambigua=%s|nao_reco=%s"


        ),

        finalized.triple_ready(),


        tuple(sorted(z.value for z in finalized.uploads)),


        len(finalized.warnings),




        sorted({x.value for x in finalized.ambiguous_kinds}),


        finalized.unrecognized_files,


    )


    return finalized




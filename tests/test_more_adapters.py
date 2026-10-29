import json
from pathlib import Path

import pytest

import execution.more_adapters as more
from execution.adapters import ADAPTERS
from execution.compiler import compile_plan, run_plan
from execution.contracts import StepContext
from query_opentargets import OpenTargetsError
from schemas import QueryManifestV2
from execution.records import OUTPUT_MODELS

PLANS = Path(__file__).resolve().parent.parent / "outputs" / "plans"
KNOWN_TARGETS = PLANS / "phase2_pdac_known_targets_batched_v4.json"

PDAC = {"id": "EFO_0002518", "name": "pancreatic ductal adenocarcinoma", "symbol": "", "entity_type": "disease"}
CRC = {"id": "MONDO_0005575", "name": "colorectal cancer", "symbol": "", "entity_type": "disease"}
KRAS = {"id": "ENSG00000133703", "name": "KRAS proto-oncogene", "symbol": "KRAS", "entity_type": "target"}
TP53 = {"id": "ENSG00000141510", "name": "tumor protein p53", "symbol": "TP53", "entity_type": "target"}


def _ctx(operation, board=None, entities=None, **extra):
    return StepContext(
        step_id="s", operation=operation, entities=entities or {}, board=board or {}, **extra
    )


def _row(target_id, symbol, rank, score, datasources=()):
    return {
        "rank": rank,
        "target_id": target_id,
        "target_symbol": symbol,
        "target_name": symbol,
        "overall_association_score": score,
        "datasource_scores": [{"datasource_id": d, "score": s} for d, s in datasources],
    }


def _found(disease, rows, total=None):
    return {
        "disease_id": disease["id"],
        "disease_name": disease["name"],
        "requested_target_count": len(rows),
        "returned_target_count": len(rows),
        "total_associated_target_count": total if total is not None else len(rows),
        "targets": rows,
        "limitations": [],
    }


def _by_disease_board(pdac_total=None):
    pdac = _found(
        PDAC,
        [
            _row("T1", "KRAS", 1, 0.9, [("literature", 0.8), ("somatic_mutation", 0.7)]),
            _row("T2", "TP53", 2, 0.6, [("literature", 0.5)]),
        ],
        total=pdac_total,
    )
    crc = _found(
        CRC,
        [
            _row("T1", "KRAS", 1, 0.8, [("literature", 0.6)]),
            _row("T3", "APC", 2, 0.7, [("somatic_mutation", 0.4)]),
        ],
    )
    return {
        "disease_target_association": {
            "by_disease": {PDAC["id"]: pdac, CRC["id"]: crc},
            "failed_diseases": {},
        }
    }


# ---------- registration ----------

def test_all_stage_one_batch_one_operations_are_registered():
    for operation in (
        "resolve_disease_target",
        "resolve_disease_targets",
        "resolve_disease_and_drugs",
        "discover_targets_per_disease",
        "retrieve_exact_pairs",
        "retrieve_datasource_scores",
        "rank_datasource_contributions",
        "intersect_targets",
        "compare_association_evidence",
        "join_disease_target_drug",
        "rank_candidates",
        "retrieve_mechanisms",
        "retrieve_mechanisms_and_modalities",
        "retrieve_clinical_stage",
    ):
        assert operation in ADAPTERS


# ---------- resolution ----------

def test_disease_and_target_are_resolved_separately(monkeypatch):
    answers = {"PDAC": PDAC, "KRAS": KRAS}
    monkeypatch.setattr(
        more,
        "resolve_mention",
        lambda mention, entity_type, entity_id=None, resolver=None: {
            k: v for k, v in answers[mention].items() if k != "entity_type"
        },
    )
    entities = {
        "e1": {"entity_type": "disease", "mention": "PDAC"},
        "e2": {"entity_type": "target", "mention": "KRAS"},
    }
    result = ADAPTERS["resolve_disease_target"].run(_ctx("resolve_disease_target", entities=entities))
    assert result.status == "success"
    assert result.provides == ["resolved_disease", "resolved_target"]
    assert result.payload["e1"]["id"] == PDAC["id"] and result.payload["e1"]["entity_type"] == "disease"
    assert result.payload["e2"]["id"] == KRAS["id"] and result.payload["e2"]["entity_type"] == "target"


def test_resolution_without_a_target_is_an_error():
    entities = {"e1": {"entity_type": "disease", "mention": "PDAC"}}
    result = ADAPTERS["resolve_disease_target"].run(_ctx("resolve_disease_target", entities=entities))
    assert result.status == "error"
    assert "target" in result.error


def test_disease_and_drug_resolution_provides_drug_capability(monkeypatch):
    monkeypatch.setattr(
        more,
        "resolve_mention",
        lambda mention, entity_type, entity_id=None, resolver=None: {"id": f"id-{mention}", "name": mention, "symbol": ""},
    )
    entities = {
        "e1": {"entity_type": "disease", "mention": "PDAC"},
        "e2": {"entity_type": "drug", "mention": "Olaparib"},
        "e3": {"entity_type": "drug", "mention": "Gemcitabine"},
    }
    result = ADAPTERS["resolve_disease_and_drugs"].run(_ctx("resolve_disease_and_drugs", entities=entities))
    assert result.status == "success"
    assert result.provides == ["resolved_disease", "resolved_drug"]
    assert {result.payload[ref]["entity_type"] for ref in ("e2", "e3")} == {"drug"}


def test_resolution_failure_is_an_error(monkeypatch):
    def boom(*args, **kwargs):
        raise OpenTargetsError("no such disease")

    monkeypatch.setattr(more, "resolve_mention", boom)
    entities = {
        "e1": {"entity_type": "disease", "mention": "x"},
        "e2": {"entity_type": "target", "mention": "y"},
    }
    result = ADAPTERS["resolve_disease_target"].run(_ctx("resolve_disease_target", entities=entities))
    assert result.status == "error" and "no such disease" in result.error


# ---------- discover_targets_per_disease ----------

def _discover_stub(failing=()):
    found = {
        PDAC["id"]: _by_disease_board()["disease_target_association"]["by_disease"][PDAC["id"]],
        CRC["id"]: _by_disease_board()["disease_target_association"]["by_disease"][CRC["id"]],
    }

    def fake(disease_name, top_n=10, disease_id=None):
        if disease_id in failing:
            raise OpenTargetsError("timeout")
        return found[disease_id]

    return fake


def _two_diseases_board():
    return {"resolved_disease": {"d1": PDAC, "d2": CRC}}


def test_each_disease_keeps_its_own_targets(monkeypatch):
    monkeypatch.setattr(more, "discover_disease_targets", _discover_stub())
    result = ADAPTERS["discover_targets_per_disease"].run(
        _ctx("discover_targets_per_disease", board=_two_diseases_board())
    )
    assert result.status == "success"
    assert set(result.payload["by_disease"]) == {PDAC["id"], CRC["id"]}
    shared = next(t for t in result.payload["targets"] if t["target_id"] == "T1")
    assert shared["disease_ids"] == [PDAC["id"], CRC["id"]]
    assert "overall_association_score" not in shared  # a score belongs to one disease
    assert result.provides == ["disease_target_association", "target_candidates"]


def test_one_failed_disease_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(more, "discover_disease_targets", _discover_stub(failing={CRC["id"]}))
    result = ADAPTERS["discover_targets_per_disease"].run(
        _ctx("discover_targets_per_disease", board=_two_diseases_board())
    )
    assert result.status == "partial_error"
    assert CRC["id"] in result.payload["failed_diseases"]
    assert PDAC["id"] in result.payload["by_disease"]


def test_all_diseases_failing_is_an_error(monkeypatch):
    monkeypatch.setattr(
        more, "discover_disease_targets", _discover_stub(failing={PDAC["id"], CRC["id"]})
    )
    result = ADAPTERS["discover_targets_per_disease"].run(
        _ctx("discover_targets_per_disease", board=_two_diseases_board())
    )
    assert result.status == "error"


def test_single_disease_keeps_the_old_payload_shape(monkeypatch):
    monkeypatch.setattr(more, "discover_disease_targets", _discover_stub())
    result = ADAPTERS["discover_targets_per_disease"].run(
        _ctx("discover_targets_per_disease", board={"resolved_disease": {"d1": PDAC}})
    )
    assert result.payload["targets"][0]["overall_association_score"] == 0.9
    assert result.payload["disease_id"] == PDAC["id"]


def test_top_n_filter_is_passed_through(monkeypatch):
    seen = {}

    def fake(disease_name, top_n=10, disease_id=None):
        seen["top_n"] = top_n
        return _found(PDAC, [_row("T1", "KRAS", 1, 0.9)])

    monkeypatch.setattr(more, "discover_disease_targets", fake)
    ADAPTERS["discover_targets_per_disease"].run(
        _ctx("discover_targets_per_disease", board={"resolved_disease": {"d1": PDAC}}, filters={"top_n": 3})
    )
    assert seen["top_n"] == 3


def test_no_resolved_disease_is_an_error():
    result = ADAPTERS["discover_targets_per_disease"].run(_ctx("discover_targets_per_disease"))
    assert result.status == "error"


# ---------- retrieve_exact_pairs ----------

def _pair(disease, target, score, *, truncated=False):
    return {
        "disease_id": disease["id"],
        "disease_name": disease["name"],
        "target_id": target["id"],
        "target_symbol": target["symbol"],
        "target_name": target["name"],
        "overall_association_score": score,
        "datasource_scores": [{"datasource_id": "literature", "score": 0.5}],
        "datasource_evidence": [
            {
                "datasource_id": "clinical_precedence",
                "total_evidence_count": 4,
                "fetched_evidence_count": 2 if truncated else 4,
                "truncated": truncated,
                "records": [],
            }
        ],
        "limitations": [],
    }


def test_exact_pairs_cover_every_disease_target_combination(monkeypatch):
    calls = []

    def fake(disease_id, target_id, disease_query="", target_query=""):
        calls.append((disease_id, target_id))
        disease = PDAC if disease_id == PDAC["id"] else CRC
        target = KRAS if target_id == KRAS["id"] else TP53
        return _pair(disease, target, 0.5)

    monkeypatch.setattr(more, "get_disease_target_evidence_by_ids", fake)
    board = {"resolved_disease": {"d1": PDAC, "d2": CRC}, "resolved_target": {"t1": KRAS, "t2": TP53}}
    result = ADAPTERS["retrieve_exact_pairs"].run(_ctx("retrieve_exact_pairs", board=board))
    assert result.status == "success"
    assert len(calls) == 4 and len(result.payload["pairs"]) == 4
    assert result.provides == ["disease_target_association"]


def test_failed_pair_makes_the_step_partial(monkeypatch):
    def fake(disease_id, target_id, disease_query="", target_query=""):
        if disease_id == CRC["id"]:
            raise OpenTargetsError("bad gateway")
        return _pair(PDAC, KRAS, 0.5)

    monkeypatch.setattr(more, "get_disease_target_evidence_by_ids", fake)
    board = {"resolved_disease": {"d1": PDAC, "d2": CRC}, "resolved_target": {"t1": KRAS}}
    result = ADAPTERS["retrieve_exact_pairs"].run(_ctx("retrieve_exact_pairs", board=board))
    assert result.status == "partial_error"
    assert f"{CRC['id']}|{KRAS['id']}" in result.payload["failed_pairs"]


def test_truncated_evidence_is_flagged(monkeypatch):
    monkeypatch.setattr(
        more,
        "get_disease_target_evidence_by_ids",
        lambda *a, **k: _pair(PDAC, KRAS, 0.5, truncated=True),
    )
    board = {"resolved_disease": {"d1": PDAC}, "resolved_target": {"t1": KRAS}}
    result = ADAPTERS["retrieve_exact_pairs"].run(_ctx("retrieve_exact_pairs", board=board))
    assert result.status == "partial_truncated" and result.truncated
    assert (result.total_count, result.fetched_count) == (4, 2)


def test_exact_pairs_need_both_entity_types():
    result = ADAPTERS["retrieve_exact_pairs"].run(
        _ctx("retrieve_exact_pairs", board={"resolved_disease": {"d1": PDAC}})
    )
    assert result.status == "error"


def test_pair_without_an_association_is_no_records(monkeypatch):
    monkeypatch.setattr(
        more,
        "get_disease_target_evidence_by_ids",
        lambda *a, **k: {**_pair(PDAC, KRAS, None), "datasource_scores": [], "datasource_evidence": []},
    )
    board = {"resolved_disease": {"d1": PDAC}, "resolved_target": {"t1": KRAS}}
    result = ADAPTERS["retrieve_exact_pairs"].run(_ctx("retrieve_exact_pairs", board=board))
    assert result.status == "no_records"


# ---------- datasource scores and their ranking ----------

def test_datasource_scores_are_flattened_per_pair():
    result = ADAPTERS["retrieve_datasource_scores"].run(
        _ctx("retrieve_datasource_scores", board=_by_disease_board())
    )
    assert result.status == "success" and result.provides == ["datasource_scores"]
    assert result.payload["pair_count"] == 4
    assert len(result.payload["rows"]) == 5
    assert any("not record counts" in text for text in result.payload["limitations"])


def test_datasource_scores_need_association_data():
    result = ADAPTERS["retrieve_datasource_scores"].run(_ctx("retrieve_datasource_scores"))
    assert result.status == "error"


def test_datasources_are_ranked_inside_each_pair():
    scores = ADAPTERS["retrieve_datasource_scores"].run(
        _ctx("retrieve_datasource_scores", board=_by_disease_board())
    )
    board = {"datasource_scores": scores.payload}
    result = ADAPTERS["rank_datasource_contributions"].run(
        _ctx("rank_datasource_contributions", board=board)
    )
    assert result.status == "success"
    ranked = result.payload["by_pair"][f"{PDAC['id']}|T1"]
    assert [row["datasource_id"] for row in ranked] == ["literature", "somatic_mutation"]
    assert [row["ranked_position"] for row in ranked] == [1, 2]


def test_ranking_datasources_without_scores_is_an_error():
    result = ADAPTERS["rank_datasource_contributions"].run(_ctx("rank_datasource_contributions"))
    assert result.status == "error"


def test_ranking_spec_can_override_the_default():
    scores = ADAPTERS["retrieve_datasource_scores"].run(
        _ctx("retrieve_datasource_scores", board=_by_disease_board())
    )
    result = ADAPTERS["rank_datasource_contributions"].run(
        _ctx(
            "rank_datasource_contributions",
            board={"datasource_scores": scores.payload},
            ranking={"primary_metric": "score", "direction": "ascending", "secondary_metrics": []},
        )
    )
    ranked = result.payload["by_pair"][f"{PDAC['id']}|T1"]
    assert [row["datasource_id"] for row in ranked] == ["somatic_mutation", "literature"]


# ---------- intersection and comparison ----------

def test_intersection_keeps_each_diseases_own_scores():
    result = ADAPTERS["intersect_targets"].run(_ctx("intersect_targets", board=_by_disease_board()))
    assert result.status == "success" and result.provides == ["target_intersection"]
    shared = result.payload["shared_targets"]
    assert [t["target_id"] for t in shared] == ["T1"]
    assert shared[0]["per_disease"][PDAC["id"]]["overall_association_score"] == 0.9
    assert shared[0]["per_disease"][CRC["id"]]["overall_association_score"] == 0.8
    assert result.payload["not_shared_target_ids"][PDAC["id"]] == ["T2"]
    assert result.payload["not_shared_target_ids"][CRC["id"]] == ["T3"]
    assert any("top" in text or "fetched" in text for text in result.payload["limitations"])


def test_intersection_reports_truncation_of_the_fetched_lists():
    result = ADAPTERS["intersect_targets"].run(
        _ctx("intersect_targets", board=_by_disease_board(pdac_total=500))
    )
    assert result.truncated is True


def test_empty_intersection_is_no_records_not_an_error():
    board = _by_disease_board()
    board["disease_target_association"]["by_disease"][CRC["id"]]["targets"] = [
        _row("T9", "OTHER", 1, 0.3)
    ]
    result = ADAPTERS["intersect_targets"].run(_ctx("intersect_targets", board=board))
    assert result.status == "no_records"
    assert result.payload["shared_targets"] == []


def test_intersection_needs_two_diseases():
    board = _by_disease_board()
    del board["disease_target_association"]["by_disease"][CRC["id"]]
    result = ADAPTERS["intersect_targets"].run(_ctx("intersect_targets", board=board))
    assert result.status == "error"


def test_comparison_uses_null_for_missing_values_not_zero():
    result = ADAPTERS["compare_association_evidence"].run(
        _ctx("compare_association_evidence", board=_by_disease_board())
    )
    assert result.status == "success"
    by_symbol = {row["target_symbol"]: row for row in result.payload["rows"]}
    assert by_symbol["APC"]["overall_association_score"][PDAC["id"]] is None
    assert by_symbol["KRAS"]["datasource_scores"]["somatic_mutation"][CRC["id"]] is None
    assert by_symbol["KRAS"]["datasource_scores"]["literature"][PDAC["id"]] == 0.8
    assert "overall_score" not in by_symbol["KRAS"]
    assert result.payload["datasource_ids"] == ["literature", "somatic_mutation"]


def test_comparison_is_limited_to_the_intersection_when_there_is_one():
    board = _by_disease_board()
    intersection = ADAPTERS["intersect_targets"].run(_ctx("intersect_targets", board=board))
    board["target_intersection"] = intersection.payload
    result = ADAPTERS["compare_association_evidence"].run(
        _ctx("compare_association_evidence", board=board)
    )
    assert [row["target_symbol"] for row in result.payload["rows"]] == ["KRAS"]


def test_comparison_works_on_exact_pair_data():
    board = {
        "disease_target_association": {
            "pairs": [_pair(PDAC, KRAS, 0.7), _pair(CRC, KRAS, 0.4)],
            "failed_pairs": {},
        }
    }
    result = ADAPTERS["compare_association_evidence"].run(
        _ctx("compare_association_evidence", board=board)
    )
    row = result.payload["rows"][0]
    assert row["overall_association_score"] == {PDAC["id"]: 0.7, CRC["id"]: 0.4}


def test_comparison_needs_association_data():
    result = ADAPTERS["compare_association_evidence"].run(_ctx("compare_association_evidence"))
    assert result.status == "error"


# ---------- join and rank ----------

def _drug(drug_id, name, stage, mechanisms=("inhibitor",)):
    return {
        "drug_id": drug_id,
        "drug_name": name,
        "drug_type": "Small molecule",
        "max_clinical_stage": stage,
        "mechanisms_of_action": list(mechanisms),
        "disease_indications": [],
        "exact_disease_match": False,
    }


def _drug_board():
    board = _by_disease_board()
    board["target_drug_evidence"] = {
        "by_target": {
            "T1": {"drugs": [_drug("D1", "SOTORASIB", "APPROVAL"), _drug("D2", "ADAGRASIB", "PHASE_2")]},
            "T2": {"drugs": []},
        },
        "failed_targets": {},
    }
    return board


def test_join_makes_one_edge_per_disease_target_drug():
    result = ADAPTERS["join_disease_target_drug"].run(
        _ctx("join_disease_target_drug", board=_drug_board())
    )
    assert result.status == "success" and result.provides == ["disease_target_drug_edges"]
    edges = result.payload["edges"]
    assert len(edges) == 4  # T1 appears in two diseases x two drugs
    assert {edge["disease_id"] for edge in edges} == {PDAC["id"], CRC["id"]}
    assert result.payload["targets_without_drug_records"] == ["T2"]
    assert result.payload["targets_not_queried"] == ["T3"]
    assert any("does not show that the drug works" in text for text in result.payload["limitations"])


def test_join_reports_failed_targets():
    board = _drug_board()
    board["target_drug_evidence"]["failed_targets"] = {"T3": "timeout"}
    result = ADAPTERS["join_disease_target_drug"].run(_ctx("join_disease_target_drug", board=board))
    assert result.status == "partial_error"
    assert "T3" not in result.payload["targets_not_queried"]


def test_join_needs_both_inputs():
    result = ADAPTERS["join_disease_target_drug"].run(
        _ctx("join_disease_target_drug", board=_by_disease_board())
    )
    assert result.status == "error"


def test_candidates_are_ranked_by_the_requested_metric():
    joined = ADAPTERS["join_disease_target_drug"].run(
        _ctx("join_disease_target_drug", board=_drug_board())
    )
    result = ADAPTERS["rank_candidates"].run(
        _ctx(
            "rank_candidates",
            board={"disease_target_drug_edges": joined.payload},
            ranking={
                "primary_metric": "overall_association_score",
                "direction": "descending",
                "secondary_metrics": [],
            },
        )
    )
    assert result.status == "success"
    assert result.payload["source_capability"] == "disease_target_drug_edges"
    scores = [c["overall_association_score"] for c in result.payload["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_ranking_without_a_spec_is_an_error():
    result = ADAPTERS["rank_candidates"].run(_ctx("rank_candidates", board=_drug_board()))
    assert result.status == "error" and "ranking" in result.error


def test_ranking_by_a_text_metric_is_refused():
    joined = ADAPTERS["join_disease_target_drug"].run(
        _ctx("join_disease_target_drug", board=_drug_board())
    )
    result = ADAPTERS["rank_candidates"].run(
        _ctx(
            "rank_candidates",
            board={"disease_target_drug_edges": joined.payload},
            ranking={"primary_metric": "max_clinical_stage", "direction": "descending", "secondary_metrics": []},
        )
    )
    assert result.status == "error" and "not numeric" in result.error


def test_ranking_falls_back_to_target_candidates():
    board = {
        "target_candidates": {
            "targets": [
                {"target_id": "T1", "overall_association_score": 0.2},
                {"target_id": "T2", "overall_association_score": 0.9},
            ]
        }
    }
    result = ADAPTERS["rank_candidates"].run(
        _ctx(
            "rank_candidates",
            board=board,
            ranking={"primary_metric": "overall_association_score", "direction": "descending", "secondary_metrics": []},
        )
    )
    assert [c["target_id"] for c in result.payload["candidates"]] == ["T2", "T1"]
    assert result.payload["source_capability"] == "target_candidates"


def test_ranking_with_nothing_to_rank_is_an_error():
    result = ADAPTERS["rank_candidates"].run(
        _ctx(
            "rank_candidates",
            ranking={"primary_metric": "overall_association_score", "direction": "descending", "secondary_metrics": []},
        )
    )
    assert result.status == "error"


# ---------- mechanisms and clinical stage ----------

def _mechanism(drug_id):
    return {
        "drug_id": drug_id,
        "drug_name": drug_id,
        "drug_type": "Small molecule",
        "mechanism_count": 1,
        "mechanisms": [],
        "targets": [],
    }


def test_mechanisms_for_explicit_drugs(monkeypatch):
    monkeypatch.setattr(more, "get_drug_targets_by_id", lambda drug_id: _mechanism(drug_id))
    drugs = {"d1": {"id": "CHEMBL1", "name": "A", "entity_type": "drug"}, "d2": {"id": "CHEMBL2", "name": "B", "entity_type": "drug"}}
    result = ADAPTERS["retrieve_mechanisms"].run(_ctx("retrieve_mechanisms", board={"resolved_drug": drugs}))
    assert result.status == "success" and result.provides == ["mechanism_evidence"]
    assert set(result.payload["by_drug"]) == {"CHEMBL1", "CHEMBL2"}


def test_mechanisms_for_the_drugs_found_on_targets(monkeypatch):
    monkeypatch.setattr(more, "get_drug_targets_by_id", lambda drug_id: _mechanism(drug_id))
    result = ADAPTERS["retrieve_mechanisms"].run(
        _ctx("retrieve_mechanisms", board=_drug_board())
    )
    assert set(result.payload["by_drug"]) == {"D1", "D2"}


def test_modality_comes_from_the_drug_type(monkeypatch):
    monkeypatch.setattr(more, "get_drug_targets_by_id", lambda drug_id: _mechanism(drug_id))
    result = ADAPTERS["retrieve_mechanisms_and_modalities"].run(
        _ctx("retrieve_mechanisms_and_modalities", board=_drug_board())
    )
    assert result.payload["by_drug"]["D1"]["modality"] == "Small molecule"


def test_mechanism_lookup_is_capped_and_says_so(monkeypatch):
    monkeypatch.setattr(more, "get_drug_targets_by_id", lambda drug_id: _mechanism(drug_id))
    result = ADAPTERS["retrieve_mechanisms"].run(
        _ctx("retrieve_mechanisms", board=_drug_board(), filters={"max_drugs": 1})
    )
    assert result.status == "partial_truncated" and result.truncated
    assert (result.total_count, result.fetched_count) == (2, 1)


def test_one_failed_drug_is_partial(monkeypatch):
    def fake(drug_id):
        if drug_id == "D2":
            raise OpenTargetsError("not found")
        return _mechanism(drug_id)

    monkeypatch.setattr(more, "get_drug_targets_by_id", fake)
    result = ADAPTERS["retrieve_mechanisms"].run(_ctx("retrieve_mechanisms", board=_drug_board()))
    assert result.status == "partial_error"
    assert "D2" in result.payload["failed_drugs"]


def test_mechanisms_need_drugs():
    result = ADAPTERS["retrieve_mechanisms"].run(_ctx("retrieve_mechanisms"))
    assert result.status == "error"


def test_clinical_stage_reports_stages_without_ranking_them():
    result = ADAPTERS["retrieve_clinical_stage"].run(
        _ctx("retrieve_clinical_stage", board=_drug_board())
    )
    assert result.status == "success" and result.provides == ["clinical_stage_evidence"]
    t1 = result.payload["by_target"]["T1"]
    assert set(t1["stages_present"]) == {"APPROVAL", "PHASE_2"}
    assert t1["drug_record_count"] == 2
    assert result.payload["by_target"]["T2"]["drug_record_count"] == 0
    assert any("not an approval" in text for text in result.payload["limitations"])


def test_clinical_stage_with_no_drugs_is_no_records():
    board = {"target_drug_evidence": {"by_target": {"T2": {"drugs": []}}, "failed_targets": {}}}
    result = ADAPTERS["retrieve_clinical_stage"].run(_ctx("retrieve_clinical_stage", board=board))
    assert result.status == "no_records"


def test_clinical_stage_needs_drug_evidence():
    result = ADAPTERS["retrieve_clinical_stage"].run(_ctx("retrieve_clinical_stage"))
    assert result.status == "error"


# ---------- a whole plan through run_plan ----------

def _entity(ref, mention):
    return {
        "entity_ref": ref,
        "entity_type": "disease",
        "mention": mention,
        "entity_id": None,
        "canonical_name": None,
        "resolution_status": "unresolved",
    }


def _step(step_id, operation, depends_on, entity_refs, outputs, **extra):
    return {
        "step_id": step_id,
        "operation": operation,
        "description": operation,
        "depends_on": depends_on,
        "entity_refs": entity_refs,
        "outputs": outputs,
        **extra,
    }


def _two_disease_plan():
    data = json.loads(KNOWN_TARGETS.read_text())
    data["original_question"] = "胰臟癌與大腸直腸癌共同的標的有哪些？"
    data["research_objective"] = "找出兩個疾病共有的標的並並列比較證據。"
    data["entities"] = [_entity("entity_1", "胰臟癌"), _entity("entity_2", "大腸直腸癌")]
    data["plan_steps"] = [
        _step("resolve_diseases_1", "resolve_diseases", [], ["entity_1", "entity_2"], ["resolved_disease_1"]),
        _step("per_disease_1", "discover_targets_per_disease", ["resolve_diseases_1"], ["entity_1", "entity_2"], ["disease_target_association_1", "target_candidates_1"]),
        _step("intersect_1", "intersect_targets", ["per_disease_1"], [], ["target_intersection_1"]),
        _step("compare_1", "compare_association_evidence", ["intersect_1"], [], ["association_comparison_1"]),
    ]
    return QueryManifestV2.model_validate(data)


def test_two_disease_plan_runs_end_to_end_on_stubbed_data(monkeypatch):
    import execution.adapters as adapters

    ids = {"胰臟癌": PDAC, "大腸直腸癌": CRC}
    monkeypatch.setattr(
        adapters,
        "resolve_mention",
        lambda mention, entity_type, entity_id=None, resolver=None: {
            k: v for k, v in ids[mention].items() if k != "entity_type"
        },
    )
    monkeypatch.setattr(more, "discover_disease_targets", _discover_stub())
    plan = _two_disease_plan()
    assert [call.adapter for call in compile_plan(plan)] == [
        "ResolveEntityAdapter",
        "DiscoverTargetsPerDiseaseAdapter",
        "IntersectTargetsAdapter",
        "CompareAssociationEvidenceAdapter",
    ]
    trace = run_plan(plan)
    assert [r.status for r in trace] == ["success"] * 4
    compare = trace[-1].payload
    assert [row["target_symbol"] for row in compare["rows"]] == ["KRAS"]


def test_a_failed_first_disease_stops_the_chain_honestly(monkeypatch):
    import execution.adapters as adapters

    def boom(*args, **kwargs):
        raise OpenTargetsError("down")

    monkeypatch.setattr(adapters, "resolve_mention", boom)
    trace = run_plan(_two_disease_plan())
    assert [r.status for r in trace] == ["error", "not_executed", "not_executed", "not_executed"]


def _plan_from(question, entities, steps):
    data = json.loads(KNOWN_TARGETS.read_text())
    data["original_question"] = question
    data["research_objective"] = question
    data["entities"] = entities
    data["plan_steps"] = steps
    return QueryManifestV2.model_validate(data)


def test_drug_repurposing_chain_runs_end_to_end_on_stubbed_data(monkeypatch):
    import execution.adapters as adapters

    monkeypatch.setattr(
        adapters,
        "resolve_entity",
        lambda name, kind, entity_id=None: {"id": PDAC["id"], "name": PDAC["name"], "symbol": ""},
    )
    monkeypatch.setattr(
        adapters,
        "discover_disease_targets",
        lambda *a, **k: _found(
            PDAC, [_row("T1", "KRAS", 1, 0.5), _row("T2", "TP53", 2, 0.9)]
        ),
    )

    def drugs(target_id, efo_id):
        found = [_drug("D1", "SOTORASIB", "APPROVAL")] if target_id == "T1" else []
        return {
            "ensembl_id": target_id,
            "efo_id": efo_id,
            "total_drug_count": len(found),
            "fetched_drug_count": len(found),
            "drugs": found,
        }

    monkeypatch.setattr(adapters, "get_target_drug_evidence_by_ids", drugs)
    plan = _plan_from(
        "胰臟癌有哪些標的已有藥物？",
        [_entity("entity_1", "胰臟癌")],
        [
            _step("resolve_disease_1", "resolve_disease", [], ["entity_1"], ["resolved_disease_1"]),
            _step("discover_1", "discover_targets", ["resolve_disease_1"], ["entity_1"], ["disease_target_association_1", "target_candidates_1"]),
            _step("drugs_1", "retrieve_target_drugs", ["discover_1"], ["entity_1"], ["target_drug_evidence_1"]),
            _step("join_1", "join_disease_target_drug", ["discover_1", "drugs_1"], [], ["disease_target_drug_edges_1"]),
            _step(
                "rank_1",
                "rank_candidates",
                ["join_1"],
                [],
                ["ranked_candidates_1"],
                ranking={"primary_metric": "overall_association_score", "direction": "descending", "secondary_metrics": []},
            ),
        ],
    )
    trace = run_plan(plan)
    assert [r.status for r in trace] == ["success"] * 5
    candidates = trace[-1].payload["candidates"]
    assert [c["drug_id"] for c in candidates] == ["D1"]
    assert trace[3].payload["targets_without_drug_records"] == ["T2"]


def test_datasource_contribution_chain_runs_end_to_end_on_stubbed_data(monkeypatch):
    import execution.adapters as adapters

    answers = {"胰臟癌": PDAC, "KRAS": KRAS}
    monkeypatch.setattr(
        more,
        "resolve_mention",
        lambda mention, entity_type, entity_id=None, resolver=None: {
            k: v for k, v in answers[mention].items() if k != "entity_type"
        },
    )
    monkeypatch.setattr(
        adapters,
        "get_disease_target_evidence_by_ids",
        lambda *a, **k: _pair(PDAC, KRAS, 0.7)
        | {
            "datasource_scores": [
                {"datasource_id": "literature", "score": 0.4},
                {"datasource_id": "somatic_mutation", "score": 0.9},
            ]
        },
    )
    target_entity = {**_entity("entity_2", "KRAS"), "entity_type": "target"}
    plan = _plan_from(
        "KRAS 與胰臟癌的證據主要來自哪些資料來源？",
        [_entity("entity_1", "胰臟癌"), target_entity],
        [
            _step("resolve_1", "resolve_disease_target", [], ["entity_1", "entity_2"], ["resolved_disease_1", "resolved_target_1"]),
            _step("pair_1", "retrieve_exact_pair", ["resolve_1"], ["entity_1", "entity_2"], ["disease_target_association_1"]),
            _step("scores_1", "retrieve_datasource_scores", ["pair_1"], [], ["datasource_scores_1"]),
            _step("rank_1", "rank_datasource_contributions", ["scores_1"], [], ["ranked_datasources_1"]),
        ],
    )
    trace = run_plan(plan)
    assert [r.status for r in trace] == ["success"] * 4
    ranked = next(iter(trace[-1].payload["by_pair"].values()))
    assert [row["datasource_id"] for row in ranked] == ["somatic_mutation", "literature"]


# ---------- rank_diseases ----------

_DISEASE_RANKING = {
    "primary_metric": "overall_association_score",
    "direction": "descending",
    "secondary_metrics": [],
}


def _pairs_board():
    def pair(disease_id, name, score):
        return {
            "disease_id": disease_id,
            "disease_name": name,
            "target_id": KRAS["id"],
            "target_symbol": "KRAS",
            "overall_association_score": score,
        }

    return {
        "disease_target_association": {
            "pairs": [
                pair("D_NONE", "no score disease", None),
                pair(CRC["id"], "colorectal cancer", 0.8),
                pair(PDAC["id"], "pancreatic ductal adenocarcinoma", 0.9),
            ]
        }
    }


def test_rank_diseases_orders_by_score_and_keeps_missing_last():
    result = ADAPTERS["rank_diseases"].run(
        _ctx("rank_diseases", board=_pairs_board(), ranking=_DISEASE_RANKING)
    )
    assert result.status == "success"
    assert [d["disease_id"] for d in result.payload["diseases"]] == [
        PDAC["id"], CRC["id"], "D_NONE",
    ]
    assert result.payload["diseases"][-1]["overall_association_score"] is None
    assert result.provides == ["disease_ranking"]


def test_rank_diseases_payload_matches_its_output_model():
    result = ADAPTERS["rank_diseases"].run(
        _ctx("rank_diseases", board=_pairs_board(), ranking=_DISEASE_RANKING)
    )
    OUTPUT_MODELS["rank_diseases"].model_validate(result.payload)


def test_rank_diseases_without_a_ranking_spec_is_an_error():
    result = ADAPTERS["rank_diseases"].run(_ctx("rank_diseases", board=_pairs_board()))
    assert result.status == "error" and "ranking" in result.error


def test_rank_diseases_without_association_data_is_an_error():
    result = ADAPTERS["rank_diseases"].run(_ctx("rank_diseases", ranking=_DISEASE_RANKING))
    assert result.status == "error" and "disease_target_association" in result.error


def test_rank_diseases_by_a_text_metric_is_refused():
    board = _pairs_board()
    board["disease_target_association"]["pairs"][0]["disease_name"] = "text"
    result = ADAPTERS["rank_diseases"].run(
        _ctx(
            "rank_diseases",
            board=board,
            ranking={"primary_metric": "disease_name", "direction": "ascending", "secondary_metrics": []},
        )
    )
    assert result.status == "error" and "not numeric" in result.error


# ---------- retrieve_associated_diseases ----------

def _target_diseases(target, rows, total=None):
    diseases = [
        {
            "rank": rank,
            "disease_id": disease_id,
            "disease_name": name,
            "target_id": target["id"],
            "target_symbol": target["symbol"],
            "target_name": target["name"],
            "overall_association_score": score,
            "datasource_scores": [],
        }
        for rank, (disease_id, name, score) in enumerate(rows, start=1)
    ]
    return {
        "target_id": target["id"],
        "target_symbol": target["symbol"],
        "requested_disease_count": 10,
        "returned_disease_count": len(diseases),
        "total_associated_disease_count": total or len(diseases),
        "diseases": diseases,
    }


def _fake_associated_diseases(found, failing=()):
    def fake(ensembl_id, top_n=10):
        if ensembl_id in failing:
            raise OpenTargetsError("timeout")
        return found[ensembl_id]

    return fake


def _targets_board(*targets):
    return {"resolved_target": {f"e{i}": t for i, t in enumerate(targets, start=1)}}


KRAS_DISEASES = _target_diseases(
    KRAS, [(PDAC["id"], PDAC["name"], 0.9), (CRC["id"], CRC["name"], 0.8)]
)
TP53_DISEASES = _target_diseases(TP53, [(CRC["id"], CRC["name"], 0.7)])


def _run_associated(monkeypatch, board, found=None, failing=(), **extra):
    monkeypatch.setattr(
        more,
        "get_target_associated_diseases",
        _fake_associated_diseases(
            found or {KRAS["id"]: KRAS_DISEASES, TP53["id"]: TP53_DISEASES}, failing
        ),
    )
    return ADAPTERS["retrieve_associated_diseases"].run(
        _ctx("retrieve_associated_diseases", board=board, **extra)
    )


def test_associated_diseases_are_collected_for_the_resolved_target(monkeypatch):
    result = _run_associated(monkeypatch, _targets_board(KRAS))
    assert result.status == "success"
    assert result.provides == ["disease_target_association"]
    assert [p["disease_id"] for p in result.payload["pairs"]] == [PDAC["id"], CRC["id"]]
    assert result.truncated is False
    OUTPUT_MODELS["retrieve_associated_diseases"].model_validate(result.payload)


def test_associated_diseases_report_truncation(monkeypatch):
    found = {KRAS["id"]: _target_diseases(KRAS, [(PDAC["id"], PDAC["name"], 0.9)], total=500)}
    result = _run_associated(monkeypatch, _targets_board(KRAS), found=found)
    assert result.truncated is True
    assert (result.total_count, result.fetched_count) == (500, 1)


def test_one_failing_target_gives_a_partial_error(monkeypatch):
    result = _run_associated(
        monkeypatch, _targets_board(KRAS, TP53), failing={TP53["id"]}
    )
    assert result.status == "partial_error"
    assert TP53["id"] in result.payload["failed_targets"]
    assert len(result.payload["pairs"]) == 2


def test_all_targets_failing_is_an_error(monkeypatch):
    result = _run_associated(monkeypatch, _targets_board(KRAS), failing={KRAS["id"]})
    assert result.status == "error" and "timeout" in result.error


def test_associated_diseases_need_a_resolved_target(monkeypatch):
    result = _run_associated(monkeypatch, {})
    assert result.status == "error" and "resolved target" in result.error


def test_associated_diseases_refuse_a_bad_top_n(monkeypatch):
    result = _run_associated(monkeypatch, _targets_board(KRAS), filters={"top_n": 0})
    assert result.status == "error" and "top_n" in result.error


def test_associated_diseases_feed_rank_diseases(monkeypatch):
    got = _run_associated(monkeypatch, _targets_board(KRAS))
    ranked = ADAPTERS["rank_diseases"].run(
        _ctx(
            "rank_diseases",
            board={"disease_target_association": got.payload},
            ranking=_DISEASE_RANKING,
        )
    )
    assert ranked.status == "success"
    assert [d["disease_id"] for d in ranked.payload["diseases"]] == [PDAC["id"], CRC["id"]]



# ---------- compare_disease_rank ----------

def _ranked_diseases_board():
    got = ADAPTERS["rank_diseases"].run(
        _ctx("rank_diseases", board=_pairs_board(), ranking=_DISEASE_RANKING)
    )
    return {"disease_ranking": got.payload}


def _compare(board, **extra):
    return ADAPTERS["compare_disease_rank"].run(
        _ctx("compare_disease_rank", board=board, **extra)
    )


def test_disease_rank_is_found_and_marked_top():
    result = _compare({**_ranked_diseases_board(), "resolved_disease": {"e1": PDAC}})
    assert result.status == "success"
    assert result.provides == ["disease_rank_comparison"]
    assert result.payload["found_in_ranking"] is True
    assert result.payload["ranked_position"] == 1 and result.payload["is_top"] is True
    OUTPUT_MODELS["compare_disease_rank"].model_validate(result.payload)


def test_disease_rank_reports_a_lower_position():
    result = _compare({**_ranked_diseases_board(), "resolved_disease": {"e1": CRC}})
    assert result.payload["ranked_position"] == 2
    assert result.payload["is_top"] is False
    assert result.payload["top_disease_id"] == PDAC["id"]


def test_disease_missing_from_the_ranking_is_not_called_unassociated():
    other = {"id": "EFO_9999999", "name": "other", "symbol": "", "entity_type": "disease"}
    result = _compare({**_ranked_diseases_board(), "resolved_disease": {"e1": other}})
    assert result.status == "success"
    assert result.payload["found_in_ranking"] is False
    assert result.payload["ranked_position"] is None
    assert any("not evidence of no association" in t for t in result.payload["limitations"])


def test_disease_rank_needs_a_ranking():
    result = _compare({"resolved_disease": {"e1": PDAC}})
    assert result.status == "error" and "disease_ranking" in result.error


def test_disease_rank_needs_exactly_one_disease():
    result = _compare(
        {**_ranked_diseases_board(), "resolved_disease": {"e1": PDAC, "e2": CRC}}
    )
    assert result.status == "error" and "one disease" in result.error


# ---------- genetic evidence without a clinical drug ----------

def _candidates_board():
    return {
        "resolved_disease": {"d": PDAC},
        "target_candidates": {
            "targets": [
                {"target_id": tid, "target_symbol": tid, "target_name": tid}
                for tid in ("T1", "T2", "T3")
            ]
        },
    }


def _fake_datatype_scores(monkeypatch, by_target):
    monkeypatch.setattr(
        more,
        "get_disease_targets_datatype_scores",
        lambda disease_id, target_ids: {
            "disease_id": disease_id,
            "disease_name": PDAC["name"],
            "requested_target_count": len(target_ids),
            "returned_target_count": len(by_target),
            "by_target": by_target,
        },
    )


def test_genetic_evidence_keeps_missing_scores_missing(monkeypatch):
    _fake_datatype_scores(
        monkeypatch,
        {
            "T1": {"datatype_scores": [
                {"datatype_id": "genetic_association", "score": 0.7},
                {"datatype_id": "literature", "score": 0.9},
            ]},
            "T2": {"datatype_scores": [{"datatype_id": "literature", "score": 0.5}]},
        },
    )
    result = ADAPTERS["retrieve_genetic_evidence"].run(
        _ctx("retrieve_genetic_evidence", board=_candidates_board())
    )
    scores = {r["target_id"]: r["genetic_evidence_score"] for r in result.payload["targets"]}
    assert scores == {"T1": 0.7, "T2": None, "T3": None}
    assert result.status == "success"
    assert result.provides == ["genetic_evidence"]
    OUTPUT_MODELS["retrieve_genetic_evidence"].model_validate(result.payload)


def test_genetic_evidence_needs_one_disease():
    board = _candidates_board()
    del board["resolved_disease"]
    result = ADAPTERS["retrieve_genetic_evidence"].run(
        _ctx("retrieve_genetic_evidence", board=board)
    )
    assert result.status == "error" and "one resolved disease" in result.error


def test_genetic_evidence_api_failure_is_an_error(monkeypatch):
    def boom(disease_id, target_ids):
        raise OpenTargetsError("timeout")

    monkeypatch.setattr(more, "get_disease_targets_datatype_scores", boom)
    result = ADAPTERS["retrieve_genetic_evidence"].run(
        _ctx("retrieve_genetic_evidence", board=_candidates_board())
    )
    assert result.status == "error" and "timeout" in result.error


def _genetic_and_stage_board():
    def stage(count):
        return {
            "drug_record_count": count,
            "stages_present": ["PHASE_2"] if count else [],
            "drugs": [],
        }

    scores = [("T1", 0.7), ("T2", 0.4), ("T3", None), ("T4", 0.9), ("T5", 0.8)]
    return {
        "genetic_evidence": {
            "disease_id": PDAC["id"],
            "datatype_id": "genetic_association",
            "targets": [
                {
                    "target_id": tid,
                    "target_symbol": tid,
                    "target_name": tid,
                    "genetic_evidence_score": score,
                }
                for tid, score in scores
            ],
            "limitations": [],
        },
        "clinical_stage_evidence": {
            "by_target": {"T1": stage(0), "T2": stage(2), "T3": stage(0), "T5": stage(0)},
            "failed_targets": {"T4": "timeout"},
        },
    }


def test_supported_without_drug_separates_none_from_unknown():
    result = ADAPTERS["filter_supported_without_clinical_drug"].run(
        _ctx("filter_supported_without_clinical_drug", board=_genetic_and_stage_board())
    )
    assert [t["target_id"] for t in result.payload["targets"]] == ["T1", "T5"]
    assert sorted(result.payload["excluded_target_ids"]) == ["T2", "T3"]
    assert result.payload["unevaluable_target_ids"] == ["T4"]
    assert result.status == "partial_error"
    OUTPUT_MODELS["filter_supported_without_clinical_drug"].model_validate(result.payload)


def test_supported_without_drug_applies_an_extra_score_filter():
    result = ADAPTERS["filter_supported_without_clinical_drug"].run(
        _ctx(
            "filter_supported_without_clinical_drug",
            board=_genetic_and_stage_board(),
            filters={"genetic_evidence_score": {"gte": 0.75}},
        )
    )
    assert [t["target_id"] for t in result.payload["targets"]] == ["T5"]
    assert "T1" in result.payload["excluded_target_ids"]


def test_supported_without_drug_rejects_an_unknown_filter_field():
    result = ADAPTERS["filter_supported_without_clinical_drug"].run(
        _ctx(
            "filter_supported_without_clinical_drug",
            board=_genetic_and_stage_board(),
            filters={"not_a_field": {"gte": 1}},
        )
    )
    assert result.status == "error" and "not_a_field" in result.error


def test_supported_without_drug_needs_both_inputs():
    result = ADAPTERS["filter_supported_without_clinical_drug"].run(
        _ctx("filter_supported_without_clinical_drug", board={})
    )
    assert result.status == "error" and "clinical_stage_evidence" in result.error


# ---------- first-in-class opportunity ----------

def test_genetic_and_somatic_evidence_stay_separate(monkeypatch):
    _fake_datatype_scores(
        monkeypatch,
        {
            "T1": {"datatype_scores": [
                {"datatype_id": "genetic_association", "score": 0.7},
                {"datatype_id": "somatic_mutation", "score": 0.9},
            ]},
            "T2": {"datatype_scores": [{"datatype_id": "somatic_mutation", "score": 0.4}]},
        },
    )
    result = ADAPTERS["retrieve_genetic_and_somatic_evidence"].run(
        _ctx("retrieve_genetic_and_somatic_evidence", board=_candidates_board())
    )
    rows = {
        r["target_id"]: (r["genetic_evidence_score"], r["somatic_evidence_score"])
        for r in result.payload["targets"]
    }
    assert rows == {"T1": (0.7, 0.9), "T2": (None, 0.4), "T3": (None, None)}
    assert result.provides == ["genetic_somatic_evidence"]
    OUTPUT_MODELS["retrieve_genetic_and_somatic_evidence"].model_validate(result.payload)


def _first_in_class_board():
    def drugs(count):
        return {"total_drug_count": count, "fetched_drug_count": count, "drugs": []}

    rows = [
        ("T1", 0.9, None),
        ("T2", 0.2, 0.3),
        ("T3", None, None),
        ("T4", None, 0.8),
        ("T5", 0.7, 0.6),
        ("T6", 0.95, 0.9),
    ]
    return {
        "genetic_somatic_evidence": {
            "disease_id": PDAC["id"],
            "datatype_ids": ["genetic_association", "somatic_mutation"],
            "targets": [
                {
                    "target_id": tid,
                    "target_symbol": tid,
                    "target_name": tid,
                    "genetic_evidence_score": g,
                    "somatic_evidence_score": s,
                }
                for tid, g, s in rows
            ],
            "limitations": [],
        },
        "target_drug_evidence": {
            "by_target": {
                "T1": drugs(0), "T2": drugs(0), "T3": drugs(0), "T5": drugs(3), "T6": drugs(0),
            },
            "failed_targets": {"T4": "timeout"},
        },
    }


_STRONG = {"max_genetic_somatic_score": {"gte": 0.6}}


def test_first_in_class_keeps_strong_evidence_with_no_drug_record():
    result = ADAPTERS["filter_first_in_class_candidates"].run(
        _ctx("filter_first_in_class_candidates", board=_first_in_class_board(), filters=_STRONG)
    )
    assert [t["target_id"] for t in result.payload["targets"]] == ["T1", "T6"]
    assert sorted(result.payload["excluded_target_ids"]) == ["T2", "T3", "T5"]
    assert result.payload["unevaluable_target_ids"] == ["T4"]
    assert result.status == "partial_error"
    OUTPUT_MODELS["filter_first_in_class_candidates"].model_validate(result.payload)


def test_first_in_class_needs_an_explicit_strength_filter():
    result = ADAPTERS["filter_first_in_class_candidates"].run(
        _ctx("filter_first_in_class_candidates", board=_first_in_class_board())
    )
    assert result.status == "error" and "evidence-strength" in result.error


def test_first_in_class_rejects_an_unknown_filter_field():
    result = ADAPTERS["filter_first_in_class_candidates"].run(
        _ctx(
            "filter_first_in_class_candidates",
            board=_first_in_class_board(),
            filters={"not_a_field": {"gte": 1}},
        )
    )
    assert result.status == "error" and "not_a_field" in result.error


def test_first_in_class_needs_both_inputs():
    result = ADAPTERS["filter_first_in_class_candidates"].run(
        _ctx("filter_first_in_class_candidates", board={}, filters=_STRONG)
    )
    assert result.status == "error" and "target_drug_evidence" in result.error


def _opportunity_board():
    def candidate(tid, score):
        return {
            "target_id": tid,
            "target_symbol": tid,
            "max_genetic_somatic_score": score,
            "known_drug_count": 0,
        }

    def assessed(*values):
        return {
            "assessment_count": len(values),
            "assessments": [
                {"modality": f"M{i}", "value": value, "label": ""}
                for i, value in enumerate(values)
            ],
        }

    return {
        "first_in_class_candidates": {
            "targets": [candidate("T1", 0.9), candidate("T6", 0.95), candidate("T7", 0.8)]
        },
        "tractability_evidence": {
            "by_target": {"T1": assessed(True, False), "T6": assessed(False)},
            "failed_targets": {},
        },
    }


_OPPORTUNITY_RANKING = {
    "primary_metric": "max_genetic_somatic_score",
    "direction": "descending",
    "secondary_metrics": [],
}


def test_opportunities_are_ranked_by_the_metric_and_warn_about_tractability():
    result = ADAPTERS["rank_opportunities"].run(
        _ctx("rank_opportunities", board=_opportunity_board(), ranking=_OPPORTUNITY_RANKING)
    )
    ranked = {o["target_id"]: o for o in result.payload["opportunities"]}
    assert [o["target_id"] for o in result.payload["opportunities"]] == ["T6", "T1", "T7"]
    assert ranked["T1"]["warnings"] == []
    assert any("No tractability bucket" in w for w in ranked["T6"]["warnings"])
    assert any("not assessed" in w for w in ranked["T7"]["warnings"])
    OUTPUT_MODELS["rank_opportunities"].model_validate(result.payload)


def test_opportunity_ranking_needs_a_spec_and_candidates():
    no_spec = ADAPTERS["rank_opportunities"].run(
        _ctx("rank_opportunities", board=_opportunity_board())
    )
    assert no_spec.status == "error" and "ranking" in no_spec.error
    no_candidates = ADAPTERS["rank_opportunities"].run(
        _ctx("rank_opportunities", board={}, ranking=_OPPORTUNITY_RANKING)
    )
    assert no_candidates.status == "error" and "first_in_class_candidates" in no_candidates.error


def test_empty_candidate_list_gives_no_records():
    result = ADAPTERS["rank_opportunities"].run(
        _ctx(
            "rank_opportunities",
            board={"first_in_class_candidates": {"targets": []}},
            ranking=_OPPORTUNITY_RANKING,
        )
    )
    assert result.status == "no_records"


def test_opportunity_ranking_refuses_a_text_metric():
    result = ADAPTERS["rank_opportunities"].run(
        _ctx(
            "rank_opportunities",
            board=_opportunity_board(),
            ranking={"primary_metric": "target_symbol", "direction": "ascending", "secondary_metrics": []},
        )
    )
    assert result.status == "error" and "not numeric" in result.error
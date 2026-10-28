import pytest

import execution.adapters as adapters
import query_opentargets
from execution.adapters import CheckIndicationAdapter, RetrieveApprovedIndicationsAdapter
from execution.contracts import StepContext
from execution.records import OUTPUT_MODELS
from query_opentargets import OpenTargetsError

PDAC = "MONDO_0005184"
OLAPARIB = "CHEMBL521686"


def _entry():
    return {
        "drug_id": OLAPARIB,
        "drug_name": "OLAPARIB",
        "total_indication_count": 2,
        "fetched_indication_count": 2,
        "indications": [
            {
                "disease_id": PDAC,
                "disease_name": "pancreatic ductal adenocarcinoma",
                "max_clinical_stage": "PHASE_2",
            },
            {
                "disease_id": "MONDO_0021040",
                "disease_name": "pancreatic neoplasm",
                "max_clinical_stage": "APPROVAL",
            },
        ],
    }


def _evidence():
    return {"by_drug": {OLAPARIB: _entry()}, "failed_drugs": {}}


def _ctx(operation, board, entities=None):
    return StepContext(
        step_id="s", operation=operation, entities=entities or {}, board=board
    )


def _ancestors(monkeypatch):
    monkeypatch.setattr(
        adapters,
        "get_disease_ancestors",
        lambda efo_id: {
            "disease_id": efo_id,
            "disease_name": "",
            "ancestor_ids": ["MONDO_0021040", "MONDO_0009831"],
        },
    )


def test_retrieve_collects_per_drug_and_keeps_failures(monkeypatch):
    def fake(chembl_id):
        if chembl_id == "BAD":
            raise OpenTargetsError("boom")
        return _entry()

    monkeypatch.setattr(adapters, "get_drug_indications_by_id", fake)
    board = {
        "resolved_drug": {
            "d1": {"id": OLAPARIB, "name": "OLAPARIB", "symbol": ""},
            "d2": {"id": "BAD", "name": "x", "symbol": ""},
        }
    }
    result = RetrieveApprovedIndicationsAdapter().run(
        _ctx("retrieve_approved_indications", board)
    )
    assert result.status == "partial_error"
    assert list(result.payload["by_drug"]) == [OLAPARIB]
    assert "BAD" in result.payload["failed_drugs"]
    assert result.provides == ["approval_evidence"]
    OUTPUT_MODELS["retrieve_approved_indications"].model_validate(result.payload)


def test_retrieve_without_a_resolved_drug_is_an_error():
    result = RetrieveApprovedIndicationsAdapter().run(
        _ctx("retrieve_approved_indications", {})
    )
    assert result.status == "error"


def test_check_reports_broader_term_approval_separately(monkeypatch):
    _ancestors(monkeypatch)
    board = {
        "approval_evidence": _evidence(),
        "resolved_disease": {
            "disease_1": {"id": PDAC, "name": "pancreatic ductal adenocarcinoma", "symbol": ""}
        },
    }
    result = CheckIndicationAdapter().run(_ctx("check_indication", board))
    assert result.status == "success"
    row = result.payload["checks"][0]
    assert row["status"] == "approved_on_broader_term"
    assert row["exact_stage"] == "PHASE_2"
    assert result.provides == ["indication_check"]
    OUTPUT_MODELS["check_indication"].model_validate(result.payload)


def test_check_fails_loudly_when_ancestors_are_unavailable(monkeypatch):
    def broken(efo_id):
        raise OpenTargetsError("down")

    monkeypatch.setattr(adapters, "get_disease_ancestors", broken)
    board = {
        "approval_evidence": _evidence(),
        "resolved_disease": {"disease_1": {"id": PDAC, "name": "", "symbol": ""}},
    }
    result = CheckIndicationAdapter().run(_ctx("check_indication", board))
    assert result.status == "error"
    assert "down" in result.error


def test_check_resolves_the_disease_mention_when_no_resolve_step_ran(monkeypatch):
    _ancestors(monkeypatch)
    monkeypatch.setattr(
        adapters,
        "resolve_mention",
        lambda mention, entity, entity_id=None, resolver=None: {
            "id": PDAC,
            "name": mention,
            "symbol": "",
        },
    )
    entities = {
        "disease_1": {
            "entity_type": "disease",
            "mention": "pancreatic ductal adenocarcinoma",
        }
    }
    result = CheckIndicationAdapter().run(
        _ctx("check_indication", {"approval_evidence": _evidence()}, entities)
    )
    assert result.payload["disease_id"] == PDAC


def test_drug_indications_are_normalized(monkeypatch):
    def fake_graphql(query, variables):
        assert variables == {"chemblId": OLAPARIB}
        return {
            "drug": {
                "id": OLAPARIB,
                "name": "OLAPARIB",
                "indications": {
                    "count": 2,
                    "rows": [
                        {
                            "maxClinicalStage": "APPROVAL",
                            "disease": {"id": "MONDO_0021040", "name": "pancreatic neoplasm"},
                        },
                        {
                            "maxClinicalStage": "PHASE_2",
                            "disease": {"id": PDAC, "name": "pancreatic ductal adenocarcinoma"},
                        },
                    ],
                },
            }
        }

    monkeypatch.setattr(query_opentargets, "_graphql", fake_graphql)
    result = query_opentargets.get_drug_indications_by_id(OLAPARIB)
    assert result["total_indication_count"] == result["fetched_indication_count"] == 2
    assert result["indications"][0] == {
        "disease_id": "MONDO_0021040",
        "disease_name": "pancreatic neoplasm",
        "max_clinical_stage": "APPROVAL",
    }


def test_unknown_drug_is_an_error_not_an_empty_list(monkeypatch):
    monkeypatch.setattr(query_opentargets, "_graphql", lambda query, variables: {"drug": None})
    with pytest.raises(OpenTargetsError):
        query_opentargets.get_drug_indications_by_id("CHEMBL0")
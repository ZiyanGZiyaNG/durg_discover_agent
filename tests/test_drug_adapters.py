import pytest

import execution.adapters as adapters
import query_opentargets
from execution.adapters import ADAPTERS, RetrieveDrugTargetsAdapter
from execution.contracts import StepContext
from execution.records import OUTPUT_MODELS
from query_opentargets import EntityNotFoundError, OpenTargetsError, resolve_entity

OLAPARIB = "CHEMBL521686"


def _ctx(operation, board, entities=None):
    return StepContext(
        step_id="s", operation=operation, entities=entities or {}, board=board
    )


def test_drug_name_resolves_to_the_exact_match_only(monkeypatch):
    hits = [
        {"id": OLAPARIB, "name": "OLAPARIB"},
        {"id": "CHEMBL4650351", "name": "PARPI"},
    ]
    monkeypatch.setattr(
        query_opentargets, "_graphql", lambda query, variables: {"search": {"hits": hits}}
    )
    assert resolve_entity("Olaparib", "drug") == {
        "id": OLAPARIB,
        "name": "OLAPARIB",
        "symbol": "",
    }


def test_drug_id_is_verified_and_unknown_id_is_an_error(monkeypatch):
    monkeypatch.setattr(
        query_opentargets,
        "_graphql",
        lambda query, variables: {"drug": {"id": OLAPARIB, "name": "OLAPARIB"}},
    )
    assert resolve_entity("x", "drug", OLAPARIB)["id"] == OLAPARIB
    monkeypatch.setattr(query_opentargets, "_graphql", lambda query, variables: {"drug": None})
    with pytest.raises(EntityNotFoundError):
        resolve_entity("x", "drug", "CHEMBL0")


def test_drug_targets_are_deduplicated_across_mechanisms(monkeypatch):
    parp1 = {"id": "ENSG00000143799", "approvedSymbol": "PARP1", "approvedName": "poly(ADP-ribose) polymerase 1"}
    parp2 = {"id": "ENSG00000129484", "approvedSymbol": "PARP2", "approvedName": "poly(ADP-ribose) polymerase 2"}
    rows = [
        {"actionType": "INHIBITOR", "mechanismOfAction": "PARP 1, 2 inhibitor", "targets": [parp1, parp2]},
        {"actionType": "INHIBITOR", "mechanismOfAction": "PARP 1 inhibitor", "targets": [parp1]},
    ]
    monkeypatch.setattr(
        query_opentargets,
        "_graphql",
        lambda query, variables: {
            "drug": {"id": OLAPARIB, "name": "OLAPARIB", "mechanismsOfAction": {"rows": rows}}
        },
    )
    result = query_opentargets.get_drug_targets_by_id(OLAPARIB)
    assert result["mechanism_count"] == 2
    assert [t["target_symbol"] for t in result["targets"]] == ["PARP1", "PARP2"]
    assert result["mechanisms"][1]["target_ids"] == ["ENSG00000143799"]


def test_unknown_drug_is_an_error_not_an_empty_list(monkeypatch):
    monkeypatch.setattr(query_opentargets, "_graphql", lambda query, variables: {"drug": None})
    with pytest.raises(OpenTargetsError):
        query_opentargets.get_drug_targets_by_id("CHEMBL0")


def _record(drug_id):
    return {
        "drug_id": drug_id,
        "drug_name": "OLAPARIB",
        "mechanism_count": 1,
        "mechanisms": [
            {"action_type": "INHIBITOR", "mechanism_of_action": "PARP inhibitor", "target_ids": ["ENSG00000143799"]}
        ],
        "targets": [
            {"target_id": "ENSG00000143799", "target_symbol": "PARP1", "target_name": "poly(ADP-ribose) polymerase 1"}
        ],
    }


def test_retrieve_drug_targets_keeps_failed_drugs_separate(monkeypatch):
    def fake(chembl_id):
        if chembl_id == "BAD":
            raise OpenTargetsError("boom")
        return _record(chembl_id)

    monkeypatch.setattr(adapters, "get_drug_targets_by_id", fake)
    board = {
        "resolved_drug": {
            "d1": {"id": OLAPARIB, "name": "OLAPARIB", "symbol": ""},
            "d2": {"id": "BAD", "name": "x", "symbol": ""},
        }
    }
    result = RetrieveDrugTargetsAdapter().run(_ctx("retrieve_drug_targets", board))
    assert result.status == "partial_error"
    assert list(result.payload["by_drug"]) == [OLAPARIB]
    assert "BAD" in result.payload["failed_drugs"]
    assert result.provides == ["target_drug_evidence"]
    OUTPUT_MODELS["retrieve_drug_targets"].model_validate(result.payload)


def test_resolve_drug_uses_the_drug_entity(monkeypatch):
    monkeypatch.setattr(
        adapters,
        "resolve_mention",
        lambda mention, entity, entity_id=None, resolver=None: {
            "id": OLAPARIB,
            "name": "OLAPARIB",
            "symbol": "",
        },
    )
    entities = {"drug_1": {"entity_type": "drug", "mention": "Olaparib"}}
    result = ADAPTERS["resolve_drug"].run(_ctx("resolve_drug", {}, entities))
    assert result.status == "success"
    assert result.provides == ["resolved_drug"]
    assert result.payload["drug_1"]["id"] == OLAPARIB
    OUTPUT_MODELS["resolve_drug"].model_validate(result.payload)
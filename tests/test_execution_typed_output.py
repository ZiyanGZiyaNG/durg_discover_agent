from unittest.mock import patch

from execution.adapters import ADAPTERS
from execution.compiler import run_plan
from execution.records import OUTPUT_MODELS, DiscoveredTargets
from schemas import QueryManifestV2
from test_execution_acceptance import FOUND, _drugs, _fake_resolve, _tractability
from test_natural_language_planner import _valid_payload


def test_every_operation_with_an_adapter_has_a_typed_output():
    assert set(OUTPUT_MODELS) == set(ADAPTERS)


def test_open_targets_shaped_payload_with_extra_fields_is_accepted():
    # Same shape query_opentargets returns, including fields we do not model.
    payload = {
        "disease_query": "pancreatic carcinoma",
        "disease_id": "EFO_1",
        "returned_target_count": 1,
        "total_associated_target_count": 7656,
        "targets": [
            {
                "rank": 1,
                "target_id": "ENSG00000133703",
                "target_symbol": "KRAS",
                "target_name": "KRAS proto-oncogene",
                "overall_association_score": 0.82,
                "datasource_scores": [{"datasource_id": "x", "score": 0.5}],
            }
        ],
        "limitations": ["note"],
    }
    assert (
        DiscoveredTargets.model_validate(payload)
        .targets[0]
        .target_id.startswith("ENSG")
    )


def test_payload_that_breaks_its_model_is_an_error_and_stops_later_steps():
    plan = QueryManifestV2.model_validate(_valid_payload())
    broken = {**FOUND, "targets": [{"target_symbol": "T1"}]}  # row has no target_id
    with (
        patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
        patch("execution.adapters.discover_disease_targets", return_value=broken),
        patch("execution.adapters.get_target_drug_evidence_by_ids", side_effect=_drugs),
        patch(
            "execution.adapters.get_target_tractability_by_id",
            side_effect=_tractability,
        ),
    ):
        trace = run_plan(plan)

    assert trace[0].status == "success"
    assert trace[1].status == "error"
    assert "DiscoveredTargets" in trace[1].error
    assert "target_id" in trace[1].error
    assert trace[1].payload is None
    assert [r.status for r in trace[2:]] == ["not_executed"] * 4

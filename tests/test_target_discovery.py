import json
from unittest.mock import patch

from agent import discovery_output_path, run_target_discovery_as_json
from query_opentargets import discover_disease_targets


@patch("query_opentargets._graphql")
@patch(
    "query_opentargets._search",
    return_value={"id": "MONDO_0004975", "name": "Alzheimer disease"},
)
def test_discovery_sorts_targets_and_preserves_datasource_scores(_search, graphql):
    graphql.return_value = {
        "disease": {
            "id": "MONDO_0004975",
            "name": "Alzheimer disease",
            "associatedTargets": {
                "count": 50,
                "rows": [
                    {
                        "target": {
                            "id": "ENSG2",
                            "approvedSymbol": "B",
                            "approvedName": "Target B",
                        },
                        "score": 0.7,
                        "datasourceScores": [{"id": "eva", "score": 0.8}],
                    },
                    {
                        "target": {
                            "id": "ENSG1",
                            "approvedSymbol": "A",
                            "approvedName": "Target A",
                        },
                        "score": 0.9,
                        "datasourceScores": [],
                    },
                ],
            },
        }
    }
    result = discover_disease_targets("Alzheimer's disease", top_n=2)
    assert result["targets"][0]["target_symbol"] == "A"
    assert result["targets"][0]["rank"] == 1
    assert result["targets"][1]["datasource_scores"][0]["datasource_id"] == "eva"
    assert result["total_associated_target_count"] == 50


@patch("agent1.discover_disease_targets")
def test_discovery_cli_json_has_strict_agent1_metadata(discover):
    discover.return_value = {
        "disease_query": "Alzheimer's disease",
        "disease_id": "MONDO_0004975",
        "disease_name": "Alzheimer disease",
        "requested_target_count": 10,
        "returned_target_count": 1,
        "total_associated_target_count": 1,
        "targets": [
            {
                "rank": 1,
                "target_id": "ENSG1",
                "target_symbol": "APP",
                "target_name": "amyloid beta precursor protein",
                "overall_association_score": 0.8,
                "datasource_scores": [],
            }
        ],
        "limitations": [],
    }
    payload = json.loads(run_target_discovery_as_json("Alzheimer's disease", 10))
    assert payload["producer"] == "agent1"
    assert payload["target_discovery"]["targets"][0]["target_symbol"] == "APP"


def test_discovery_filename_uses_disease_even_with_old_filename():
    result = discovery_output_path(
        "pancreatic cancer",
        10,
        r"D:\NCKU_Drug_Discovery\outputs\Alzheimer_top10_targets.json",
    )
    assert result == (
        r"D:\NCKU_Drug_Discovery\outputs\pancreatic_cancer_top10_targets.json"
    )


def test_discovery_filename_sanitizes_apostrophe():
    result = discovery_output_path("Alzheimer's disease", 10, None)
    assert result == "Alzheimer_s_disease_top10_targets.json"


def test_discovery_existing_output_directory_is_not_replaced_by_parent():
    result = discovery_output_path(
        "pancreatic cancer",
        10,
        r"D:\NCKU_Drug_Discovery\outputs",
    )
    assert result == (
        r"D:\NCKU_Drug_Discovery\outputs\pancreatic_cancer_top10_targets.json"
    )


def test_discovery_filename_uses_disease_even_with_old_filename(tmp_path):
    old_output = tmp_path / "Alzheimer_top10_targets.json"
    result = discovery_output_path("pancreatic cancer", 10, str(old_output))
    assert result == str(tmp_path / "pancreatic_cancer_top10_targets.json")


def test_discovery_existing_output_directory_is_not_replaced_by_parent(tmp_path):
    result = discovery_output_path("pancreatic cancer", 10, str(tmp_path))
    assert result == str(tmp_path / "pancreatic_cancer_top10_targets.json")
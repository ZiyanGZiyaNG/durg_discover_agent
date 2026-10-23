import json

import pytest

from agent import Agent1EvidenceInputError, load_agent1_evidence


SAMPLE = {
    "disease_query": "PDAC",
    "target_query": "KRAS",
    "disease_id": "MONDO_0005184",
    "disease_name": "PDAC",
    "target_id": "ENSG00000133703",
    "target_symbol": "KRAS",
    "target_name": "KRAS",
    "overall_association_score": 0.5,
    "datasource_scores": [],
    "rank_within_fetched_targets": 1,
    "fetched_target_count": 10,
    "limitations": [],
}


METADATA = {
    "producer": "agent1",
    "run_id": "f7db8184-cba8-4a45-b101-39c659f2209a",
    "generated_at": "2026-08-06T00:00:00+00:00",
}


def test_agent2_loader_rejects_raw_evidence_without_agent1_metadata(tmp_path):
    path = tmp_path / "agent1.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    with pytest.raises(Agent1EvidenceInputError):
        load_agent1_evidence(str(path))


def test_agent2_loader_accepts_strict_agent1_output(tmp_path):
    path = tmp_path / "agent1.json"
    path.write_text(
        json.dumps({**METADATA, "agent_1_evidence": SAMPLE}), encoding="utf-8"
    )
    assert load_agent1_evidence(str(path)).disease_id == "MONDO_0005184"


def test_agent2_loader_rejects_wrong_producer(tmp_path):
    path = tmp_path / "wrong-producer.json"
    path.write_text(
        json.dumps({**METADATA, "producer": "manual", "agent_1_evidence": SAMPLE}),
        encoding="utf-8",
    )
    with pytest.raises(Agent1EvidenceInputError):
        load_agent1_evidence(str(path))

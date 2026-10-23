from unittest.mock import patch

from agent1 import BioinformaticsEvidenceAgent
from schemas import DiseaseTargetInput


SAMPLE = {
    "disease_query": "pancreatic ductal adenocarcinoma",
    "target_query": "KRAS",
    "disease_id": "MONDO_0005184",
    "disease_name": "pancreatic ductal adenocarcinoma",
    "target_id": "ENSG00000133703",
    "target_symbol": "KRAS",
    "target_name": "KRas proto-oncogene, GTPase",
    "overall_association_score": 0.512,
    "datasource_scores": [{"datasource_id": "cancer_gene_census", "score": 0.81}],
    "rank_within_fetched_targets": 1,
    "fetched_target_count": 100,
    "limitations": ["Test fixture."],
}


@patch("agent1.get_disease_target_evidence", return_value=SAMPLE)
def test_agent_1_returns_stable_schema(_mock):
    result = BioinformaticsEvidenceAgent().run(
        DiseaseTargetInput(disease="pancreatic ductal adenocarcinoma", target="KRAS")
    )
    assert result.disease_id == "MONDO_0005184"
    assert result.target_id == "ENSG00000133703"
    assert result.datasource_scores[0].score == 0.81
    _mock.assert_called_once_with(
        "pancreatic ductal adenocarcinoma",
        "KRAS",
        max_evidence_records=500,
    )

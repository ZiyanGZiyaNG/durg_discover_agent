from execution.analysis import classify_indication

PDAC = "MONDO_0005184"
ANCESTORS = ["MONDO_0021040", "MONDO_0009831"]



def _row(disease_id, stage, name=None):
    return {
        "disease_id": disease_id,
        "disease_name": name or disease_id,
        "max_clinical_stage": stage,
    }


def test_approval_on_the_exact_disease():
    result = classify_indication([_row(PDAC, "APPROVAL")], PDAC, ANCESTORS)
    assert result["status"] == "approved_exact"


def test_approval_only_on_a_broader_term_is_not_an_exact_approval():
    rows = [_row(PDAC, "PHASE_2"), _row("MONDO_0021040", "APPROVAL", "pancreatic neoplasm")]
    result = classify_indication(rows, PDAC, ANCESTORS)
    assert result["status"] == "approved_on_broader_term"
    assert result["exact_stage"] == "PHASE_2"
    assert result["broader_approvals"] == [
        {"disease_id": "MONDO_0021040", "disease_name": "pancreatic neoplasm"}
    ]


def test_studied_but_not_approved():
    result = classify_indication([_row(PDAC, "PHASE_2")], PDAC, ANCESTORS)
    assert result["status"] == "studied_not_approved"
    assert result["exact_stage"] == "PHASE_2"


def test_broader_term_in_trials_only_does_not_count_as_approval():
    result = classify_indication([_row("MONDO_0009831", "PHASE_3")], PDAC, ANCESTORS)
    assert result["status"] == "not_found_in_source"


def test_absence_is_unconfirmed_not_negative():
    result = classify_indication([_row("MONDO_0000001", "APPROVAL")], PDAC, ANCESTORS)
    assert result["status"] == "not_found_in_source"
    assert result["exact_stage"] is None
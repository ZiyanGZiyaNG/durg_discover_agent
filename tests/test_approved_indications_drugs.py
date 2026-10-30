from execution.adapters import _drugs_for_step
from execution.contracts import StepContext


def _ctx(board):
    return StepContext(
        step_id="s", operation="retrieve_approved_indications", entities={}, board=board
    )


def test_drugs_fall_back_to_target_drug_evidence():
    board = {
        "target_drug_evidence": {
            "by_target": {
                "T1": {"drugs": [{"drug_id": "D1", "drug_name": "a"}, {"drug_id": "D2"}]},
                "T2": {"drugs": [{"drug_id": "D1", "drug_name": "a"}]},
            }
        }
    }
    drugs = _drugs_for_step(_ctx(board))
    assert [d["id"] for d in drugs] == ["D1", "D2"]


def test_explicit_drugs_win():
    board = {
        "resolved_drug": {"entity_9": {"id": "X", "name": "x"}},
        "target_drug_evidence": {"by_target": {"T1": {"drugs": [{"drug_id": "D1"}]}}},
    }
    ctx = StepContext(
        step_id="s",
        operation="retrieve_approved_indications",
        entities={"entity_9": {"entity_type": "drug"}},
        board=board,
    )
    assert [d["id"] for d in _drugs_for_step(ctx)] == ["X"]


def test_no_drugs_gives_empty_list():
    assert _drugs_for_step(_ctx({})) == []

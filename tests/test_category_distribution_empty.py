from execution.contracts import StepContext
from execution.trial_analysis_adapters import AggregateCategoryDistributionAdapter


def _run(inputs):
    ctx = StepContext(
        step_id="s",
        operation="aggregate_category_distribution",
        entities={},
        aggregation={"category_distribution": {"metrics": ["categories"]}},
        inputs=inputs,
    )
    return AggregateCategoryDistributionAdapter().run(ctx)


def test_empty_upstream_is_no_records():
    result = _run({"classified": {"records": []}})
    assert result.status == "no_records"
    assert result.total_count == 0


def test_missing_field_is_still_an_error():
    assert _run({"classified": {"records": [{"other": 1}]}}).status == "error"


def test_list_valued_field_counts_each_element():
    rows = [{"categories": ["safety", "efficacy"]}, {"categories": ["safety"]}]
    result = _run({"classified": {"records": rows}})
    counts = {c["category"]: c["category_count"] for c in result.payload["distributions"]["categories"]}
    assert result.status == "success"
    assert counts == {"safety": 2, "efficacy": 1}

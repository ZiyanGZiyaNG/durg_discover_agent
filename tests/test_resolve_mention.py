import pytest

from query_opentargets import (
    EntityNotFoundError,
    EntityResolutionError,
    OpenTargetsError,
    mention_variants,
    resolve_mention,
)

MENTION = "胰臟癌（pancreatic ductal adenocarcinoma, PDAC）"


def test_variants_whole_text_first_then_bracketed_then_outside():
    assert mention_variants(MENTION) == [
        MENTION,
        "pancreatic ductal adenocarcinoma",
        "PDAC",
        "胰臟癌",
    ]


def test_plain_mention_has_a_single_variant():
    assert mention_variants("KRAS") == ["KRAS"]


def test_falls_back_to_the_bracketed_english_name():
    def resolver(name, kind, entity_id=None):
        if name == "pancreatic ductal adenocarcinoma":
            return {"id": "EFO_X", "name": name, "symbol": ""}
        raise EntityNotFoundError(f"not found: {name}")

    assert resolve_mention(MENTION, "disease", resolver=resolver)["id"] == "EFO_X"


def test_explicit_id_skips_variant_search():
    calls = []

    def resolver(name, kind, entity_id=None):
        calls.append((name, entity_id))
        return {"id": entity_id, "name": "x", "symbol": ""}

    resolve_mention(MENTION, "disease", "MONDO_1", resolver=resolver)
    assert calls == [(MENTION, "MONDO_1")]


def test_ambiguous_variants_report_merged_candidates():
    def resolver(name, kind, entity_id=None):
        raise EntityResolutionError(
            name, kind, [{"id": "A", "name": "a"}, {"id": "B", "name": "b"}]
        )

    with pytest.raises(EntityResolutionError) as excinfo:
        resolve_mention(MENTION, "disease", resolver=resolver)
    assert {c["id"] for c in excinfo.value.candidates} == {"A", "B"}


def test_nothing_found_lists_what_was_tried():
    def resolver(name, kind, entity_id=None):
        raise EntityNotFoundError("nope")

    with pytest.raises(EntityNotFoundError) as excinfo:
        resolve_mention(MENTION, "disease", resolver=resolver)
    assert "PDAC" in str(excinfo.value)


def test_service_errors_are_not_retried_per_variant():
    calls = []

    def resolver(name, kind, entity_id=None):
        calls.append(name)
        raise OpenTargetsError("boom")

    with pytest.raises(OpenTargetsError):
        resolve_mention(MENTION, "disease", resolver=resolver)
    assert len(calls) == 1

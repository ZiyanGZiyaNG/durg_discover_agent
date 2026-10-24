import json
import time

import pytest
from langchain_core.messages import AIMessage

from natural_language_planner import DirectJsonPlanAgent, NaturalLanguageResearchPlanner
from test_natural_language_planner import QUESTION, _valid_payload


class _ServiceError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class APIConnectionError(Exception):  # the class name must match the openai SDK's
    pass


class _FlakyModel:
    def __init__(self, failures, error=lambda: _ServiceError(500)):
        self.failures, self.error, self.calls = failures, error, 0

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error()
        return AIMessage(content=json.dumps(_valid_payload(), ensure_ascii=False))


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    monkeypatch.setenv("PLANNER_SERVICE_RETRY_BASE_SECONDS", "0")


def _invoke(model):
    return DirectJsonPlanAgent(model).invoke({"messages": []})["structured_response"]


def test_500_is_retried_inside_one_call():
    model = _FlakyModel(failures=2)
    assert _invoke(model).original_question == QUESTION
    assert model.calls == 3


def test_retries_run_out():
    model = _FlakyModel(failures=99)
    with pytest.raises(_ServiceError):
        _invoke(model)
    assert model.calls == 4


def test_404_is_not_retried():
    model = _FlakyModel(failures=99, error=lambda: _ServiceError(404))
    with pytest.raises(_ServiceError):
        _invoke(model)
    assert model.calls == 1


def test_429_and_connection_errors_are_retried():
    for error in (lambda: _ServiceError(429), APIConnectionError):
        model = _FlakyModel(failures=1, error=error)
        _invoke(model)
        assert model.calls == 2


class _SlowThenFast:
    def __init__(self):
        self.calls = 0

    def invoke(self, request, config=None):
        self.calls += 1
        if self.calls == 1:
            time.sleep(0.5)
        return {"structured_response": _valid_payload()}


def _slow_run(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("PLANNER_ATTEMPT_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("PLANNER_TOTAL_TIMEOUT_SECONDS", "30")
    return NaturalLanguageResearchPlanner(agent=_SlowThenFast()).run_with_report(
        QUESTION, max_attempts=3
    )


def test_hosted_provider_continues_after_a_timeout(monkeypatch):
    report = _slow_run(monkeypatch, "nim")
    assert report.status == "passed"
    assert [a.model_generation_timeout for a in report.attempts] == [True, False]


def test_ollama_still_stops_after_a_timeout(monkeypatch):
    report = _slow_run(monkeypatch, "ollama")
    assert report.status == "failed" and len(report.attempts) == 1

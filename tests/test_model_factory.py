import pytest

from model_factory import describe_llm_settings, get_ollama_num_predict, get_ollama_settings


@pytest.mark.parametrize(
    ("profile", "model", "context"),
    [
        ("lightweight", "qwen3:8b", 4096),
        ("development", "qwen3:14b", 4096),
        ("validation", "qwen3:32b", 2048),
    ],
)
def test_ollama_profiles(monkeypatch, profile, model, context):
    monkeypatch.setenv("OLLAMA_PROFILE", profile)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    assert get_ollama_settings() == (model, context)


def test_explicit_model_and_context_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_PROFILE", "development")
    monkeypatch.setenv("OLLAMA_MODEL", "custom:model")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "3072")
    assert get_ollama_settings() == ("custom:model", 3072)


def test_validation_context_is_hardware_capped(monkeypatch):
    monkeypatch.setenv("OLLAMA_PROFILE", "validation")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    with pytest.raises(ValueError, match="capped"):
        get_ollama_settings()


def test_ollama_generation_has_a_safe_default_limit(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
    assert get_ollama_num_predict() == 4096


def test_ollama_generation_limit_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "3072")
    assert get_ollama_num_predict() == 3072


def test_ollama_generation_limit_rejects_tiny_values(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "100")
    with pytest.raises(ValueError, match="at least 256"):
        get_ollama_num_predict()


def test_describe_llm_settings_reports_model_without_secrets(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nim")
    monkeypatch.setenv("NIM_MODEL", "nvidia/example-model")
    monkeypatch.setenv("NIM_API_KEY", "nvapi-secret-value")
    monkeypatch.delenv("NIM_BASE_URL", raising=False)
    info = describe_llm_settings()
    assert info["provider"] == "nim"
    assert info["model"] == "nvidia/example-model"
    assert info["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert "secret" not in str(info)


def test_describe_llm_settings_for_ollama_uses_profile(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_PROFILE", "lightweight")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    info = describe_llm_settings()
    assert (info["model"], info["num_ctx"]) == ("qwen3:8b", 4096)

"""Configurable chat-model backend for local Ollama, Anthropic, or NVIDIA NIM."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaProfile:
    """A hardware-aware local model preset."""

    model: str
    num_ctx: int
    purpose: str


OLLAMA_PROFILES = {
    # RTX 5060 Laptop (8 GB VRAM): fast deployment/fallback profile.
    "lightweight": OllamaProfile("qwen3:8b", 4096, "deployment"),
    # Daily development baseline; may split work between VRAM and system RAM.
    "development": OllamaProfile("qwen3:14b", 4096, "feature development"),
    # Quality comparison only. 32B runs mostly from 32 GB system RAM and is slow.
    "validation": OllamaProfile("qwen3:32b", 2048, "quality validation"),
}


def get_ollama_settings() -> tuple[str, int]:
    """Resolve an Ollama model/context from a named profile and env overrides."""

    profile_name = os.getenv("OLLAMA_PROFILE", "development").strip().lower()
    try:
        profile = OLLAMA_PROFILES[profile_name]
    except KeyError as exc:
        choices = ", ".join(OLLAMA_PROFILES)
        raise ValueError(
            f"Unsupported OLLAMA_PROFILE={profile_name!r}. Choose {choices}."
        ) from exc

    model = os.getenv("OLLAMA_MODEL", profile.model).strip()
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", str(profile.num_ctx)))
    if num_ctx < 512:
        raise ValueError("OLLAMA_NUM_CTX must be at least 512.")
    if profile_name == "validation" and num_ctx > 4096:
        raise ValueError(
            "The validation profile is capped at 4096 tokens on this 32 GB machine."
        )
    return model, num_ctx


def get_ollama_num_predict() -> int:
    """Bound reasoning plus answer tokens to prevent context-shift loops."""

    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "4096"))
    if num_predict < 256:
        raise ValueError("OLLAMA_NUM_PREDICT must be at least 256.")
    return num_predict


def get_chat_model() -> Any:
    """Return the configured LangChain chat model.

    Ollama is the default and requires no paid API key. Set LLM_PROVIDER to
    ``anthropic`` or ``nim`` to use a hosted backend.
    """

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "Ollama support is not installed. Run: pip install -r requirements.txt"
            ) from exc
        model, num_ctx = get_ollama_settings()
        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=get_ollama_num_predict(),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        )

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic."
            )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic support is not installed. Run: "
                "pip install -r requirements-anthropic.txt"
            ) from exc
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=temperature,
        )

    if provider == "nim":
        api_key = os.getenv("NIM_API_KEY")
        if not api_key:
            raise RuntimeError("NIM_API_KEY is required when LLM_PROVIDER=nim.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "NIM support is not installed. Run: "
                "pip install langchain-nvidia-ai-endpoints"
            ) from exc
        return ChatOpenAI(
            model=os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct"),
              api_key=api_key,
              base_url=os.getenv("NIM_BASE_URL") or "https://integrate.api.nvidia.com/v1",
              temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider!r}. "
        "Choose 'ollama', 'anthropic' or 'nim'."
    )
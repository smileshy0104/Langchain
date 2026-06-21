"""Shared model configuration for Human-in-the-Loop examples."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model


def load_env_files() -> None:
    """Load .env files from this example folder and nearby examples."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    current_dir = Path(__file__).parent
    for env_path in (
        current_dir / ".env",
        current_dir.parent / "langchain_context_engineering_examples" / ".env",
        current_dir.parent / "langchain" / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path, override=False)


load_env_files()


MODEL_PROVIDER = os.getenv("LANGCHAIN_MODEL_PROVIDER", "custom").lower()
DEFAULT_MODEL_NAME = os.getenv("LANGCHAIN_DEFAULT_MODEL", "gpt-5.5")
DEFAULT_TEMPERATURE = float(os.getenv("LANGCHAIN_DEFAULT_TEMPERATURE", "0.2"))


def require_env(key: str) -> str:
    value = os.getenv(key, "").strip()
    if not value or "your-" in value.lower():
        raise ValueError(f"请先配置环境变量 {key}")
    return value


def get_custom_base_url() -> str:
    base_url = require_env("CUSTOM_MODEL_BASE_URL").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def create_configured_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Create a chat model from environment variables.

    For OpenAI-compatible vendors, set:
    - LANGCHAIN_MODEL_PROVIDER=custom
    - CUSTOM_MODEL_BASE_URL=https://ai-api.kkidc.com
    - CUSTOM_MODEL_API_KEY=...
    - LANGCHAIN_DEFAULT_MODEL=gpt-5.5
    """

    model_name = model_name or DEFAULT_MODEL_NAME
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature

    if MODEL_PROVIDER == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=require_env("CUSTOM_MODEL_API_KEY"),
            base_url=get_custom_base_url(),
        )

    return init_chat_model(
        model_name,
        model_provider=MODEL_PROVIDER,
        temperature=temperature,
    )

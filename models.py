"""Central model factory: resolve MODEL_NAME into a LangChain chat model.

Supported MODEL_NAME formats (set via env var or passed explicitly):

  ollama:<model-id>          Native Ollama path (LangChain ChatOllama).
                             - With OLLAMA_API_KEY set  -> Ollama Cloud
                               (https://ollama.com, no local server needed).
                             - Without a key            -> local Ollama server
                               (http://localhost:11434).
                             Examples: "ollama:gpt-oss:120b", "ollama:qwen3:32b",
                                       "ollama:gemma4:31b"

  ollama-cloud:<model-id>    Explicit Ollama Cloud path via the OpenAI-
                             compatible endpoint (https://ollama.com/v1).
                             Requires OLLAMA_API_KEY. Uses langchain-openai
                             ChatOpenAI under the hood, which is the documented
                             Ollama Cloud integration:
                               https://docs.ollama.com/api/openai-compatibility
                             Example: "ollama-cloud:gpt-oss:120b"

  <provider>:<model>         Anything else (e.g. "openai:gpt-4o-mini",
                             "google_genai:gemini-2.0-flash",
                             "anthropic:claude-haiku-4-5") is passed through
                             to LangChain's init_chat_model unchanged.

Env vars:
  MODEL_NAME        Default model string (see formats above).
  OLLAMA_API_KEY    API key from https://ollama.com/settings/keys.
                    Required for any Ollama Cloud usage.
  OLLAMA_BASE_URL   Optional override. Defaults to https://ollama.com/v1
                    for the ollama-cloud: path. Ignored for native ollama:
                    path unless explicitly set (ChatOllama auto-selects
                    cloud vs local based on OLLAMA_API_KEY).

All three agent modules (agent_react, agent_plan_execute, agent_hybrid)
build their LangChain agents through get_chat_model(), so setting
MODEL_NAME + OLLAMA_API_KEY is enough to run a real model.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
OLLAMA_CLOUD_API_URL = "https://ollama.com/api"


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. "
            + (
                "Create one at https://ollama.com/settings/keys and "
                "add it to your .env (see .env.example)."
                if name == "OLLAMA_API_KEY"
                else "Set it in your environment or .env (see .env.example)."
            )
        )
    return value


def resolve_model_name(model_name: str | None = None) -> str:
    """Return the effective MODEL_NAME or raise a helpful error."""
    raw = (model_name or os.environ.get("MODEL_NAME", "")).strip()
    if not raw:
        raise RuntimeError(
            "MODEL_NAME is not set. Examples:\n"
            '  MODEL_NAME=ollama-cloud:gpt-oss:120b  (Ollama Cloud, recommended)\n'
            '  MODEL_NAME=ollama:gpt-oss:120b        (Cloud if OLLAMA_API_KEY set)\n'
            "  MODEL_NAME=openai:gpt-4o-mini         (non-Ollama provider)\n"
            "See .env.example."
        )
    return raw


def describe_model_config(model_name: str | None = None) -> dict:
    """Return a redacted summary of how a model string will be built.

    Never includes the API key value. Useful for logging / debugging.
    """
    raw = resolve_model_name(model_name)
    if raw.startswith("ollama-cloud:"):
        return {
            "kind": "ollama-cloud-openai-compat",
            "model": raw.split(":", 1)[1],
            "base_url": os.environ.get("OLLAMA_BASE_URL", OLLAMA_CLOUD_BASE_URL),
            "has_api_key": bool(os.environ.get("OLLAMA_API_KEY", "").strip()),
        }
    if raw.startswith("ollama:"):
        return {
            "kind": "ollama-native",
            "model": raw.split(":", 1)[1],
            "uses_cloud": bool(os.environ.get("OLLAMA_API_KEY", "").strip()),
            "base_url_override": os.environ.get("OLLAMA_BASE_URL", ""),
        }
    return {"kind": "init_chat_model-passthrough", "model": raw}


def get_chat_model(model_name: str | None = None, temperature: float = 0.0, **kwargs):
    """Build a LangChain chat model from a MODEL_NAME string.

    Returns an instance suitable for create_agent(model=...).
    Pass temperature=0 for deterministic tool-calling behaviour.
    Extra kwargs are forwarded to the underlying chat-model constructor
    (e.g. timeout, max_tokens / num_predict).
    """
    raw = resolve_model_name(model_name)

    # --- Explicit Ollama Cloud via OpenAI-compatible endpoint ---
    if raw.startswith("ollama-cloud:"):
        model_id = raw.split(":", 1)[1].strip()
        if not model_id:
            raise RuntimeError(
                "MODEL_NAME='ollama-cloud:' is missing a model id. "
                "Example: MODEL_NAME=ollama-cloud:gpt-oss:120b "
                "(use the name from https://ollama.com/api/tags, "
                'e.g. "gemma4:31b").'
            )
        api_key = _require_env("OLLAMA_API_KEY")
        base_url = os.environ.get("OLLAMA_BASE_URL", OLLAMA_CLOUD_BASE_URL)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "ollama-cloud: models need the 'langchain-openai' package. "
                "Install it with: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(
            model=model_id, api_key=api_key, base_url=base_url,  # type: ignore[arg-type]
            temperature=temperature, **kwargs,
        )

    # --- Native Ollama (cloud when OLLAMA_API_KEY is set, else local) ---
    if raw.startswith("ollama:"):
        model_id = raw.split(":", 1)[1].strip()
        if not model_id:
            raise RuntimeError(
                "MODEL_NAME='ollama:' is missing a model id. "
                "Example: MODEL_NAME=ollama:gpt-oss:120b"
            )
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "ollama: models need the 'langchain-ollama' package. "
                "Install it with: pip install langchain-ollama"
            ) from exc
        params: dict = {"model": model_id, "temperature": temperature}
        # Only override base_url if the user explicitly set one; otherwise
        # ChatOllama selects Ollama Cloud (OLLAMA_API_KEY present) vs the
        # local daemon automatically.
        base_url_override = os.environ.get("OLLAMA_BASE_URL", "").strip()
        if base_url_override:
            params["base_url"] = base_url_override
        params.update(kwargs)
        return ChatOllama(**params)

    # --- Any other provider: delegate to LangChain's resolver ---
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:
        raise RuntimeError(
            "Could not import init_chat_model from langchain. "
            "Upgrade with: pip install -U langchain"
        ) from exc
    return init_chat_model(raw, temperature=temperature, **kwargs)

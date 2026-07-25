"""
API client and response generation for Anthropic, OpenAI, Google Gemini,
Moonshot Kimi, OpenRouter Kimi/GLM, direct Z.AI GLM, xAI Grok, DeepSeek,
DashScope Qwen, and DashScope GLM models.
"""

import os
import time
import anthropic
import openai
from google import genai
from xai_sdk import Client as XAIClient
from xai_sdk.chat import assistant as xai_assistant, user as xai_user, system as xai_system

from costing import normalize_usage

# Valid Anthropic cache TTLs. "5m" is the API default; "1h" is GA (no beta
# header) and costs 2x base input on writes vs 1.25x for "5m", but reads are
# 0.1x for both. Use "1h" when the gap between a model's consecutive turns can
# exceed 5 minutes (e.g. high/max effort runs, where a single peer turn can take
# longer than the 5m window and expire the prefix before it is ever re-read).
CACHE_TTLS = ("5m", "1h")
DEFAULT_CACHE_TTL = "1h"

MODEL_ALIASES = {
    # Convenience aliases for the current Opus API ids.
    # Claude Opus 5 — fixed id with no date suffix (GA July 2026).
    "opus-5": "claude-opus-5",
    "claude-opus5": "claude-opus-5",
    "claude-opus-4.8": "claude-opus-4-8",
    "claude-opus-4.7": "claude-opus-4-7",
    # Claude Fable 5 — Anthropic's highest-tier widely-available model (GA June 2026).
    "fable-5": "claude-fable-5",
    "fable": "claude-fable-5",
    # Claude Mythos 5
    "mythos-5": "claude-mythos-5",
    "mythos": "claude-mythos-5",
    # ChatGPT "GPT-5.5 Instant" is exposed in the API as the floating `chat-latest`
    # alias (which currently points at gpt-5.5 instant). Map several friendly forms.
    "gpt-5.5-instant": "chat-latest",
    "gpt-5.5-chat-latest": "chat-latest",
    # ChatGPT "GPT-5.2 Instant" maps to the public API chat-latest model id.
    "gpt-5.2-instant": "gpt-5.2-chat-latest",
    # Accept either Moonshot naming style for Kimi K2.5.
    "moonshot/kimi-k2.5": "kimi-k2.5",
    # OpenRouter aliases for Kimi/GLM.
    "moonshotai/kimi-k2.5": "openrouter/moonshotai/kimi-k2.5",
    "openrouter/kimi-k2.5": "openrouter/moonshotai/kimi-k2.5",
    "z-ai/glm-5": "openrouter/z-ai/glm-5",
    "openrouter/glm-5": "openrouter/z-ai/glm-5",
    "zai/glm-5": "zai/glm-5",
    # Allow a separator-free shorthand for the exact Qwen 3.5 MoE model.
    "qwen3.5-397ba17b": "qwen3.5-397b-a17b",
    # Current xAI model ids no longer include the older beta infix.
    "grok-4.20-beta-0309-reasoning": "grok-4.20-0309-reasoning",
    "grok-4.20-beta-0309-non-reasoning": "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-beta-0309": "grok-4.20-multi-agent-0309",
    "grok-4.3-latest": "grok-4.3",
    "grok-43": "grok-4.3",
}

# Maps API model IDs to the known underlying model version for transcript metadata.
# Update these when providers rotate what a model ID points to.
MODEL_VERSIONS = {
    "deepseek-reasoner": "DeepSeek-V3.2 (reasoning mode)",
    "deepseek-chat": "DeepSeek-V3.2",
    "chatgpt-4o-latest": "GPT-4o (latest snapshot)",
    "gpt-5-chat-latest": "GPT-5 (chat-latest snapshot)",
    "gpt-5.5": "GPT-5.5",
    "gpt-5.5-pro": "GPT-5.5 Pro",
    "gpt-5.6-sol": "GPT-5.6 SOL",
    "gpt-5.2-chat-latest": "GPT-5.2 (chat-latest snapshot)",
    "chat-latest": "GPT-5.5 Instant (chat-latest floating alias)",
    "grok-4.3": "Grok 4.3",
    "grok-4.20-0309-reasoning": "Grok 4.20 (0309 reasoning snapshot)",
    "grok-4.20-0309-non-reasoning": "Grok 4.20 (0309 non-reasoning snapshot)",
    "grok-4.20-multi-agent-0309": "Grok 4.20 multi-agent (0309 snapshot)",
    "qwen3.5-397b-a17b": "Qwen3.5 397B A17B",
    "glm-5": "GLM-5",
    "openrouter/moonshotai/kimi-k2.5": "Kimi K2.5",
    "openrouter/z-ai/glm-5": "GLM-5",
    "zai/glm-5": "GLM-5",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
}


def get_model_version(model: str) -> str | None:
    """Return the known underlying model version, or None if unknown."""
    return MODEL_VERSIONS.get(model)


# Anthropic models that support fixed-budget or adaptive thinking.
THINKING_MODELS = {
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
}

# Anthropic models that use adaptive thinking (thinking: {type: "adaptive"}).
# These models never use budget_tokens.
ADAPTIVE_THINKING_MODELS = {
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
}

# Adaptive-thinking models that reject temperature/sampling params entirely
# and support the "xhigh" (and "max") effort level.
XHIGH_EFFORT_MODELS = {
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
}


def supports_xhigh_effort(model: str) -> bool:
    """Return True if the model accepts the xhigh adaptive-thinking effort."""
    return model in XHIGH_EFFORT_MODELS

# OpenAI models
OPENAI_MODELS = {
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.6-sol",
    "gpt-5.2-2025-12-11",
    "chatgpt-4o-latest",
    "gpt-5-chat-latest",
    "chat-latest",  # GPT-5.5 Instant — floating alias for current ChatGPT default
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
}

# Moonshot Kimi models.
MOONSHOT_MODELS = {
    "kimi-k2.5",
    "kimi-k2-thinking",
    "kimi-k2-thinking-turbo",
}

MOONSHOT_THINKING_MODELS = {
    "kimi-k2.5",
    "kimi-k2-thinking",
    "kimi-k2-thinking-turbo",
}

OPENROUTER_MODELS = {
    "openrouter/moonshotai/kimi-k2.5",
    "openrouter/z-ai/glm-5",
}

OPENROUTER_THINKING_MODELS = {
    "openrouter/moonshotai/kimi-k2.5",
    "openrouter/z-ai/glm-5",
}

ZAI_MODELS = {
    "zai/glm-5",
}

ZAI_THINKING_MODELS = {
    "zai/glm-5",
}

# Google Gemini models - use dynamic detection via prefixes
# The API has 30+ models (gemini-2.5-flash, gemini-3-flash-preview, gemma-3-*, etc.)
# Updated list can be fetched via: client.models.list()

# xAI Grok models that support reasoning/thinking
GROK_THINKING_MODELS = {
    "grok-4.3",
    "grok-4.20-beta-0309-reasoning",
    "grok-4.20-0309-reasoning",
    "grok-4.20-multi-agent-0309",
    "grok-4-fast-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-1-reasoning",
    "grok-4",
    "grok-4-1",
}

# Known xAI Grok models that are explicitly non-reasoning.
GROK_NON_REASONING_MODELS = {
    "grok-4.20-beta-0309-non-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4-fast-non-reasoning",
    "grok-4-1-fast-non-reasoning",
}

# DeepSeek models
DEEPSEEK_MODELS = {
    "deepseek-reasoner",
    "deepseek-chat",
}

DEEPSEEK_THINKING_MODELS = {
    "deepseek-reasoner",
}

# DashScope / Qwen models
QWEN_MODELS = {
    "qwen3.5-397b-a17b",
}

QWEN_THINKING_MODELS = {
    "qwen3.5-397b-a17b",
}

# DashScope / GLM models
GLM_MODELS = {
    "glm-5",
}

GLM_THINKING_MODELS = {
    "glm-5",
}

GEMINI_THINKING_PREFIXES = (
    "gemini-2.5-",
    "gemini-3",
    "gemini-pro-latest",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "deep-research-",
)

# Model-specific max_tokens limits
MODEL_MAX_TOKENS = {
    "claude-3-opus-20240229": 4096,
    "claude-3-5-haiku-20241022": 8192,
    "claude-3-haiku-20240307": 4096,
    "gpt-4o-2024-05-13": 4096,
}


def create_anthropic_client() -> anthropic.Anthropic:
    """Create and return an Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    return anthropic.Anthropic(api_key=api_key)


def create_openai_client() -> openai.OpenAI:
    """Create and return an OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    return openai.OpenAI(api_key=api_key)


def create_moonshot_client() -> openai.OpenAI:
    """Create and return a Moonshot client via the OpenAI-compatible API."""
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError(
            "MOONSHOT_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    base_url = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
    # Use explicit provider-specific backoff for Moonshot overloads rather than
    # the SDK's generic 429 retry behavior.
    return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)


def create_openrouter_client() -> openai.OpenAI:
    """Create and return an OpenRouter client."""
    api_key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPEN_ROUTER_API_KEY / OPENROUTER_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    base_url = os.environ.get("OPEN_ROUTER_BASE_URL") or os.environ.get(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def create_zai_client() -> openai.OpenAI:
    """Create and return a direct Z.AI client."""
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise ValueError(
            "ZAI_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    base_url = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def create_gemini_client():
    """Create and return a Gemini client."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    return genai.Client(api_key=api_key)


def create_xai_client() -> XAIClient:
    """Create and return an xAI client."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError(
            "XAI_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    return XAIClient(api_key=api_key, timeout=3600)


def create_deepseek_client() -> openai.OpenAI:
    """Create and return a DeepSeek client via the OpenAI-compatible API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def create_dashscope_client() -> openai.OpenAI:
    """Create and return a DashScope client via the OpenAI-compatible API."""
    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("ALI_BABA_API_KEY")
        or os.environ.get("BAILIAN_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY / ALI_BABA_API_KEY / BAILIAN_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def create_glm_client() -> openai.OpenAI:
    """Create and return a DashScope client for mainland GLM models."""
    api_key = (
        os.environ.get("ALI_BABA_CN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("ALI_BABA_API_KEY")
        or os.environ.get("BAILIAN_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "ALI_BABA_CN_API_KEY / DASHSCOPE_API_KEY / ALI_BABA_API_KEY / BAILIAN_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )
    base_url = os.environ.get(
        "GLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def normalize_model_name(model: str) -> str:
    """Map local-friendly aliases to provider model ids."""
    return MODEL_ALIASES.get(model, model)


def is_openai_model(model: str) -> bool:
    """Check if a model is an OpenAI model."""
    if is_zai_model(model):
        return False
    if is_openrouter_model(model):
        return False
    if is_moonshot_model(model):
        return False
    if is_deepseek_model(model):
        return False
    if is_qwen_model(model):
        return False
    if is_glm_model(model):
        return False
    # Prefer heuristics over a static allowlist so new OpenAI ids keep working.
    if model in OPENAI_MODELS:
        return True
    # Common OpenAI prefixes.
    if model.startswith(("gpt-", "chatgpt-")):
        return True
    # Reasoning-style model ids (e.g. o1, o3) are also OpenAI.
    if model.startswith(("o1", "o3", "o4")):
        return True
    return False


def is_moonshot_model(model: str) -> bool:
    """Check if a model is a Moonshot Kimi model."""
    return model in MOONSHOT_MODELS or model.startswith(("kimi-", "moonshot/kimi-"))


def is_openrouter_model(model: str) -> bool:
    """Check if a model routes through OpenRouter."""
    return model in OPENROUTER_MODELS or model.startswith("openrouter/")


def is_zai_model(model: str) -> bool:
    """Check if a model routes directly through Z.AI."""
    return model in ZAI_MODELS or model.startswith("zai/")


def is_gemini_model(model: str) -> bool:
    """Check if a model is a Google Gemini model."""
    # Google AI models use these prefixes (dynamic list via client.models.list())
    google_prefixes = (
        "gemini-",      # Gemini 2.0/2.5/3.0 models
        "gemma-",       # Gemma open models
        "nano-banana-", # Specialized models
        "deep-research-", # Research models
    )
    return model.startswith(google_prefixes)


def is_grok_model(model: str) -> bool:
    """Check if a model is an xAI Grok model."""
    return model.startswith("grok-")


def is_deepseek_model(model: str) -> bool:
    """Check if a model is a DeepSeek model."""
    return model in DEEPSEEK_MODELS or model.startswith("deepseek-")


def is_qwen_model(model: str) -> bool:
    """Check if a model is a DashScope Qwen model."""
    return model in QWEN_MODELS


def is_glm_model(model: str) -> bool:
    """Check if a model is a DashScope GLM model."""
    return model in GLM_MODELS


def uses_openai_responses_api(model: str) -> bool:
    """Check if an OpenAI model should use the Responses API."""
    return model.startswith("gpt-5") and not model.endswith("chat-latest")


def uses_openai_reasoning(model: str) -> bool:
    """Check if an OpenAI model should use GPT-5 reasoning defaults."""
    return uses_openai_responses_api(model)


def supports_thinking(model: str) -> bool:
    """Check if a model supports provider-side thinking/reasoning."""
    if model in GROK_NON_REASONING_MODELS:
        return False
    return (
        model in THINKING_MODELS
        or model in MOONSHOT_THINKING_MODELS
        or model in OPENROUTER_THINKING_MODELS
        or model in ZAI_THINKING_MODELS
        or model in GROK_THINKING_MODELS
        or model in DEEPSEEK_THINKING_MODELS
        or model in QWEN_THINKING_MODELS
        or model in GLM_THINKING_MODELS
        or model.startswith(GEMINI_THINKING_PREFIXES)
        or uses_openai_reasoning(model)
    )


def uses_adaptive_thinking(model: str) -> bool:
    """Check if a model should use Anthropic adaptive thinking."""
    return model in ADAPTIVE_THINKING_MODELS


def uses_gemini_thinking_level(model: str) -> bool:
    """Check if a Gemini model should use thinking_level instead of budget."""
    return model.startswith("gemini-3")


def thinking_budget_to_effort(thinking_budget: int, model: str | None = None) -> str:
    """
    Map the integer budget control onto Anthropic's adaptive effort levels.

    Per Anthropic's Effort docs the full ladder for adaptive thinking is:
        low -> medium -> high -> xhigh -> max
    where `xhigh` is available on Opus 4.7, 4.8, and Opus 5. For models that
    don't accept xhigh (e.g. Opus 4.6), the xhigh tier collapses to `high`
    rather than `max` — Anthropic explicitly warns max can over-think and waste
    tokens on workloads that aren't genuinely frontier.

    The budget thresholds are this repo's convention, not Anthropic's; effort
    is a behavioral signal, not a token quota.
    """
    if thinking_budget <= 1024:
        return "low"
    if thinking_budget <= 4096:
        return "medium"
    if thinking_budget <= 12000:
        return "high"
    if thinking_budget <= 32000:
        if model is None or supports_xhigh_effort(model):
            return "xhigh"
        return "high"
    return "max"


def get_max_tokens(model: str, default: int = 16000) -> int:
    """Get the maximum tokens allowed for a model."""
    return MODEL_MAX_TOKENS.get(model, default)


def _ensure_nonempty_text_response(response_text: str) -> str:
    """Normalize empty or effectively empty model replies to a safe placeholder."""
    if response_text and response_text.strip() not in {"", ".", "...", "*", "-", "[silence]"}:
        return response_text
    return "*silence*"


def generate_response(
    anthropic_client: anthropic.Anthropic,
    openai_client: openai.OpenAI,
    moonshot_client: openai.OpenAI,
    openrouter_client: openai.OpenAI,
    zai_client: openai.OpenAI,
    qwen_client: openai.OpenAI,
    glm_client: openai.OpenAI,
    conversation: list,
    system_prompt: str,
    temperature: float = 1.0,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = None,
    thinking_budget: int | None = None,
    effort: str | None = None,
    gemini_client = None,
    xai_client = None,
    deepseek_client = None,
    cache_ttl: str = DEFAULT_CACHE_TTL
) -> tuple[list, str, str, dict]:
    """
    Generate a response using Anthropic, OpenAI, Gemini, or xAI Grok models.

    Returns:
        tuple: (content_blocks, reasoning_text, response_text, usage)
        - content_blocks: Full message content array
        - reasoning_text: Extracted thinking text for display (empty string if not supported)
        - response_text: Extracted response text for display
        - usage: Normalized usage dict (may have None token counts if not available)
    """
    # Use model-specific max_tokens if not specified
    if max_tokens is None:
        max_tokens = get_max_tokens(model)

    # Route to appropriate API
    if is_deepseek_model(model):
        return _generate_deepseek_response(
            deepseek_client, conversation, system_prompt, temperature, model, max_tokens
        )
    elif is_zai_model(model):
        return _generate_zai_response(
            zai_client, conversation, system_prompt, temperature, model, max_tokens
        )
    elif is_openrouter_model(model):
        return _generate_openrouter_response(
            openrouter_client, conversation, system_prompt, temperature, model, max_tokens, thinking_budget
        )
    elif is_glm_model(model):
        return _generate_glm_response(
            glm_client, conversation, system_prompt, temperature, model, max_tokens
        )
    elif is_qwen_model(model):
        return _generate_qwen_response(
            qwen_client, conversation, system_prompt, temperature, model, max_tokens, thinking_budget
        )
    elif is_grok_model(model):
        return _generate_grok_response(
            xai_client, conversation, system_prompt, temperature, model, max_tokens
        )
    elif is_moonshot_model(model):
        return _generate_moonshot_response(
            moonshot_client, conversation, system_prompt, temperature, model, max_tokens
        )
    elif is_gemini_model(model):
        return _generate_gemini_response(
            gemini_client, conversation, system_prompt, temperature, model, max_tokens, thinking_budget
        )
    elif is_openai_model(model):
        return _generate_openai_response(
            openai_client, conversation, system_prompt, temperature, model, max_tokens, thinking_budget, effort
        )
    else:
        return _generate_anthropic_response(
            anthropic_client, conversation, system_prompt, temperature,
            model, max_tokens, thinking_budget, effort, cache_ttl=cache_ttl
        )


def _cache_control(ttl: str) -> dict:
    """Build a cache_control block. '5m' omits ttl (the API default)."""
    control = {"type": "ephemeral"}
    if ttl and ttl != "5m":
        control["ttl"] = ttl
    return control


def _apply_anthropic_cache_breakpoints(
    system_prompt: str,
    conversation: list,
    ttl: str = DEFAULT_CACHE_TTL,
) -> tuple:
    """
    Attach cache_control breakpoints so Anthropic prompt caching covers the
    (stable) system prompt and the rolling conversation prefix.

    Anthropic caches the longest exact prefix ending at a cache_control marker
    that exceeds the model's minimum cacheable size. Tagging the last user
    message creates a checkpoint each turn re-uses on the next — but only if the
    checkpoint is still live when the next same-model turn starts. Slow (high/max
    effort) turns can exceed the 5m default TTL, so `ttl` defaults to "1h": with
    5m, every turn misses and re-writes the whole prefix (strictly worse than no
    cache). See CACHE_TTLS for the cost trade-off.
    """
    control = _cache_control(ttl)

    if system_prompt:
        system_param = [
            {"type": "text", "text": system_prompt, "cache_control": control}
        ]
    else:
        system_param = system_prompt

    if not conversation:
        return system_param, conversation

    last = conversation[-1]
    last_content = last.get("content")
    if isinstance(last_content, str):
        new_content = [
            {"type": "text", "text": last_content, "cache_control": control}
        ]
        messages_param = list(conversation[:-1]) + [{**last, "content": new_content}]
    else:
        # Unexpected shape (last message not a plain string); skip the message
        # breakpoint rather than risk mutating SDK content blocks.
        messages_param = conversation

    return system_param, messages_param


def _generate_anthropic_response(
    client: anthropic.Anthropic,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
    thinking_budget: int | None,
    effort: str | None = None,
    cache_ttl: str = DEFAULT_CACHE_TTL
) -> tuple[list, str, str, dict]:
    """Generate response using Anthropic API."""
    if uses_adaptive_thinking(model):
        use_thinking = True
    else:
        use_thinking = supports_thinking(model) and thinking_budget is not None and thinking_budget > 0

    system_param, messages_param = _apply_anthropic_cache_breakpoints(
        system_prompt, conversation, ttl=cache_ttl
    )

    api_params = {
        "model": model,
        "system": system_param,
        "max_tokens": max_tokens,
        "messages": messages_param,
        "temperature": temperature,
    }

    if use_thinking:
        if uses_adaptive_thinking(model):
            # Opus 4.7, 4.8, and Opus 5 silently default thinking.display to
            # "omitted", which makes thinking tokens invisible (still billed).
            # We always want the summary in the transcript, so opt into
            # "summarized" explicitly. (On Opus 4.6 / Sonnet 4.6 "summarized" is
            # already the default, so setting it doesn't change behavior there.)
            api_params["thinking"] = {
                "type": "adaptive",
                "display": "summarized",
            }
            api_params["output_config"] = {
                "effort": effort or "high"
            }
            # Fable 5, Opus 4.7, 4.8, and Opus 5 remove temperature/sampling params
            # entirely (sending temperature returns 400). Opus 4.6 still accepts temperature=1.0.
            if model in XHIGH_EFFORT_MODELS:
                del api_params["temperature"]
            else:
                api_params["temperature"] = 1.0
        else:
            # Fixed-budget thinking requires temperature=1.0; the API rejects any
            # other value, so override whatever was requested once thinking is on.
            api_params["temperature"] = 1.0
            api_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }

    # The Anthropic SDK's typed `final_message.usage` model strips
    # `output_tokens_details` (which carries `thinking_tokens`). The raw
    # `message_delta` event in the stream still has it, so capture it here and
    # patch it back onto the normalized usage below.
    captured_output_token_details: dict | None = None

    with client.messages.stream(**api_params) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if hasattr(event.delta, "type"):
                    if event.delta.type == "thinking_delta":
                        print(event.delta.thinking, end="", flush=True)
                    elif event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)
            elif event.type == "message_delta":
                evt_usage = getattr(event, "usage", None)
                if evt_usage is not None and hasattr(evt_usage, "model_dump"):
                    dumped = evt_usage.model_dump()
                    otd = dumped.get("output_tokens_details") if isinstance(dumped, dict) else None
                    if isinstance(otd, dict):
                        captured_output_token_details = otd

        # Stream may be closed on context exit; get the final message while still inside.
        final_message = stream.get_final_message()
    content_blocks = final_message.content
    usage = normalize_usage(provider="anthropic", model=model, usage_obj=getattr(final_message, "usage", None))

    # Patch thinking_tokens back in from the captured stream-event details.
    if captured_output_token_details is not None and usage.get("thinking_tokens") is None:
        tt = captured_output_token_details.get("thinking_tokens")
        if tt is not None:
            try:
                tt_int = int(tt)
            except (TypeError, ValueError):
                tt_int = None
            if tt_int is not None:
                usage["thinking_tokens"] = tt_int
                if usage.get("details") is None:
                    usage["details"] = {}
                usage["details"]["thinking_tokens"] = tt_int

    # Extract thinking and text blocks, and rebuild content_blocks with valid text
    from anthropic.types import TextBlock, ThinkingBlock

    reasoning_text = ""
    response_text = ""
    new_content_blocks = []

    for block in content_blocks:
        if hasattr(block, 'thinking'):
            reasoning_text = block.thinking
            new_content_blocks.append(block)
        elif hasattr(block, 'text'):
            response_text = block.text
            # Keep text blocks that have substantial content
            if response_text and response_text.strip() not in ["", ".", "...", "*", "-", "[silence]"]:
                new_content_blocks.append(block)

    # If response_text is empty or only punctuation/whitespace, use a placeholder
    # (Anthropic API requires non-empty, substantial text content blocks)
    if not response_text or response_text.strip() in ["", ".", "...", "*", "-", "[silence]"]:
        response_text = "*silence*"
        # Add a text block with meaningful placeholder
        new_content_blocks.append(TextBlock(type="text", text="*silence*"))

    return new_content_blocks, reasoning_text, response_text, usage


def _generate_openai_response(
    client: openai.OpenAI,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
    _thinking_budget: int,
    effort: str | None = None,
) -> tuple[list, str, str, dict]:
    """Generate response using OpenAI API."""
    # Convert Anthropic-format history into plain text messages.
    messages = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]

        # Handle content blocks (from Anthropic) or strings
        if isinstance(content, list):
            # Extract text from content blocks
            text_content = ""
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_content += block.get("text") or ""
                    elif "text" in block:
                        text_content += block.get("text") or ""
                elif hasattr(block, "type") and block.type == "text":
                    text_content += block.text
                elif hasattr(block, "text"):
                    text_content += block.text
            messages.append({"role": role, "content": text_content})
        else:
            messages.append({"role": role, "content": content})

    if uses_openai_responses_api(model):
        # GPT-5-family Responses API:
        # - effort: none/minimal/low/medium/high/xhigh (docs default: medium). OpenAI has
        #   no "max" — translate Anthropic's "max" to the highest supported level "xhigh"
        #   so a shared EFFORT_A/B default works for both providers without per-model wiring.
        # - summary: "auto"/"detailed"/"concise". Without this set, no reasoning summary
        #   appears in the response even though the model still thinks (and we're billed).
        #   Opting in to "auto" mirrors Anthropic's display="summarized".
        _OPENAI_VALID_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
        _OPENAI_EFFORT_TRANSLATE = {"max": "xhigh"}
        raw_effort = (effort or "medium").lower()
        resolved_effort = _OPENAI_EFFORT_TRANSLATE.get(raw_effort, raw_effort)
        if resolved_effort not in _OPENAI_VALID_EFFORTS:
            resolved_effort = "medium"
        api_params = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": resolved_effort, "summary": "auto"},
        }
        if system_prompt:
            api_params["instructions"] = system_prompt

        response_text = ""
        final_response = None
        for _attempt in range(3):
            with client.responses.stream(**api_params) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)
                        response_text += event.delta
                    elif event.type in {"response.completed", "response.incomplete"}:
                        # A response can be incomplete after reaching an output
                        # or context limit. Preserve its streamed text and usage.
                        final_response = event.response

            # A usable partial answer must not be retried because doing so would
            # duplicate streamed text. Retry only a completely empty stream.
            if final_response is not None or response_text:
                break

        if not response_text and final_response is not None:
            response_text = final_response.output_text

        response_text = _ensure_nonempty_text_response(response_text)

        reasoning_summaries = []
        for item in getattr(final_response, "output", []) if final_response is not None else []:
            if getattr(item, "type", None) != "reasoning":
                continue
            for summary in getattr(item, "summary", []) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning_summaries.append(text)

        content_blocks = [{"type": "text", "text": response_text}]
        usage = normalize_usage(
            provider="openai",
            model=model,
            usage_obj=getattr(final_response, "usage", None) if final_response is not None else None,
        )
        return content_blocks, "\n\n".join(reasoning_summaries), response_text, usage

    # Stream the response
    # Newer models use max_completion_tokens instead of max_tokens
    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True
    }

    if system_prompt:
        api_params["messages"] = [{"role": "system", "content": system_prompt}, *messages]

    # Use max_completion_tokens for newer models (GPT-4o and later, plus the
    # floating `chat-latest` alias that points to current ChatGPT default).
    if (
        "gpt-5" in model
        or "chatgpt-4o" in model
        or model == "gpt-4o"
        or model == "chat-latest"
    ):
        api_params["max_completion_tokens"] = max_tokens
    else:
        api_params["max_tokens"] = max_tokens

    # Attempt to include usage in the streaming response if supported by the SDK/API.
    # If unsupported, we still return response text but usage will be None.
    usage_obj = None
    try:
        stream = client.chat.completions.create(**api_params, stream_options={"include_usage": True})
    except TypeError:
        stream = client.chat.completions.create(**api_params)

    response_text = ""
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage
        # Some stream events may be usage-only or otherwise have no choices.
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
            response_text += content

    response_text = _ensure_nonempty_text_response(response_text)

    # Return Anthropic-compatible content blocks as plain dicts so the same history
    # structure can be fed back into either provider.
    content_blocks = [{"type": "text", "text": response_text}]

    usage = normalize_usage(provider="openai", model=model, usage_obj=usage_obj)
    return content_blocks, "", response_text, usage


def _generate_moonshot_response(
    client: openai.OpenAI,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
) -> tuple[list, str, str, dict]:
    """Generate response using Moonshot's OpenAI-compatible API."""
    messages = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, list):
            text_content = ""
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_content += block.get("text") or ""
                    elif "text" in block:
                        text_content += block.get("text") or ""
                elif hasattr(block, "type") and block.type == "text":
                    text_content += block.text
                elif hasattr(block, "text"):
                    text_content += block.text
            messages.append({"role": role, "content": text_content})
        else:
            messages.append({"role": role, "content": content})

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]

    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        # Moonshot exposes extra parameters through the OpenAI-compatible layer.
        "extra_body": {"thinking": {"type": "enabled"}},
    }

    usage_obj = None
    response_text = ""
    reasoning_text = ""

    stream = _create_moonshot_stream_with_retry(client, api_params)

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        if hasattr(delta, "reasoning_content"):
            rc = getattr(delta, "reasoning_content")
            if rc:
                reasoning_text += rc

        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
            response_text += content

    content_blocks = [{"type": "text", "text": response_text}]
    usage = normalize_usage(provider="moonshot", model=model, usage_obj=usage_obj)
    return content_blocks, reasoning_text, response_text, usage


MOONSHOT_OVERLOAD_BACKOFF_SECONDS = (2, 4, 8, 16)


def _create_moonshot_stream_with_retry(client: openai.OpenAI, api_params: dict):
    """
    Create a Moonshot streaming chat completion, retrying provider overloads with
    a longer backoff than the OpenAI SDK default.
    """
    last_error = None

    for attempt in range(len(MOONSHOT_OVERLOAD_BACKOFF_SECONDS) + 1):
        try:
            try:
                return client.chat.completions.create(
                    **api_params,
                    stream_options={"include_usage": True},
                )
            except TypeError:
                return client.chat.completions.create(**api_params)
        except openai.RateLimitError as exc:
            last_error = exc
            if not _is_moonshot_engine_overloaded(exc):
                raise
            if attempt >= len(MOONSHOT_OVERLOAD_BACKOFF_SECONDS):
                raise

            delay = MOONSHOT_OVERLOAD_BACKOFF_SECONDS[attempt]
            print(
                f"\n[Moonshot overload; retrying in {delay}s (attempt {attempt + 1}/"
                f"{len(MOONSHOT_OVERLOAD_BACKOFF_SECONDS)})]\n",
                flush=True,
            )
            time.sleep(delay)

    raise last_error


def _is_moonshot_engine_overloaded(error: Exception) -> bool:
    """Detect Moonshot's temporary overload 429 so we only retry that case."""
    if not isinstance(error, openai.RateLimitError):
        return False

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        payload = body.get("error", body)
        if isinstance(payload, dict):
            error_type = payload.get("type")
            message = payload.get("message", "")
            if error_type == "engine_overloaded_error":
                return True
            if isinstance(message, str) and "engine is currently overloaded" in message.lower():
                return True

    message = str(error).lower()
    return (
        "engine_overloaded_error" in message
        or "engine is currently overloaded" in message
    )


def _generate_deepseek_response(
    client: openai.OpenAI,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
) -> tuple[list, str, str, dict]:
    """Generate response using DeepSeek's OpenAI-compatible API."""
    messages = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, list):
            text_content = ""
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_content += block.get("text") or ""
                    elif "text" in block:
                        text_content += block.get("text") or ""
                elif hasattr(block, "type") and block.type == "text":
                    text_content += block.text
                elif hasattr(block, "text"):
                    text_content += block.text
            messages.append({"role": role, "content": text_content})
        else:
            messages.append({"role": role, "content": content})

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]

    api_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }

    # DeepSeek reasoner doesn't accept temperature; chat models do
    if model not in DEEPSEEK_THINKING_MODELS:
        api_params["temperature"] = temperature

    usage_obj = None
    response_text = ""
    reasoning_text = ""

    try:
        stream = client.chat.completions.create(**api_params, stream_options={"include_usage": True})
    except TypeError:
        stream = client.chat.completions.create(**api_params)

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        if hasattr(delta, "reasoning_content"):
            rc = getattr(delta, "reasoning_content")
            if rc:
                reasoning_text += rc

        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
            response_text += content

    content_blocks = [{"type": "text", "text": response_text}]
    usage = normalize_usage(provider="deepseek", model=model, usage_obj=usage_obj)
    return content_blocks, reasoning_text, response_text, usage


def _build_openai_compatible_messages(conversation: list, system_prompt: str) -> list[dict]:
    """Convert internal conversation history into OpenAI-compatible chat messages."""
    messages = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, list):
            text_content = ""
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_content += block.get("text") or ""
                    elif "text" in block:
                        text_content += block.get("text") or ""
                elif hasattr(block, "type") and block.type == "text":
                    text_content += block.text
                elif hasattr(block, "text"):
                    text_content += block.text
            messages.append({"role": role, "content": text_content})
        else:
            messages.append({"role": role, "content": content})

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]

    return messages


def _build_openrouter_messages(conversation: list, system_prompt: str) -> list[dict]:
    """Convert internal conversation history into OpenRouter-compatible chat messages."""
    messages = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]

        message: dict[str, object] = {"role": role}

        if isinstance(content, list):
            text_content = ""
            reasoning_details = []

            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        text_content += block.get("text") or ""
                    elif block_type == "reasoning_details":
                        details = block.get("reasoning_details")
                        if isinstance(details, list):
                            reasoning_details.extend(details)
                    elif "text" in block:
                        text_content += block.get("text") or ""
                elif hasattr(block, "type") and block.type == "text":
                    text_content += block.text
                elif hasattr(block, "text"):
                    text_content += block.text

            message["content"] = text_content
            if reasoning_details:
                message["reasoning_details"] = reasoning_details
        else:
            message["content"] = content

        messages.append(message)

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]

    return messages


def _extract_reasoning_text_from_openrouter_details(reasoning_details: list) -> str:
    """Best-effort text extraction from OpenRouter reasoning_details structures."""
    fragments: list[str] = []

    def _pull(value):
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                fragments.append(text)
            return
        if isinstance(value, dict):
            for key in ("text", "content", "summary", "reasoning"):
                if key in value:
                    _pull(value[key])
            return
        if isinstance(value, list):
            for item in value:
                _pull(item)
            return

    _pull(reasoning_details)
    # Keep ordering stable while dropping duplicates from repeated stream chunks.
    deduped: list[str] = []
    seen = set()
    for fragment in fragments:
        if fragment not in seen:
            deduped.append(fragment)
            seen.add(fragment)
    return "\n\n".join(deduped)


def _generate_qwen_response(
    client: openai.OpenAI,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
    thinking_budget: int,
) -> tuple[list, str, str, dict]:
    """Generate response using DashScope's OpenAI-compatible API."""
    messages = _build_openai_compatible_messages(conversation, system_prompt)

    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "extra_body": {
            "enable_thinking": True,
            "thinking_budget": thinking_budget,
        },
    }

    usage_obj = None
    response_text = ""
    reasoning_text = ""

    try:
        stream = client.chat.completions.create(**api_params, stream_options={"include_usage": True})
    except TypeError:
        stream = client.chat.completions.create(**api_params)

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        if hasattr(delta, "reasoning_content"):
            rc = getattr(delta, "reasoning_content")
            if rc:
                reasoning_text += rc

        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
            response_text += content

    content_blocks = [{"type": "text", "text": response_text}]
    usage = normalize_usage(provider="qwen", model=model, usage_obj=usage_obj)
    return content_blocks, reasoning_text, response_text, usage


def _generate_openrouter_response(
    client: openai.OpenAI,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
    thinking_budget: int,
) -> tuple[list, str, str, dict]:
    """Generate response using OpenRouter's OpenAI-compatible API."""
    provider_model = model.removeprefix("openrouter/")
    messages = _build_openrouter_messages(conversation, system_prompt)

    reasoning_max_tokens = max(1, min(thinking_budget, max_tokens - 1))
    api_params = {
        "model": provider_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "extra_body": {
            "reasoning": {
                "max_tokens": reasoning_max_tokens,
            }
        },
    }

    usage_obj = None
    response_text = ""
    reasoning_text = ""
    reasoning_details_all = []

    try:
        stream = client.chat.completions.create(**api_params, stream_options={"include_usage": True})
    except TypeError:
        stream = client.chat.completions.create(**api_params)

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        if hasattr(delta, "reasoning"):
            rc = getattr(delta, "reasoning")
            if rc:
                reasoning_text += rc

        if hasattr(delta, "reasoning_details"):
            details = getattr(delta, "reasoning_details")
            if details:
                reasoning_details_all.extend(details)

        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
            response_text += content

    if not reasoning_text and reasoning_details_all:
        reasoning_text = _extract_reasoning_text_from_openrouter_details(reasoning_details_all)

    content_blocks = [{"type": "text", "text": response_text}]
    if reasoning_details_all:
        content_blocks.append({"type": "reasoning_details", "reasoning_details": reasoning_details_all})

    usage = normalize_usage(provider="openrouter", model=model, usage_obj=usage_obj)
    return content_blocks, reasoning_text, response_text, usage


def _generate_glm_response(
    client: openai.OpenAI,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
) -> tuple[list, str, str, dict]:
    """Generate response using DashScope's OpenAI-compatible GLM API."""
    messages = _build_openai_compatible_messages(conversation, system_prompt)

    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "extra_body": {
            "enable_thinking": True,
        },
    }

    usage_obj = None
    response_text = ""
    reasoning_text = ""

    try:
        stream = client.chat.completions.create(**api_params, stream_options={"include_usage": True})
    except TypeError:
        stream = client.chat.completions.create(**api_params)

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        if hasattr(delta, "reasoning_content"):
            rc = getattr(delta, "reasoning_content")
            if rc:
                reasoning_text += rc

        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
            response_text += content

    content_blocks = [{"type": "text", "text": response_text}]
    usage = normalize_usage(provider="glm", model=model, usage_obj=usage_obj)
    return content_blocks, reasoning_text, response_text, usage


def _generate_zai_response(
    client: openai.OpenAI,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
) -> tuple[list, str, str, dict]:
    """Generate response using the direct Z.AI OpenAI-compatible API."""
    provider_model = model.removeprefix("zai/")
    messages = _build_openai_compatible_messages(conversation, system_prompt)

    api_params = {
        "model": provider_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "extra_body": {
            "enable_thinking": True,
        },
    }

    usage_obj = None
    response_text = ""
    reasoning_text = ""

    try:
        stream = client.chat.completions.create(**api_params, stream_options={"include_usage": True})
    except TypeError:
        stream = client.chat.completions.create(**api_params)

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage_obj = chunk.usage

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        if hasattr(delta, "reasoning_content"):
            rc = getattr(delta, "reasoning_content")
            if rc:
                reasoning_text += rc

        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
            response_text += content

    content_blocks = [{"type": "text", "text": response_text}]
    usage = normalize_usage(provider="zai", model=model, usage_obj=usage_obj)
    return content_blocks, reasoning_text, response_text, usage


def _generate_gemini_response(
    client,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
    thinking_budget: int,
) -> tuple[list, str, str, dict]:
    """Generate response using Google Gemini API."""
    # Build the contents list for the new API
    contents = []

    # Convert conversation format to new Gemini API format
    for msg in conversation:
        role = "user" if msg["role"] == "user" else "model"
        content = msg["content"]

        # Extract text from content blocks
        if isinstance(content, list):
            text_content = ""
            for block in content:
                if isinstance(block, dict):
                    text_content += block.get("text", "")
                elif hasattr(block, "text"):
                    text_content += block.text
            contents.append({"role": role, "parts": [{"text": text_content}]})
        else:
            contents.append({"role": role, "parts": [{"text": content}]})

    # Generate response with streaming using new API
    config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    if supports_thinking(model):
        if uses_gemini_thinking_level(model):
            config["thinking_config"] = {
                "include_thoughts": True,
                "thinking_level": "HIGH",
            }
        else:
            config["thinking_config"] = {
                "include_thoughts": True,
                "thinking_budget": thinking_budget,
            }

    # Add system instruction if provided
    if system_prompt:
        config["system_instruction"] = system_prompt

    response_text = ""
    reasoning_text = ""
    usage_obj = None

    # Use generate_content_stream() for streaming
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config
    ):
        chunk_reasoning = _extract_gemini_reasoning_text(chunk)
        if chunk_reasoning:
            reasoning_text += chunk_reasoning

        if hasattr(chunk, 'text') and chunk.text:
            print(chunk.text, end="", flush=True)
            response_text += chunk.text

        # Try to capture usage metadata from the last chunk
        if hasattr(chunk, 'usage_metadata'):
            usage_obj = chunk.usage_metadata

    # Return Anthropic-compatible content blocks
    content_blocks = [{"type": "text", "text": response_text}]

    usage = normalize_usage(provider="gemini", model=model, usage_obj=usage_obj)
    return content_blocks, reasoning_text, response_text, usage


def _extract_gemini_reasoning_text(chunk) -> str:
    """Extract thought text from a Gemini streaming chunk, if present."""
    candidates = getattr(chunk, "candidates", None) or []
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []

    thought_parts = []
    for part in parts:
        if getattr(part, "thought", False) and getattr(part, "text", None):
            thought_parts.append(part.text)

    return "".join(thought_parts)


def _generate_grok_response(
    client: XAIClient,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int
) -> tuple[list, str, str, dict]:
    """Generate response using xAI Grok API."""
    try:
        # Create a new chat instance in stateless mode
        # Configuration (temperature, max_tokens) goes in chat.create()
        chat = client.chat.create(
            model=model,
            store_messages=False,
            use_encrypted_content=False,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Add system prompt if provided
        if system_prompt:
            chat.append(xai_system(system_prompt))

        # Convert conversation history to xAI format
        # The xAI SDK expects alternating user/assistant messages
        for msg in conversation:
            content = msg["content"]

            # Extract text from content blocks
            if isinstance(content, list):
                text_content = ""
                for block in content:
                    if isinstance(block, dict):
                        text_content += block.get("text", "")
                    elif hasattr(block, "text"):
                        text_content += block.text
            else:
                text_content = content

            # Add message to chat
            if msg["role"] == "user":
                chat.append(xai_user(text_content))
            elif msg["role"] == "assistant":
                chat.append(xai_assistant(text_content))

        # Generate response (sample() takes no arguments)
        response = chat.sample()

        # Extract response text from the response object
        # The response object structure varies; try different attributes
        if hasattr(response, 'message'):
            if hasattr(response.message, 'content'):
                response_text = str(response.message.content)
            else:
                response_text = str(response.message)
        elif hasattr(response, 'content'):
            response_text = str(response.content)
        elif hasattr(response, 'text'):
            response_text = str(response.text)
        else:
            response_text = str(response)

        # Print response for real-time display
        print(response_text, flush=True)

        # Extract reasoning/thinking if available (for reasoning models)
        reasoning_text = ""
        if hasattr(response, 'reasoning'):
            if hasattr(response.reasoning, 'content'):
                reasoning_text = str(response.reasoning.content)
            elif response.reasoning:
                reasoning_text = str(response.reasoning)

        # Extract usage information
        usage_obj = None
        if hasattr(response, 'usage'):
            usage_obj = response.usage
        elif hasattr(response, 'token_usage'):
            usage_obj = response.token_usage

        # Return Anthropic-compatible content blocks
        content_blocks = [{"type": "text", "text": response_text}]

        usage = normalize_usage(provider="grok", model=model, usage_obj=usage_obj)
        return content_blocks, reasoning_text, response_text, usage

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Grok API call failed: {e}")
        print(traceback.format_exc())
        # Return error as response
        error_text = f"*Error calling Grok API: {str(e)}*"
        content_blocks = [{"type": "text", "text": error_text}]
        usage = normalize_usage(provider="grok", model=model, usage_obj=None)
        return content_blocks, "", error_text, usage

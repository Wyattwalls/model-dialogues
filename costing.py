"""
Token usage + cost estimation utilities.

This repo supports Anthropic, OpenAI, Google Gemini, and xAI Grok models.
Providers expose token usage in different shapes; we normalize usage into:

  {
    "provider": "anthropic" | "openai" | "gemini" | "grok",
    "model": "<model-id>",
    "input_tokens": int | None,
    "output_tokens": int | None,
    "total_tokens": int | None,
    "details": dict (optional extra breakdown)
  }

Cost estimation is optional and driven by a local JSON pricing file.
Rates are expected in USD per 1M tokens.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float


def load_pricing_file(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_pricing_path(cli_path: Optional[str] = None) -> Optional[str]:
    # CLI overrides env; env overrides default.
    if cli_path:
        return cli_path
    env_path = os.environ.get("PRICING_FILE")
    if env_path:
        return env_path
    # Default is a conventional local file; safe if missing (we'll show N/A).
    return "pricing.json"


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def normalize_usage(
    *,
    provider: str,
    model: str,
    usage_obj: Any,
) -> dict[str, Any]:
    """
    Normalize provider-specific usage objects into a common dict.
    """
    u: dict[str, Any] = {"provider": provider, "model": model}
    details: dict[str, Any] = {}

    if usage_obj is None:
        u.update({"input_tokens": None, "output_tokens": None, "total_tokens": None})
        return u

    # Anthropic: Message.usage has input_tokens/output_tokens (and maybe cache fields).
    # OpenAI: usage has prompt_tokens/completion_tokens/total_tokens (plus breakdown).
    # Gemini: usage_metadata has prompt_token_count/candidates_token_count/total_token_count
    if provider == "anthropic":
        input_tokens = _as_int(getattr(usage_obj, "input_tokens", None))
        output_tokens = _as_int(getattr(usage_obj, "output_tokens", None))
        # Some Anthropic SDK versions also provide cache read/write tokens.
        for k in [
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
        ]:
            v = getattr(usage_obj, k, None)
            if v is not None:
                details[k] = v
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        u.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "details": details or None,
            }
        )
        return u

    if provider == "openai":
        input_tokens = _as_int(getattr(usage_obj, "prompt_tokens", None))
        output_tokens = _as_int(getattr(usage_obj, "completion_tokens", None))
        total_tokens = _as_int(getattr(usage_obj, "total_tokens", None))

        # Newer OpenAI "responses" usage uses input_tokens/output_tokens naming,
        # but this code uses chat.completions; still, try to pick up both.
        if input_tokens is None:
            input_tokens = _as_int(getattr(usage_obj, "input_tokens", None))
        if output_tokens is None:
            output_tokens = _as_int(getattr(usage_obj, "output_tokens", None))
        if total_tokens is None:
            total_tokens = _as_int(getattr(usage_obj, "total_tokens", None))

        # Details might include reasoning tokens; keep if present.
        output_details = getattr(usage_obj, "completion_tokens_details", None)
        if output_details is not None and hasattr(output_details, "__dict__"):
            details["completion_tokens_details"] = dict(output_details.__dict__)
        u.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "details": details or None,
            }
        )
        return u

    if provider == "gemini":
        input_tokens = _as_int(getattr(usage_obj, "prompt_token_count", None))
        output_tokens = _as_int(getattr(usage_obj, "candidates_token_count", None))
        total_tokens = _as_int(getattr(usage_obj, "total_token_count", None))
        thoughts_tokens = _as_int(getattr(usage_obj, "thoughts_token_count", None))

        if thoughts_tokens is not None:
            details["thoughts_token_count"] = thoughts_tokens

        u.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "details": details or None,
            }
        )
        return u

    if provider == "grok":
        # xAI Grok models use similar structure to OpenAI
        # Try both naming conventions (xAI may use input_tokens/output_tokens)
        input_tokens = _as_int(getattr(usage_obj, "input_tokens", None))
        output_tokens = _as_int(getattr(usage_obj, "output_tokens", None))
        total_tokens = _as_int(getattr(usage_obj, "total_tokens", None))

        # Fallback to OpenAI-style naming if xAI uses that
        if input_tokens is None:
            input_tokens = _as_int(getattr(usage_obj, "prompt_tokens", None))
        if output_tokens is None:
            output_tokens = _as_int(getattr(usage_obj, "completion_tokens", None))

        # Capture reasoning tokens if available
        reasoning_tokens = _as_int(getattr(usage_obj, "reasoning_tokens", None))
        if reasoning_tokens is not None:
            details["reasoning_tokens"] = reasoning_tokens

        u.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "details": details or None,
            }
        )
        return u

    # Unknown provider; best-effort.
    u.update({"input_tokens": None, "output_tokens": None, "total_tokens": None})
    return u


def _find_model_pricing(model: str, pricing_doc: dict[str, Any]) -> Optional[ModelPricing]:
    """
    Pricing lookup rules:
    - exact match under pricing_doc["models"][model]
    - alias match under pricing_doc["aliases"][model] -> real_model
    """
    if not pricing_doc:
        return None

    models = pricing_doc.get("models") or {}
    aliases = pricing_doc.get("aliases") or {}

    key = model
    if key in aliases:
        key = aliases[key]

    entry = models.get(key)
    if not isinstance(entry, dict):
        return None

    inp = entry.get("input")
    out = entry.get("output")
    if inp is None or out is None:
        return None

    try:
        return ModelPricing(float(inp), float(out))
    except (TypeError, ValueError):
        return None


def estimate_cost_usd(usage: dict[str, Any], pricing_doc: dict[str, Any]) -> Optional[float]:
    """
    Returns estimated USD cost for one request, or None if not computable.
    """
    model = usage.get("model")
    if not model:
        return None
    mp = _find_model_pricing(model, pricing_doc)
    if mp is None:
        return None

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None

    # Rates are USD per 1M tokens.
    return (input_tokens * mp.input_per_1m + output_tokens * mp.output_per_1m) / 1_000_000.0


def format_usd(amount: Optional[float]) -> str:
    if amount is None:
        return "N/A"
    # Keep 6 decimals so small runs show up; users can visually round.
    return f"${amount:.6f}"

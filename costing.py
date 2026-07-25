"""
Token usage + cost estimation utilities.

This repo supports Anthropic, OpenAI, Google Gemini, Moonshot Kimi,
OpenRouter Kimi/GLM, direct Z.AI GLM, DashScope Qwen, DashScope GLM,
DeepSeek, and xAI Grok models.
Providers expose token usage in different shapes; we normalize usage into:

  {
    "provider": "anthropic" | "openai" | "gemini" | "moonshot" | "openrouter" | "zai" | "qwen" | "glm" | "deepseek" | "grok",
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
    cache_read_per_1m: float | None = None
    cache_write_per_1m: float | None = None


@dataclass(frozen=True)
class PricingTier:
    max_input_tokens: int | None
    input_per_1m: float
    output_per_1m: float
    cache_read_per_1m: float | None = None
    cache_write_per_1m: float | None = None


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
    # Moonshot / OpenRouter / Z.AI / DashScope / GLM / DeepSeek: OpenAI-compatible usage plus reasoning metadata
    if provider == "anthropic":
        input_tokens = _as_int(getattr(usage_obj, "input_tokens", None))
        output_tokens = _as_int(getattr(usage_obj, "output_tokens", None))
        # Anthropic reports cache reads and writes separately from input_tokens
        # (input_tokens is the uncached portion only).
        cache_read = _as_int(getattr(usage_obj, "cache_read_input_tokens", None))
        cache_creation = _as_int(getattr(usage_obj, "cache_creation_input_tokens", None))
        if cache_read is not None:
            details["cache_read_input_tokens"] = cache_read
        if cache_creation is not None:
            details["cache_creation_input_tokens"] = cache_creation
        # output_tokens_details.thinking_tokens reports the full thinking tokens
        # billed for adaptive/extended thinking, even when display="omitted"
        # makes the thinking content invisible in the response. Capturing it
        # lets us report what we paid for thinking vs visible response text.
        # The SDK may expose this as either an attribute or a dict.
        thinking_tokens = None
        out_details = getattr(usage_obj, "output_tokens_details", None)
        if out_details is not None:
            if isinstance(out_details, dict):
                thinking_tokens = _as_int(out_details.get("thinking_tokens"))
            else:
                thinking_tokens = _as_int(getattr(out_details, "thinking_tokens", None))
        if thinking_tokens is not None:
            details["thinking_tokens"] = thinking_tokens
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
            if cache_read is not None:
                total_tokens += cache_read
            if cache_creation is not None:
                total_tokens += cache_creation
        u.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cache_read_tokens": cache_read,
                "cache_creation_tokens": cache_creation,
                "thinking_tokens": thinking_tokens,
                "details": details or None,
            }
        )
        return u

    if provider in {"openai", "moonshot", "openrouter", "zai", "qwen", "glm", "deepseek"}:
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

        # Details might include reasoning tokens; keep if present. Responses API
        # usage uses output_tokens_details, while Chat Completions uses
        # completion_tokens_details.
        output_details = getattr(usage_obj, "completion_tokens_details", None)
        if output_details is None:
            output_details = getattr(usage_obj, "output_tokens_details", None)
        if output_details is not None and hasattr(output_details, "__dict__"):
            details["completion_tokens_details"] = dict(output_details.__dict__)
        reasoning_tokens = _as_int(getattr(usage_obj, "reasoning_tokens", None))
        if reasoning_tokens is None and output_details is not None:
            reasoning_tokens = _as_int(getattr(output_details, "reasoning_tokens", None))
        if reasoning_tokens is not None:
            details["reasoning_tokens"] = reasoning_tokens

        # Cache hits are a subset of prompt_tokens for OpenAI-compatible APIs.
        # Responses API uses input_tokens_details; Chat Completions uses
        # prompt_tokens_details. Other compatible providers may use top-level
        # prompt_cache_hit_tokens.
        cache_read = None
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        if prompt_details is None:
            prompt_details = getattr(usage_obj, "input_tokens_details", None)
        if prompt_details is not None:
            cache_read = _as_int(getattr(prompt_details, "cached_tokens", None))
            if hasattr(prompt_details, "__dict__"):
                details["prompt_tokens_details"] = dict(prompt_details.__dict__)
        if cache_read is None:
            # DeepSeek (and some OpenAI-compatible providers) report it top-level.
            cache_read = _as_int(getattr(usage_obj, "prompt_cache_hit_tokens", None))
        if cache_read is not None:
            details["cached_tokens"] = cache_read

        u.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cache_read_tokens": cache_read,
                "details": details or None,
            }
        )
        return u

    if provider == "gemini":
        input_tokens = _as_int(getattr(usage_obj, "prompt_token_count", None))
        output_tokens = _as_int(getattr(usage_obj, "candidates_token_count", None))
        total_tokens = _as_int(getattr(usage_obj, "total_token_count", None))
        thoughts_tokens = _as_int(getattr(usage_obj, "thoughts_token_count", None))
        cache_read = _as_int(getattr(usage_obj, "cached_content_token_count", None))

        if thoughts_tokens is not None:
            details["thoughts_token_count"] = thoughts_tokens
            # Google bills thinking tokens at the output rate but exposes them
            # as a separate count; fold them into output for correct billing.
            if output_tokens is not None:
                details["visible_output_tokens"] = output_tokens
                output_tokens = output_tokens + thoughts_tokens

        if cache_read is not None:
            details["cached_content_token_count"] = cache_read

        u.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cache_read_tokens": cache_read,
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

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
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

    # Unknown provider; best-effort.
    u.update({"input_tokens": None, "output_tokens": None, "total_tokens": None})
    return u


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_pricing_entry(
    entry: dict[str, Any],
    input_tokens: int | None = None,
) -> Optional[ModelPricing]:
    """Support either flat pricing or tiered pricing keyed by input token count."""
    tiers = entry.get("tiers")
    if isinstance(tiers, list):
        parsed_tiers: list[PricingTier] = []
        for tier in tiers:
            if not isinstance(tier, dict):
                continue
            inp = tier.get("input")
            out = tier.get("output")
            if inp is None or out is None:
                continue
            try:
                parsed_tiers.append(
                    PricingTier(
                        max_input_tokens=_as_int(tier.get("max_input_tokens")),
                        input_per_1m=float(inp),
                        output_per_1m=float(out),
                        cache_read_per_1m=_optional_float(tier.get("cache_read")),
                        cache_write_per_1m=_optional_float(tier.get("cache_write")),
                    )
                )
            except (TypeError, ValueError):
                continue

        if not parsed_tiers:
            return None

        if input_tokens is None:
            selected = parsed_tiers[0]
        else:
            selected = parsed_tiers[-1]
            for tier in parsed_tiers:
                if tier.max_input_tokens is None or input_tokens <= tier.max_input_tokens:
                    selected = tier
                    break

        return ModelPricing(
            selected.input_per_1m,
            selected.output_per_1m,
            cache_read_per_1m=selected.cache_read_per_1m,
            cache_write_per_1m=selected.cache_write_per_1m,
        )

    inp = entry.get("input")
    out = entry.get("output")
    if inp is None or out is None:
        return None

    try:
        return ModelPricing(
            float(inp),
            float(out),
            cache_read_per_1m=_optional_float(entry.get("cache_read")),
            cache_write_per_1m=_optional_float(entry.get("cache_write")),
        )
    except (TypeError, ValueError):
        return None


def _find_model_pricing(
    model: str,
    pricing_doc: dict[str, Any],
    input_tokens: int | None = None,
) -> Optional[ModelPricing]:
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

    return _parse_pricing_entry(entry, input_tokens=input_tokens)


def estimate_cost_usd(usage: dict[str, Any], pricing_doc: dict[str, Any]) -> Optional[float]:
    """
    Returns estimated USD cost for one request, or None if not computable.

    Applies cache discounts when pricing_doc supplies cache_read / cache_write
    rates and the usage carries cache_read_tokens / cache_creation_tokens.

    Provider semantics differ:
    - Anthropic: input_tokens is the uncached portion only; cache reads and
      writes are billed in addition.
    - OpenAI / Gemini / DeepSeek: cache_read_tokens are a subset of
      input_tokens. The uncached portion = input_tokens - cache_read_tokens.
    """
    model = usage.get("model")
    if not model:
        return None

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None

    mp = _find_model_pricing(model, pricing_doc, input_tokens=input_tokens)
    if mp is None:
        return None

    provider = usage.get("provider")
    cache_read = usage.get("cache_read_tokens") or 0
    cache_creation = usage.get("cache_creation_tokens") or 0

    # When a specific cache rate is missing, fall back to the input rate
    # (slightly over-bills cached reads, slightly under-bills Anthropic writes;
    # accurate again once the user backfills rates in pricing.json).
    cache_read_rate = mp.cache_read_per_1m if mp.cache_read_per_1m is not None else mp.input_per_1m
    cache_write_rate = mp.cache_write_per_1m if mp.cache_write_per_1m is not None else mp.input_per_1m

    if provider == "anthropic":
        cost = (
            input_tokens * mp.input_per_1m
            + cache_read * cache_read_rate
            + cache_creation * cache_write_rate
            + output_tokens * mp.output_per_1m
        )
    else:
        uncached_input = max(0, input_tokens - cache_read)
        cost = (
            uncached_input * mp.input_per_1m
            + cache_read * cache_read_rate
            + output_tokens * mp.output_per_1m
        )

    return cost / 1_000_000.0


def estimate_uncached_cost_usd(usage: dict[str, Any], pricing_doc: dict[str, Any]) -> Optional[float]:
    """
    What this request would have cost with no caching at all.

    Used to compute savings vs the no-cache baseline. Same semantics caveat as
    estimate_cost_usd: Anthropic's input_tokens excludes cache reads/writes,
    while OpenAI/Gemini include them.
    """
    model = usage.get("model")
    if not model:
        return None

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None

    mp = _find_model_pricing(model, pricing_doc, input_tokens=input_tokens)
    if mp is None:
        return None

    provider = usage.get("provider")
    cache_read = usage.get("cache_read_tokens") or 0
    cache_creation = usage.get("cache_creation_tokens") or 0

    if provider == "anthropic":
        total_input = input_tokens + cache_read + cache_creation
    else:
        total_input = input_tokens  # cache_read already a subset

    return (total_input * mp.input_per_1m + output_tokens * mp.output_per_1m) / 1_000_000.0


def format_usd(amount: Optional[float]) -> str:
    if amount is None:
        return "N/A"
    # Keep 6 decimals so small runs show up; users can visually round.
    return f"${amount:.6f}"

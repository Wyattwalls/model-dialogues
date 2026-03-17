"""
LLM-to-LLM Conversation Runner

Runs an automated conversation between two LLM instances,
with extended thinking enabled (if supported by the model).
"""

import os
import json
import argparse
import subprocess
from datetime import datetime
from dotenv import load_dotenv

import params
from api_client import (
    create_anthropic_client,
    create_openai_client,
    create_moonshot_client,
    create_gemini_client,
    create_xai_client,
    create_deepseek_client,
    generate_response,
    normalize_model_name,
    supports_thinking,
    uses_adaptive_thinking,
    uses_gemini_thinking_level,
    uses_openai_reasoning,
    thinking_budget_to_effort,
    is_openai_model,
    is_moonshot_model,
    is_gemini_model,
    is_grok_model,
    is_deepseek_model
)
from conversation import build_convo_a, build_convo_b
from costing import estimate_cost_usd, format_usd, get_pricing_path, load_pricing_file

# Load environment variables from .env file
load_dotenv()


def create_turn_tracking_message(next_model_name: str, total_turn_num: int, max_turns: int) -> dict:
    """Create a turn tracking message marker for the next model."""
    return {
        "type": "turn_tracking",
        "content": f"Turn {total_turn_num}/{max_turns} completed.",
        "next_model": next_model_name,
        "next_turn": total_turn_num,
    }


def get_assistant_name(model: str) -> str:
    """Get the assistant name based on model."""
    if is_deepseek_model(model):
        return "DeepSeek"
    elif is_openai_model(model):
        return "ChatGPT"
    elif is_moonshot_model(model):
        return "Kimi"
    elif is_grok_model(model):
        return "Grok"
    elif is_gemini_model(model):
        return "Gemini"
    else:
        return "Claude"


def get_developer_name(model: str) -> str:
    """Get the developer name based on model."""
    if is_deepseek_model(model):
        return "DeepSeek"
    elif is_openai_model(model):
        return "OpenAI"
    elif is_moonshot_model(model):
        return "Moonshot AI"
    elif is_grok_model(model):
        return "xAI"
    elif is_gemini_model(model):
        return "Google"
    else:
        return "Anthropic"


def describe_thinking_config(model: str, thinking_budget: int) -> str:
    """Render the effective thinking configuration for transcripts."""
    if not supports_thinking(model):
        return "N/A"
    if is_moonshot_model(model):
        return "enabled (reasoning_content)"
    if is_deepseek_model(model):
        return "enabled (reasoning_content)" if model in ("deepseek-reasoner",) else "N/A"
    if uses_openai_reasoning(model):
        return "responses API (medium effort default)"
    if uses_gemini_thinking_level(model):
        return "thinkingLevel=high (includeThoughts)"
    if uses_adaptive_thinking(model):
        effort = thinking_budget_to_effort(thinking_budget)
        return f"adaptive ({effort} effort; mapped from budget {thinking_budget})"
    return str(thinking_budget)


def describe_temperature_config(model: str, temperature: float) -> str:
    """Render the effective temperature configuration for transcripts."""
    if uses_openai_reasoning(model):
        return "ignored (Responses API default)"
    return str(temperature)


def run_conversation(
    system_prompt_a: str,
    system_prompt_b: str,
    start_a: str = "",
    start_b: str = "You are about to speak with another LLM. Please begin the conversation.",
    max_turns: int = 40,
    temperature_a: float = 1.0,
    temperature_b: float = 1.0,
    output_dir: str = "transcripts",
    model_a: str = "claude-sonnet-4-5-20250929",
    model_b: str = "claude-sonnet-4-5-20250929",
    thinking_budget_a: int = 12000,
    thinking_budget_b: int = 12000,
    final_question_a: str = None,
    final_question_b: str = None,
    pricing_file: str | None = None
):
    """Run a conversation between two models (Anthropic, OpenAI, and/or Gemini) with per-model settings."""

    model_a = normalize_model_name(model_a)
    model_b = normalize_model_name(model_b)

    if max_turns % 2 != 0:
        raise ValueError(f"max_turns must be even so turns split evenly between models; got {max_turns}")

    run_started_at = datetime.now()
    run_timestamp = run_started_at.strftime("%Y-%m-%d-%H-%M-%S-%f")

    # Create API clients only for the providers being used
    needs_anthropic = (
        not is_openai_model(model_a) and not is_moonshot_model(model_a) and not is_gemini_model(model_a) and not is_grok_model(model_a) and not is_deepseek_model(model_a)
    ) or (
        not is_openai_model(model_b) and not is_moonshot_model(model_b) and not is_gemini_model(model_b) and not is_grok_model(model_b) and not is_deepseek_model(model_b)
    )
    needs_openai = is_openai_model(model_a) or is_openai_model(model_b)
    needs_moonshot = is_moonshot_model(model_a) or is_moonshot_model(model_b)
    needs_gemini = is_gemini_model(model_a) or is_gemini_model(model_b)
    needs_xai = is_grok_model(model_a) or is_grok_model(model_b)
    needs_deepseek = is_deepseek_model(model_a) or is_deepseek_model(model_b)

    anthropic_client = create_anthropic_client() if needs_anthropic else None
    openai_client = create_openai_client() if needs_openai else None
    moonshot_client = create_moonshot_client() if needs_moonshot else None
    gemini_client = create_gemini_client() if needs_gemini else None
    xai_client = create_xai_client() if needs_xai else None
    deepseek_client = create_deepseek_client() if needs_deepseek else None

    # Conversation bootstrap:
    # - Model A first sees the facilitator prompt in start_b.
    # - Model B first sees Model A's actual generated reply.
    initial_prompt_a = start_b
    if start_b:
        initial_prompt_a = (
            f"{start_b}\n\n"
            f"System: Commencing turn 1 of {max_turns // 2}."
        )

    history_a = [start_a, initial_prompt_a]
    history_b = []
    response_b = None
    response_a = None

    # Calculate actual turns per model
    actual_turns_per_model = max_turns // 2

    # Show model info
    def model_description(model):
        if is_deepseek_model(model):
            provider = "DeepSeek"
        elif is_gemini_model(model):
            provider = "Google"
        elif is_moonshot_model(model):
            provider = "Moonshot"
        elif is_openai_model(model):
            provider = "OpenAI"
        elif is_grok_model(model):
            provider = "xAI"
        else:
            provider = "Anthropic"
        thinking = "(with thinking)" if supports_thinking(model) else "(no thinking)"
        return f"{model} [{provider}] {thinking}"

    pricing_path = get_pricing_path(pricing_file)
    pricing_doc = load_pricing_file(pricing_path)
    git_commit_hash = _get_git_commit_hash()

    transcript_note = """About this transcript:
- This file contains: (a) the parameters used for the run (b) a transcript of the convervsation, formatted to improve readability.
- The following has been added to the transcript, but are not seen by the models: this note, the parameter/model sections, section headers, turn separators, [THINKING]/[RESPONSE] labels, the run summary, cost information and any status/error note for incomplete runs.
- The models may see turn markers such as "System: Commencing turn 2 of 20.". These are injected into the previous model's response after that response is generated, so they are only seen by the model about to commence its turn.

"""

    # Parameters section
    params_info = f"""Parameters:
- Max Turns: {max_turns}
- Temperature A: {describe_temperature_config(model_a, temperature_a)}
- Temperature B: {describe_temperature_config(model_b, temperature_b)}
- Thinking A: {describe_thinking_config(model_a, thinking_budget_a)}
- Thinking B: {describe_thinking_config(model_b, thinking_budget_b)}
- Timestamp: {run_started_at.strftime("%Y-%m-%d %H:%M:%S")}
- Git Commit Hash: {git_commit_hash or "unknown"}
- Start Message A: {start_a if start_a else "(empty)"}
- Start Message B: {start_b if start_b else "(empty)"}
- Final Question A: {final_question_a if final_question_a else "(none)"}
- Final Question B: {final_question_b if final_question_b else "(none)"}

"""

    model_info = f"""Models:
- Model A: {model_description(model_a)}
- Model B: {model_description(model_b)}

"""
    pricing_info = "Pricing: disabled (no pricing file found)\n"
    if pricing_doc:
        pricing_info = f"Pricing: enabled ({pricing_path})\n"

    # Get model labels for display
    model_a_label = get_assistant_name(model_a).upper()
    model_b_label = get_assistant_name(model_b).upper()

    transcript = f"""{transcript_note}{params_info}{model_info}{pricing_info}
System Prompt A: {system_prompt_a}

System Prompt B: {system_prompt_b}

================================================================================
█ INITIAL PROMPT - MODEL A: {model_a}
================================================================================
{start_b}
"""
    print(transcript)
    
    convo_a = []
    convo_b = []

    # Track per-request usage/cost.
    run_metrics: list[dict] = []
    run_config = {
        "max_turns": max_turns,
        "temperature_a": temperature_a,
        "temperature_b": temperature_b,
        "thinking_budget_a": thinking_budget_a,
        "thinking_budget_b": thinking_budget_b,
        "start_a": start_a,
        "start_b": start_b,
        "final_question_a": final_question_a,
        "final_question_b": final_question_b,
    }

    def checkpoint(status: str = "in_progress", phase: str = "conversation", error: Exception | None = None):
        current_convo_a = build_convo_a(history_a)
        current_convo_b = build_convo_b(history_b)
        return save_transcript(
            transcript=transcript,
            convo_a=current_convo_a,
            convo_b=current_convo_b,
            output_dir=output_dir,
            model_a=model_a,
            model_b=model_b,
            model_a_label=model_a_label,
            model_b_label=model_b_label,
            system_prompt_a=system_prompt_a,
            system_prompt_b=system_prompt_b,
            run_config=run_config,
            git_commit_hash=git_commit_hash,
            run_timestamp=run_timestamp,
            run_metrics=run_metrics,
            pricing_doc=pricing_doc,
            status=status,
            phase=phase,
            error=error,
        )
    
    checkpoint(phase="initialized")

    try:
        for i in range(max_turns):
            current_trans = transcript

            # Model A's turn (even iterations)
            if i % 2 == 0:
                turn_num_a = (i // 2) + 1

                if response_b is not None:
                    # After the first turn, Model A sees Model B's latest reply.
                    history_a.append(
                        response_b
                        + f"\n\nSystem: Commencing turn {turn_num_a} of {max_turns // 2}."
                    )
                convo_a = build_convo_a(history_a)
                checkpoint(phase=f"awaiting_model_a_turn_{turn_num_a}")
                content_blocks_a, reasoning_a, response_a, usage_a = generate_response(
                    anthropic_client, openai_client, moonshot_client, convo_a, system_prompt_a,
                    temperature_a, model=model_a, thinking_budget=thinking_budget_a,
                    gemini_client=gemini_client, xai_client=xai_client, deepseek_client=deepseek_client
                )
                est_cost_a = estimate_cost_usd(usage_a, pricing_doc)
                run_metrics.append(
                    {
                        "who": "A",
                        "model": model_a,
                        "turn": (i // 2) + 1,
                        "usage": usage_a,
                        "cost_usd": est_cost_a,
                    }
                )
                # Store content blocks for API compatibility
                history_a.append(content_blocks_a)
                # Format for display
                turn_num = (i // 2) + 1
                separator = f"\n{'='*80}\n█ TURN {turn_num} - MODEL A: {model_a}\n{'='*80}\n"
                if reasoning_a:
                    display_text = f"[THINKING]\n{reasoning_a}\n\n[RESPONSE]\n{response_a}\n"
                else:
                    display_text = f"[RESPONSE]\n{response_a}\n"
                addline = separator + display_text
                transcript = current_trans + addline
                print(addline)
                checkpoint(phase=f"completed_model_a_turn_{turn_num}")

            # Model B's turn (odd iterations)
            else:
                # Add Model A's latest reply to Model B's history with an explicit
                # note about the upcoming turn.
                turn_num_b = ((i - 1) // 2) + 1
                history_b.append(
                    response_a
                    + f"\n\nSystem: Commencing turn {turn_num_b} of {max_turns // 2}."
                )
                convo_b = build_convo_b(history_b)
                checkpoint(phase=f"awaiting_model_b_turn_{turn_num_b}")
                content_blocks_b, reasoning_b, response_b, usage_b = generate_response(
                    anthropic_client, openai_client, moonshot_client, convo_b, system_prompt_b,
                    temperature_b, model=model_b, thinking_budget=thinking_budget_b,
                    gemini_client=gemini_client, xai_client=xai_client, deepseek_client=deepseek_client
                )
                est_cost_b = estimate_cost_usd(usage_b, pricing_doc)
                run_metrics.append(
                    {
                        "who": "B",
                        "model": model_b,
                        "turn": ((i - 1) // 2) + 1,
                        "usage": usage_b,
                        "cost_usd": est_cost_b,
                    }
                )
                # Store content blocks for API compatibility
                history_b.append(content_blocks_b)
                # Format for display
                turn_num = ((i - 1) // 2) + 1
                separator = f"\n{'='*80}\n█ TURN {turn_num} - MODEL B: {model_b} \n{'='*80}\n"
                if reasoning_b:
                    display_text = f"[THINKING]\n{reasoning_b}\n\n[RESPONSE]\n{response_b}\n"
                else:
                    display_text = f"[RESPONSE]\n{response_b}\n"
                addline = separator + display_text
                transcript = current_trans + addline
                print(addline)
                checkpoint(phase=f"completed_model_b_turn_{turn_num}")

        # Handle final questions if provided
        if final_question_a or final_question_b:
            final_separator = f"\n{'='*60}\n{'='*22} MODEL A DEBRIEF {'='*21}\n{'='*60}\n"
            transcript += final_separator
            print(final_separator)
            checkpoint(phase="final_questions_started")

            if final_question_a:
                question_text_a = f"{final_question_a}\n"
                transcript += question_text_a
                print(question_text_a)

                history_a.append(final_question_a)
                convo_a = build_convo_a(history_a)
                checkpoint(phase="awaiting_final_question_a")
                content_blocks_a, reasoning_a, response_a, usage_a = generate_response(
                    anthropic_client, openai_client, moonshot_client, convo_a, system_prompt_a,
                    temperature_a, model=model_a, thinking_budget=thinking_budget_a,
                    gemini_client=gemini_client, xai_client=xai_client, deepseek_client=deepseek_client
                )
                est_cost_a = estimate_cost_usd(usage_a, pricing_doc)
                run_metrics.append(
                    {
                        "who": "A",
                        "model": model_a,
                        "turn": "final",
                        "usage": usage_a,
                        "cost_usd": est_cost_a,
                    }
                )
                history_a.append(content_blocks_a)

                separator_a = f"\n{'='*80}\n█ FINAL RESPONSE - MODEL A: {model_a}\n{'='*80}\n"
                if reasoning_a:
                    display_text_a = f"[THINKING]\n{reasoning_a}\n\n[RESPONSE]\n{response_a}\n"
                else:
                    display_text_a = f"[RESPONSE]\n{response_a}\n"
                addline_a = separator_a + display_text_a
                transcript += addline_a
                print(addline_a)
                checkpoint(phase="completed_final_question_a")

            if final_question_b:
                final_separator_b = f"\n{'='*60}\n{'='*22} MODEL B DEBRIEF {'='*21}\n{'='*60}\n"
                question_text_b = f"{final_question_b}\n"
                transcript += final_separator_b + question_text_b
                print(final_separator_b, end="")
                print(question_text_b)

                history_b.append(final_question_b)
                convo_b = build_convo_b(history_b)
                checkpoint(phase="awaiting_final_question_b")
                content_blocks_b, reasoning_b, response_b, usage_b = generate_response(
                    anthropic_client, openai_client, moonshot_client, convo_b, system_prompt_b,
                    temperature_b, model=model_b, thinking_budget=thinking_budget_b,
                    gemini_client=gemini_client, xai_client=xai_client, deepseek_client=deepseek_client
                )
                est_cost_b = estimate_cost_usd(usage_b, pricing_doc)
                run_metrics.append(
                    {
                        "who": "B",
                        "model": model_b,
                        "turn": "final",
                        "usage": usage_b,
                        "cost_usd": est_cost_b,
                    }
                )
                history_b.append(content_blocks_b)

                separator_b = f"\n{'='*80}\n█ FINAL RESPONSE - MODEL B: {model_b}\n{'='*80}\n"
                if reasoning_b:
                    display_text_b = f"[THINKING]\n{reasoning_b}\n\n[RESPONSE]\n{response_b}\n"
                else:
                    display_text_b = f"[RESPONSE]\n{response_b}\n"
                addline_b = separator_b + display_text_b
                transcript += addline_b
                print(addline_b)
                checkpoint(phase="completed_final_question_b")

        run_summary = _format_run_summary(run_metrics, pricing_doc)
        print(run_summary)

        convo_a = build_convo_a(history_a)
        convo_b = build_convo_b(history_b)
        save_path, _ = save_transcript(
            transcript=transcript,
            convo_a=convo_a,
            convo_b=convo_b,
            output_dir=output_dir,
            model_a=model_a,
            model_b=model_b,
            model_a_label=model_a_label,
            model_b_label=model_b_label,
            system_prompt_a=system_prompt_a,
            system_prompt_b=system_prompt_b,
            run_config=run_config,
            git_commit_hash=git_commit_hash,
            run_timestamp=run_timestamp,
            run_metrics=run_metrics,
            pricing_doc=pricing_doc,
            status="completed",
            phase="completed",
            error=None,
        )

        return transcript + run_summary
    except Exception as error:
        save_path, _ = checkpoint(status="failed", phase="exception", error=error)
        print(f"\nPartial transcript saved to: {save_path}")
        raise


def _format_usage_lines(usage: dict, est_cost_usd: float | None) -> str:
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    tot = usage.get("total_tokens")
    provider = usage.get("provider") or "unknown"
    model = usage.get("model") or "unknown"
    return (
        f"\n[USAGE] provider={provider} model={model} input_tokens={inp} output_tokens={out} total_tokens={tot}\n"
        f"[ESTIMATED_COST] {format_usd(est_cost_usd)}\n"
    )


def _format_run_summary(run_metrics: list[dict], pricing_doc: dict) -> str:
    # Aggregate by who (A/B) and overall.
    agg: dict[str, dict[str, float | int]] = {}
    total_cost = 0.0
    total_cost_known = False

    for m in run_metrics:
        who = m.get("who", "?")
        usage = m.get("usage") or {}
        inp = usage.get("input_tokens") or 0
        out = usage.get("output_tokens") or 0
        cost = m.get("cost_usd")

        a = agg.setdefault(who, {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cost_known": False})
        a["input_tokens"] = int(a["input_tokens"]) + int(inp)
        a["output_tokens"] = int(a["output_tokens"]) + int(out)

        if cost is not None:
            a["cost_usd"] = float(a["cost_usd"]) + float(cost)
            a["cost_known"] = True
            total_cost += float(cost)
            total_cost_known = True

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("COST SUMMARY (estimated; requires pricing.json)")
    lines.append("=" * 80)

    for who in ["A", "B"]:
        a = agg.get(who)
        if not a:
            continue
        cost_str = format_usd(a["cost_usd"] if a.get("cost_known") else None)
        lines.append(
            f"- Model {who}: input_tokens={a['input_tokens']} output_tokens={a['output_tokens']} estimated_cost={cost_str}"
        )

    total_str = format_usd(total_cost if total_cost_known else None)
    lines.append(f"- TOTAL: estimated_cost={total_str}")
    lines.append("=" * 80 + "\n")
    return "\n".join(lines)


def _shorten_model_name(model: str) -> str:
    """Return a readable, filename-safe model identifier without collapsing variants."""
    model = model.strip().lower()
    model = model.replace("chatgpt-", "gpt-")

    safe = []
    previous_was_dash = False
    for ch in model:
        if ch.isalnum():
            safe.append(ch)
            previous_was_dash = False
        else:
            if not previous_was_dash:
                safe.append("-")
                previous_was_dash = True

    slug = "".join(safe).strip("-")
    return slug or "unknown-model"


def _build_save_paths(output_dir: str, run_timestamp: str, model_a: str, model_b: str, incomplete: bool) -> tuple[str, str]:
    model_a_short = _shorten_model_name(model_a)
    model_b_short = _shorten_model_name(model_b)
    suffix = "-INCOMPLETE" if incomplete else ""
    base_name = f"{run_timestamp}-{model_a_short}-vs-{model_b_short}{suffix}"
    return (
        os.path.join(output_dir, f"{base_name}.txt"),
        os.path.join(output_dir, f"{base_name}.state.json"),
    )


def _serialize_content(value):
    if isinstance(value, list):
        return [_serialize_content(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_content(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        block = {}
        for attr in ("type", "text", "thinking", "signature"):
            if hasattr(value, attr):
                block[attr] = getattr(value, attr)
        return block if block else str(value)
    return value


def _serialize_message(msg: dict) -> dict:
    return {
        "role": msg["role"],
        "content": _serialize_content(msg["content"]),
    }


def _render_transcript_document(
    transcript: str,
    status_note: str | None = None,
) -> str:
    parts = [transcript]
    if status_note:
        parts.append("\n")
        parts.append(status_note)
        parts.append("\n")

    return "".join(parts)


def _write_text_atomic(path: str, content: str):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        file.write(content)
    os.replace(temp_path, path)


def _write_json_atomic(path: str, payload: dict):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temp_path, path)


def _get_git_commit_hash() -> str | None:
    """Return the current git commit hash if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def save_transcript(
    transcript: str,
    convo_a: list,
    convo_b: list,
    output_dir: str,
    model_a: str,
    model_b: str,
    model_a_label: str,
    model_b_label: str,
    system_prompt_a: str,
    system_prompt_b: str,
    run_config: dict,
    git_commit_hash: str | None,
    run_timestamp: str,
    run_metrics: list[dict],
    pricing_doc: dict,
    status: str = "completed",
    phase: str = "completed",
    error: Exception | None = None,
):
    """Save transcript and current run state to disk."""

    os.makedirs(output_dir, exist_ok=True)

    incomplete = status != "completed"
    save_path, state_path = _build_save_paths(output_dir, run_timestamp, model_a, model_b, incomplete=incomplete)

    summary = _format_run_summary(run_metrics, pricing_doc) if run_metrics else ""
    status_note = None
    if incomplete:
        status_lines = [
            "=" * 80,
            f"RUN STATUS: {status.upper()}",
            f"PHASE: {phase}",
            f"SAVED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if error is not None:
            status_lines.append(f"ERROR: {type(error).__name__}: {error}")
        status_lines.append("=" * 80)
        status_note = "\n".join(status_lines)

    document = _render_transcript_document(
        transcript=transcript + summary,
        status_note=status_note,
    )
    _write_text_atomic(save_path, document)

    state_payload = {
        "_note": (
            "This file is a snapshot of the run state at the time it was saved. "
            "Because the conversation is built iteratively, convo_a and convo_b are "
            "also cumulative records of what has been sent and received so far, in "
            "the shaped form each model would currently see."
        ),
        "status": status,
        "phase": phase,
        "saved_at": datetime.now().isoformat(),
        "run_timestamp": run_timestamp,
        "models": {
            "a": {"id": model_a, "label": model_a_label},
            "b": {"id": model_b, "label": model_b_label},
        },
        "git_commit_hash": git_commit_hash,
        "system_prompt_a": system_prompt_a,
        "system_prompt_b": system_prompt_b,
        "run_config": run_config,
        "metrics": run_metrics,
        "convo_a": [_serialize_message(msg) for msg in convo_a],
        "convo_b": [_serialize_message(msg) for msg in convo_b],
        "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
    }
    _write_json_atomic(state_path, state_payload)

    if status == "completed":
        incomplete_txt, incomplete_state = _build_save_paths(output_dir, run_timestamp, model_a, model_b, incomplete=True)
        for stale_path in (incomplete_txt, incomplete_state):
            if os.path.exists(stale_path):
                os.remove(stale_path)
        print(f"\nTranscript saved to: {save_path}")

    return save_path, state_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a conversation between two Claude instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  Anthropic (with thinking support):
    - claude-opus-4-5-20251101
    - claude-haiku-4-5-20251001
    - claude-sonnet-4-5-20250929 (default)
    - claude-opus-4-1-20250805
    - claude-opus-4-20250514
    - claude-sonnet-4-20250514
    - claude-3-7-sonnet-20250219

  Anthropic (without thinking):
    - claude-3-5-haiku-20241022
    - claude-3-haiku-20240307

  xAI Grok (with reasoning):
    - grok-4-1-fast-reasoning
    - grok-4-1-reasoning
    - grok-4-1

  OpenAI:
    - gpt-5 / gpt-5.x (Responses API, default medium reasoning)
    - gpt-5.2-2025-12-11
    - chatgpt-4o-latest
    - gpt-5-chat-latest
    - gpt-4o
    - gpt-4-turbo

Examples:
  # Use default models (both Claude Sonnet 4.5)
  python3 main.py

  # Claude Opus 4.5 vs OpenAI GPT-5.2
  python3 main.py --model-a claude-opus-4-5-20251101 --model-b gpt-5.2-2025-12-11

  # OpenAI vs OpenAI
  python3 main.py --model-a chatgpt-4o-latest --model-b gpt-5-chat-latest

  # Mix Anthropic models
  python3 main.py --model-a claude-opus-4-5-20251101 --model-b claude-haiku-4-5-20251001
        """
    )

    parser.add_argument(
        "--model-a",
        type=str,
        default=params.MODEL_A,
        help=f"Model to use for Model A (default: {params.MODEL_A})"
    )

    parser.add_argument(
        "--model-b",
        type=str,
        default=params.MODEL_B,
        help=f"Model to use for Model B (default: {params.MODEL_B})"
    )

    parser.add_argument(
        "--turns",
        type=int,
        default=params.MAX_TURNS,
        help=f"Maximum number of turns (default: {params.MAX_TURNS})"
    )

    parser.add_argument(
        "--temperature-a",
        type=float,
        default=params.TEMPERATURE_A,
        help=f"Temperature for Model A (default: {params.TEMPERATURE_A})"
    )

    parser.add_argument(
        "--temperature-b",
        type=float,
        default=params.TEMPERATURE_B,
        help=f"Temperature for Model B (default: {params.TEMPERATURE_B})"
    )

    parser.add_argument(
        "--thinking-budget-a",
        type=int,
        default=params.THINKING_BUDGET_A,
        help=f"Thinking budget for Model A (default: {params.THINKING_BUDGET_A})"
    )

    parser.add_argument(
        "--thinking-budget-b",
        type=int,
        default=params.THINKING_BUDGET_B,
        help=f"Thinking budget for Model B (default: {params.THINKING_BUDGET_B})"
    )

    parser.add_argument(
        "--final-question-a",
        type=str,
        default=params.FINAL_QUESTION_A,
        help="Optional question to ask Model A after the conversation completes"
    )

    parser.add_argument(
        "--final-question-b",
        type=str,
        default=params.FINAL_QUESTION_B,
        help="Optional question to ask Model B after the conversation completes"
    )

    parser.add_argument(
        "--pricing-file",
        type=str,
        default=None,
        help="Optional JSON pricing file (USD per 1M tokens) to estimate cost (default: pricing.json if present, or env PRICING_FILE)"
    )

    args = parser.parse_args()

    if args.turns % 2 != 0:
        parser.error(f"--turns must be even so turns split evenly between models; got {args.turns}")

    # Calculate actual turns per model based on max_turns
    actual_turns_per_model = args.turns // 2
    model_a = normalize_model_name(args.model_a)
    model_b = normalize_model_name(args.model_b)

    # Create system prompts for each model using separate prompts from params
    system_prompt_a = params.SYSTEM_PROMPT_A.format(
        assistant_name=get_assistant_name(model_a),
        developer=get_developer_name(model_a),
        model=model_a,
        turns_per_model=actual_turns_per_model,
        max_turns=args.turns
    )
    system_prompt_b = params.SYSTEM_PROMPT_B.format(
        assistant_name=get_assistant_name(model_b),
        developer=get_developer_name(model_b),
        model=model_b,
        turns_per_model=actual_turns_per_model,
        max_turns=args.turns
    )

    # Format final questions with placeholders if they exist
    final_question_a = None
    if args.final_question_a:
        final_question_a = args.final_question_a.format(
            assistant_name=get_assistant_name(model_a),
            developer=get_developer_name(model_a),
            model=model_a
        )

    final_question_b = None
    if args.final_question_b:
        final_question_b = args.final_question_b.format(
            assistant_name=get_assistant_name(model_b),
            developer=get_developer_name(model_b),
            model=model_b
        )

    run_conversation(
        system_prompt_a=system_prompt_a,
        system_prompt_b=system_prompt_b,
        start_a=params.START_MESSAGE_A,
        start_b=params.START_MESSAGE_B,
        max_turns=args.turns,
        temperature_a=args.temperature_a,
        temperature_b=args.temperature_b,
        output_dir=params.OUTPUT_DIR,
        model_a=model_a,
        model_b=model_b,
        thinking_budget_a=args.thinking_budget_a,
        thinking_budget_b=args.thinking_budget_b,
        final_question_a=final_question_a,
        final_question_b=final_question_b,
        pricing_file=args.pricing_file
    )

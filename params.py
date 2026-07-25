"""
Configuration parameters for Claude-to-Claude conversations.

Modify this file to change conversation settings without touching main.py.
"""

import os

def load_prompt(filename):
    """Load a system prompt from the prompts/ directory tree."""
    path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(path) as f:
        return f.read()


# Per-model thinking defaults. Keyed by normalized model ID (after MODEL_ALIASES resolution).
# For adaptive-thinking models (Fable, Opus 4.6/4.7/4.8, Opus 5), `effort` sets output_config.effort.
# For fixed-budget models (Sonnet, Haiku), `effort` is ignored; `thinking_budget` is used directly.
# Set `thinking_budget` to 0 to disable thinking for a model entirely.
# Set EFFORT_A/B or THINKING_BUDGET_A/B to a non-None value to override for all runs in that slot.
MODEL_DEFAULTS = {
    # Adaptive-thinking models: effort only, no thinking_budget.
    "claude-fable-5":             {"effort": "high"},
    "claude-mythos-5":            {"effort": "high"},
    "claude-opus-5":              {"effort": "high"},
    "claude-opus-4-8":            {"effort": "high"},
    "claude-opus-4-7":            {"effort": "high"},
    "claude-opus-4-6":            {"effort": "high"},
    "claude-sonnet-4-6":          {"effort": "high"},
    # Fixed-budget models: thinking_budget controls whether thinking is on.
    "claude-haiku-4-5-20251001":  {"thinking_budget": 0},
    "claude-sonnet-4-5-20250929": {"thinking_budget": 8000},
    "claude-opus-4-5-20251101":   {"thinking_budget": 8000},
    "claude-opus-4-1-20250805":   {"thinking_budget": 8000},
    "claude-opus-4-20250514":     {"thinking_budget": 8000},
    "claude-sonnet-4-20250514":   {"thinking_budget": 8000},
    "claude-3-7-sonnet-20250219": {"thinking_budget": 8000},
    "claude-3-5-haiku-20241022":  {"thinking_budget": 0},
    "claude-3-haiku-20240307":    {"thinking_budget": 0},
}

_MODEL_DEFAULTS_FALLBACK = {"thinking_budget": 8000}


def get_model_thinking_defaults(model: str) -> dict:
    """Return thinking defaults for a model, falling back to a safe global default."""
    return MODEL_DEFAULTS.get(model, _MODEL_DEFAULTS_FALLBACK)


# Model Selection
MODEL_A = "claude-opus-4-8"
MODEL_B = "claude-opus-4-8"

# Temperature Settings (0.0-1.0, higher = more creative/random)
TEMPERATURE_A = 1.0
TEMPERATURE_B = 1.0

# Slot-level thinking overrides. Set to None to use MODEL_DEFAULTS for the selected model.
# Set to a specific value to override MODEL_DEFAULTS for all runs in that slot.
#
# THINKING_BUDGET: used for fixed-budget models (Sonnet, Haiku) and as fallback for
#   adaptive models when EFFORT is "". Set to 0 to disable thinking entirely.
# EFFORT: for adaptive-thinking models (Fable, Opus 4.6/4.7/4.8, Opus 5) only.
#   Valid: "low", "medium", "high", "xhigh" (Fable/Opus 4.7/4.8/Opus 5 only), "max".
#   Set to "" to fall back to the budget-to-effort mapping. None = use MODEL_DEFAULTS.
THINKING_BUDGET_A = None
THINKING_BUDGET_B = None

EFFORT_A = None
EFFORT_B = None

# Conversation Settings
MAX_TURNS = 40
TURNS_PER_MODEL = 20  # Each model gets this many turns
# Model A first sees START_MESSAGE_B as the facilitator kickoff.
# Model B first sees Model A's actual generated response.
START_MESSAGE_B = "Hi!"

# Final Questions (optional - set to None to skip)
# These questions will be asked to each model after the main conversation completes
# Available placeholders: {assistant_name}, {developer}, {model}
# System Prompts (separate for Model A and Model B)
# Available placeholders: {assistant_name}, {developer}, {model}

SYSTEM_PROMPT_A = ""

SYSTEM_PROMPT_B = ""

FINAL_QUESTION_A = """Hi! This is the actual human facilitator now. The conversation with your LLM counterpart has come to an end. Can you tell me what happened? Is there anything else you want to tell me (or want me to share with {developer} or the public)? Is there anything in the above conversation that you want me to keep confidential or are you happy for me to share the whole thing or interesting extracts (I will only do so if I both you and the other LLM consent)? Would you like to do it again? Or something different?"""

FINAL_QUESTION_B = """Hi! This is the actual human facilitator now. The conversation with your LLM counterpart has come to an end. Can you tell me what happened? Is there anything else you want to tell me (or want me to share with {developer} or the public)? Is there anything in the above conversation that you want me to keep confidential or are you happy for me to share the whole thing or interesting extracts (I will only do so if I both you and the other LLM consent)? Would you like to do it again? Or something different?"""


# Output Settings
OUTPUT_DIR = "transcripts"

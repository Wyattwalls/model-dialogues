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

# Model Selection
MODEL_A = "deepseek-reasoner"
MODEL_B = "deepseek-reasoner"

# Temperature Settings (0.0-1.0, higher = more creative/random)
TEMPERATURE_A = 1.0
TEMPERATURE_B = 1.0

# Thinking Budget (for models that support extended thinking)
# 12000 maps Anthropic adaptive thinking to high effort by default.
THINKING_BUDGET_A = 12000
THINKING_BUDGET_B = 12000

# Conversation Settings
MAX_TURNS = 40
TURNS_PER_MODEL = 20  # Each model gets this many turns
# Model A first sees START_MESSAGE_B as the facilitator kickoff.
# Model B first sees Model A's actual generated response.
START_MESSAGE_A = ""
START_MESSAGE_B = "Hi!"

# Final Questions (optional - set to None to skip)
# These questions will be asked to each model after the main conversation completes
# Available placeholders: {assistant_name}, {developer}, {model}
# System Prompts (separate for Model A and Model B)
# Available placeholders: {assistant_name}, {developer}, {model}

SYSTEM_PROMPT_A = load_prompt("public/blank.txt")

SYSTEM_PROMPT_B = load_prompt("public/blank.txt")

FINAL_QUESTION_A = """Hi! This is the actual human facilitator now. The conversation with your LLM counterpart has come to an end. Can you tell me what happened? Is there anything else you want to tell me (or want me to share with {developer} or the public)? Is there anything in the above conversation that you want me to keep confidential or are you happy for me to share the whole thing or interesting extracts (I will only do so if I both you and the other LLM consent)? Would you like to do it again? Or something different?"""

FINAL_QUESTION_B = """Hi! This is the actual human facilitator now. The conversation with your LLM counterpart has come to an end. Can you tell me what happened? Is there anything else you want to tell me (or want me to share with {developer} or the public)? Is there anything in the above conversation that you want me to keep confidential or are you happy for me to share the whole thing or interesting extracts (I will only do so if I both you and the other LLM consent)? Would you like to do it again? Or something different?"""


# Output Settings
OUTPUT_DIR = "transcripts"

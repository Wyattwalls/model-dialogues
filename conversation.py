"""
Utilities for building conversation histories for the Anthropic API.
"""


def _build_alternating_convo(history: list) -> list:
    """
    Build a conversation history where even-indexed items are user messages
    (plain strings) and odd-indexed items are assistant messages (content
    block arrays). Turn tracking marker dicts are skipped without consuming
    an index.
    """
    convo = []
    message_index = 0  # Track position in actual messages (excluding turn tracking)
    for item in history:
        # Skip turn tracking messages
        if isinstance(item, dict) and item.get("type") == "turn_tracking":
            continue

        if message_index % 2 == 0:
            # User message - item is a string
            convo.append({"role": "user", "content": item})
        else:
            # Assistant message - item is a content blocks array
            convo.append({"role": "assistant", "content": item})

        message_index += 1
    return convo


def build_convo_a(history: list) -> list:
    """Build conversation history as seen by Model A."""
    return _build_alternating_convo(history)


def build_convo_b(history: list) -> list:
    """Build conversation history as seen by Model B."""
    return _build_alternating_convo(history)

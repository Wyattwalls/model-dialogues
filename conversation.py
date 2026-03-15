"""
Utilities for building conversation histories for the Anthropic API.
"""


def build_convo_a(history: list) -> list:
    """
    Build conversation history for Model A.

    Model A sees odd-indexed items as user messages and
    even-indexed items (after 0) as assistant messages.

    Turn tracking messages are filtered out.

    Args:
        history: List where items are either:
            - strings (for user messages)
            - content block arrays (for assistant messages)
            - turn tracking message dicts (filtered out)
    """
    convo = []
    message_index = 0  # Track position in actual messages (excluding turn tracking)
    for item in history:
        # Skip turn tracking messages
        if isinstance(item, dict) and item.get("type") == "turn_tracking":
            continue

        if message_index != 0 and message_index % 2 == 0:
            # Assistant message - item is a content blocks array
            convo.append({"role": "assistant", "content": item})
        elif message_index % 2 != 0:
            # User message - item is a string
            convo.append({"role": "user", "content": item})

        message_index += 1
    return convo


def build_convo_b(history: list) -> list:
    """
    Build conversation history for Model B.

    Model B sees even-indexed items as user messages and
    odd-indexed items as assistant messages.

    Turn tracking messages are filtered out.

    Args:
        history: List where items are either:
            - strings (for user messages)
            - content block arrays (for assistant messages)
            - turn tracking message dicts (filtered out)
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

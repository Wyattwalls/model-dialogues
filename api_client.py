"""
API client and response generation for Anthropic, OpenAI, Google Gemini, and xAI Grok models.
"""

import os
import anthropic
import openai
from google import genai
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user, system as xai_system

from costing import normalize_usage


# Anthropic models that support extended thinking
THINKING_MODELS = {
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

# Anthropic models that use adaptive thinking instead of fixed budget_tokens.
ADAPTIVE_THINKING_MODELS = {
    "claude-opus-4-6",
}

# OpenAI models
OPENAI_MODELS = {
    "gpt-5.2-2025-12-11",
    "chatgpt-4o-latest",
    "gpt-5-chat-latest",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
}

# Google Gemini models - use dynamic detection via prefixes
# The API has 30+ models (gemini-2.5-flash, gemini-3-flash-preview, gemma-3-*, etc.)
# Updated list can be fetched via: client.models.list()

# xAI Grok models that support reasoning/thinking
GROK_THINKING_MODELS = {
    "grok-4-1-fast-reasoning",
    "grok-4-1-reasoning",
    "grok-4-1",
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


def is_openai_model(model: str) -> bool:
    """Check if a model is an OpenAI model."""
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


def uses_openai_responses_api(model: str) -> bool:
    """Check if an OpenAI model should use the Responses API."""
    return model.startswith("gpt-5") and not model.endswith("chat-latest")


def uses_openai_reasoning(model: str) -> bool:
    """Check if an OpenAI model should use GPT-5 reasoning defaults."""
    return uses_openai_responses_api(model)


def supports_thinking(model: str) -> bool:
    """Check if a model supports extended thinking."""
    return (
        model in THINKING_MODELS
        or model in GROK_THINKING_MODELS
        or model.startswith(GEMINI_THINKING_PREFIXES)
        or uses_openai_reasoning(model)
    )


def uses_adaptive_thinking(model: str) -> bool:
    """Check if a model should use Anthropic adaptive thinking."""
    return model in ADAPTIVE_THINKING_MODELS


def uses_gemini_thinking_level(model: str) -> bool:
    """Check if a Gemini model should use thinking_level instead of budget."""
    return model.startswith("gemini-3")


def thinking_budget_to_effort(thinking_budget: int) -> str:
    """Map the existing budget control onto Anthropic's adaptive effort levels."""
    if thinking_budget <= 1024:
        return "low"
    if thinking_budget <= 4096:
        return "medium"
    if thinking_budget <= 12000:
        return "high"
    return "max"


def get_max_tokens(model: str, default: int = 16000) -> int:
    """Get the maximum tokens allowed for a model."""
    return MODEL_MAX_TOKENS.get(model, default)


def generate_response(
    anthropic_client: anthropic.Anthropic,
    openai_client: openai.OpenAI,
    conversation: list,
    system_prompt: str,
    temperature: float = 1.0,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = None,
    thinking_budget: int = 10000,
    gemini_client = None,
    xai_client = None
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
    if is_grok_model(model):
        return _generate_grok_response(
            xai_client, conversation, system_prompt, temperature, model, max_tokens
        )
    elif is_gemini_model(model):
        return _generate_gemini_response(
            gemini_client, conversation, system_prompt, temperature, model, max_tokens, thinking_budget
        )
    elif is_openai_model(model):
        return _generate_openai_response(
            openai_client, conversation, system_prompt, temperature, model, max_tokens, thinking_budget
        )
    else:
        return _generate_anthropic_response(
            anthropic_client, conversation, system_prompt, temperature,
            model, max_tokens, thinking_budget
        )


def _generate_anthropic_response(
    client: anthropic.Anthropic,
    conversation: list,
    system_prompt: str,
    temperature: float,
    model: str,
    max_tokens: int,
    thinking_budget: int
) -> tuple[list, str, str, dict]:
    """Generate response using Anthropic API."""
    use_thinking = supports_thinking(model)

    api_params = {
        "model": model,
        "system": system_prompt,
        "max_tokens": max_tokens,
        "messages": conversation,
        "temperature": temperature,
    }

    if use_thinking:
        if uses_adaptive_thinking(model):
            api_params["thinking"] = {
                "type": "adaptive",
            }
            api_params["output_config"] = {
                "effort": thinking_budget_to_effort(thinking_budget)
            }
        else:
            api_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }

    with client.messages.stream(**api_params) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if hasattr(event.delta, "type"):
                    if event.delta.type == "thinking_delta":
                        print(event.delta.thinking, end="", flush=True)
                    elif event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)

        # Stream may be closed on context exit; get the final message while still inside.
        final_message = stream.get_final_message()
    content_blocks = final_message.content
    usage = normalize_usage(provider="anthropic", model=model, usage_obj=getattr(final_message, "usage", None))

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
        api_params = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
            # GPT-5-family reasoning models default to medium effort in the docs.
            "reasoning": {"effort": "medium"},
        }
        if system_prompt:
            api_params["instructions"] = system_prompt

        response_text = ""
        with client.responses.stream(**api_params) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                    response_text += event.delta

            final_response = stream.get_final_response()

        if not response_text:
            response_text = final_response.output_text

        reasoning_summaries = []
        for item in getattr(final_response, "output", []):
            if getattr(item, "type", None) != "reasoning":
                continue
            for summary in getattr(item, "summary", []) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning_summaries.append(text)

        content_blocks = [{"type": "text", "text": response_text}]
        usage = normalize_usage(provider="openai", model=model, usage_obj=getattr(final_response, "usage", None))
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

    # Use max_completion_tokens for newer models (GPT-4o and later)
    if "gpt-5" in model or "chatgpt-4o" in model or model == "gpt-4o":
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

    # Return Anthropic-compatible content blocks as plain dicts so the same history
    # structure can be fed back into either provider.
    content_blocks = [{"type": "text", "text": response_text}]

    usage = normalize_usage(provider="openai", model=model, usage_obj=usage_obj)
    return content_blocks, "", response_text, usage


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
            # Note: xAI SDK in stateless mode only needs user messages
            # The assistant responses are managed internally when using previous_response_id
            # For stateless without ID, we only send user messages

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

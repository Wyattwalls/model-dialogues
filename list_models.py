#!/usr/bin/env python3
"""
List available models from Anthropic, OpenAI, and Google APIs.

Usage:
    python list_models.py              # List all models
    python list_models.py --anthropic  # Only Anthropic models
    python list_models.py --openai     # Only OpenAI models
    python list_models.py --google     # Only Google models
    python list_models.py --google --details  # Show detailed Google model info
"""

import os
import argparse
from dotenv import load_dotenv
import anthropic
import openai
from google import genai


def list_anthropic_models():
    """List available Anthropic/Claude models."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')

    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return

    try:
        client = anthropic.Anthropic(api_key=api_key)
        models = client.models.list()

        print("\n" + "="*70)
        print("🤖 ANTHROPIC MODELS")
        print("="*70)

        if not models.data:
            print("No models available")
            return

        for model in models.data:
            print(f"\n📦 {model.id}")
            if hasattr(model, 'display_name'):
                print(f"   Display name: {model.display_name}")
            if hasattr(model, 'created_at'):
                print(f"   Created: {model.created_at}")

        print(f"\n✅ Total Anthropic models: {len(models.data)}")

    except Exception as e:
        print(f"❌ Error listing Anthropic models: {e}")


def list_openai_models():
    """List available OpenAI models."""
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return

    try:
        client = openai.OpenAI(api_key=api_key)
        models = client.models.list()

        print("\n" + "="*70)
        print("🤖 OPENAI MODELS")
        print("="*70)

        # Filter to show relevant chat models
        chat_models = [m for m in models.data if any(x in m.id for x in [
            'gpt-4', 'gpt-5', 'o1', 'o3', 'chatgpt'
        ])]

        if not chat_models:
            print("No chat models available")
            return

        # Sort by ID for better readability
        chat_models.sort(key=lambda m: m.id)

        for model in chat_models:
            print(f"\n📦 {model.id}")
            if hasattr(model, 'created'):
                from datetime import datetime
                created_date = datetime.fromtimestamp(model.created)
                print(f"   Created: {created_date}")
            if hasattr(model, 'owned_by'):
                print(f"   Owner: {model.owned_by}")

        print(f"\n✅ Total chat models: {len(chat_models)}")
        print(f"   (Filtered from {len(models.data)} total models)")

    except Exception as e:
        print(f"❌ Error listing OpenAI models: {e}")


def list_google_models(show_details: bool = False):
    """List available Google Gemini models with generateContent support."""
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        return

    try:
        client = genai.Client(api_key=api_key)

        print("\n" + "=" * 70)
        print("🤖 GOOGLE MODELS")
        print("=" * 70)

        if show_details:
            print(
                f"\n{'Model Name':<45} {'Display Name':<35} {'Thinking':<10} "
                f"{'Max Out':<10} {'Max In':<10}"
            )
            print("=" * 120)
        else:
            print(f"\n{'Model Name':<45} {'Thinking':<10} {'Max Tokens':<12}")
            print("=" * 75)

        models_found = []

        for model in client.models.list():
            if not getattr(model, "supported_actions", None):
                continue
            if "generateContent" not in model.supported_actions:
                continue

            name = model.name.replace("models/", "")
            models_found.append(model)

            thinking = "✓" if getattr(model, "thinking", None) else "-"
            if show_details:
                display_name = getattr(model, "display_name", "")
                if len(display_name) > 35:
                    display_name = display_name[:33] + ".."
                max_out = getattr(model, "output_token_limit", "N/A")
                max_in = getattr(model, "input_token_limit", "N/A")
                print(
                    f"{name:<45} {display_name:<35} {thinking:<10} "
                    f"{str(max_out):<10} {str(max_in):<10}"
                )
            else:
                max_tokens = getattr(model, "output_token_limit", "N/A")
                print(f"{name:<45} {thinking:<10} {str(max_tokens):<12}")

        if not models_found:
            print("No Google models available")
            return

        print(f"\n✅ Total Google models: {len(models_found)}")
        print("   ✓ = Supports thinking/reasoning")

    except Exception as e:
        print(f"❌ Error listing Google models: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="List available models from Anthropic, OpenAI, and Google APIs"
    )
    parser.add_argument(
        "--anthropic",
        action="store_true",
        help="Only list Anthropic models"
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Only list OpenAI models"
    )
    parser.add_argument(
        "--google",
        action="store_true",
        help="Only list Google models"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed Google model information"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # If no specific provider requested, list all supported providers.
    no_provider_selected = not args.anthropic and not args.openai and not args.google
    list_anthropic_flag = args.anthropic or no_provider_selected
    list_openai_flag = args.openai or no_provider_selected
    list_google_flag = args.google or no_provider_selected

    if list_anthropic_flag:
        list_anthropic_models()

    if list_openai_flag:
        list_openai_models()

    if list_google_flag:
        list_google_models(show_details=args.details)

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

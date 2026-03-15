#!/usr/bin/env python3
"""
Utility script to list all available Google Gemini models.
Run this to see what models are currently available for experiments.
"""

import os
from google import genai
from dotenv import load_dotenv

def list_models(show_details=False):
    """List all available Gemini models with generateContent support."""
    load_dotenv()
    api_key = os.environ.get('GOOGLE_API_KEY')

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in .env file")
        return

    client = genai.Client(api_key=api_key)

    print('Available Gemini models with generateContent support:\n')

    if show_details:
        print(f"{'Model Name':<45} {'Display Name':<35} {'Thinking':<10} {'Max Out':<10} {'Max In':<10}")
        print('=' * 120)
    else:
        print(f"{'Model Name':<45} {'Thinking':<10} {'Max Tokens':<12}")
        print('=' * 75)

    models_found = []

    for model in client.models.list():
        if hasattr(model, 'supported_actions') and model.supported_actions:
            if 'generateContent' in model.supported_actions:
                name = model.name.replace('models/', '')
                thinking = '✓' if (hasattr(model, 'thinking') and model.thinking) else '-'

                if show_details:
                    display = (model.display_name[:33] + '..') if hasattr(model, 'display_name') and len(model.display_name) > 35 else (model.display_name if hasattr(model, 'display_name') else '')
                    max_out = str(model.output_token_limit) if hasattr(model, 'output_token_limit') else 'N/A'
                    max_in = str(model.input_token_limit) if hasattr(model, 'input_token_limit') else 'N/A'
                    print(f'{name:<45} {display:<35} {thinking:<10} {max_out:<10} {max_in:<10}')
                else:
                    max_tokens = str(model.output_token_limit) if hasattr(model, 'output_token_limit') else 'N/A'
                    print(f'{name:<45} {thinking:<10} {max_tokens:<12}')

                models_found.append(name)

    print(f'\n✓ = Supports thinking/reasoning')
    print(f'\nTotal models found: {len(models_found)}')

    # Show models grouped by family
    print('\n\nModels by family:')
    families = {}
    for model in models_found:
        if model.startswith('gemini-'):
            if '3-' in model or '3.' in model:
                family = 'Gemini 3.x'
            elif '2.5' in model:
                family = 'Gemini 2.5'
            elif '2.0' in model or '2-' in model:
                family = 'Gemini 2.0'
            elif '1.5' in model:
                family = 'Gemini 1.5'
            elif 'robotics' in model:
                family = 'Specialized (Robotics)'
            elif 'computer-use' in model:
                family = 'Specialized (Computer Use)'
            elif 'flash-latest' in model or 'pro-latest' in model:
                family = 'Aliases (Latest)'
            else:
                family = 'Other Gemini'
        elif model.startswith('gemma-'):
            family = 'Gemma (Open Models)'
        elif model.startswith('deep-research-'):
            family = 'Specialized (Research)'
        elif model.startswith('nano-banana-'):
            family = 'Specialized (Nano Banana)'
        else:
            family = 'Other'

        if family not in families:
            families[family] = []
        families[family].append(model)

    for family, models in sorted(families.items()):
        print(f'\n{family}:')
        for model in models:
            print(f'  - {model}')

if __name__ == '__main__':
    import sys
    show_details = '--details' in sys.argv or '-d' in sys.argv

    if '--help' in sys.argv or '-h' in sys.argv:
        print('Usage: python list_gemini_models.py [--details|-d]')
        print('\nOptions:')
        print('  --details, -d    Show detailed model information')
        print('  --help, -h       Show this help message')
        sys.exit(0)

    list_models(show_details=show_details)

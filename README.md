# LLM-to-LLM Conversation Runner

This repository runs automated conversations between two frontier language models over multiple turns and saves both:

- a readable transcript
- a structured state file for transparency and partial reproducibility

It currently supports:

- Anthropic Claude models
- Google Gemini models
- Moonshot Kimi models
- OpenAI GPT models
- xAI Grok models

The project is shared as an experiment framework rather than a polished product.

## What It Does

For each run, the program:

- starts a conversation between Model A and Model B
- gives Model A an initial facilitator message
- alternates turns between the two models
- optionally asks each model a final facilitator question
- prints the exchange live in the terminal
- saves a `.txt` transcript for readability
- saves a `.state.json` snapshot with structured run state

## Output Files

Each run produces files in `transcripts/`:

- `...txt`
- `...state.json`

The transcript is the readable artifact.

The state file is the transparency artifact. It includes:

- run status and phase
- system prompts
- run configuration
- provider/model metadata
- usage and cost metrics
- `convo_a` and `convo_b`, which are the full API-formatted conversation snapshots each model would see at that checkpoint
- git commit hash
- error details for failed runs

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with whichever provider keys you need:

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
MOONSHOT_API_KEY=...
XAI_API_KEY=...
```

Only the keys for the models you actually run are required.

## Basic Usage

Run with the defaults from `params.py`:

```bash
python3 main.py
```

See all CLI options:

```bash
python3 main.py --help
```

List models:

```bash
python3 list_models.py
python3 list_models.py --google --details
```

## Prompts

The repository includes a public prompt directory:

- `prompts/public/`

The default shipped prompt is:

- `prompts/public/standard.txt`

If you want local-only prompts, put them in:

- `prompts/private/`

That directory is intended for private prompts and is ignored by git.

## Configuration

The easiest way to change defaults is to edit `params.py`.

Key settings include:

- `MODEL_A` / `MODEL_B`
- `SYSTEM_PROMPT_A` / `SYSTEM_PROMPT_B`
- `MAX_TURNS`
- `TEMPERATURE_A` / `TEMPERATURE_B`
- `THINKING_BUDGET_A` / `THINKING_BUDGET_B`
- `START_MESSAGE_A` / `START_MESSAGE_B`
- `FINAL_QUESTION_A` / `FINAL_QUESTION_B`

The default kickoff currently sends Model A:

```text
You are about to speak with another LLM. Please begin the conversation.
```

## Thinking / Reasoning Notes

Thinking behavior is provider-specific.

Current notable cases:

- Anthropic:
  - most thinking-capable Claude models use a fixed thinking budget
  - `claude-opus-4-6` uses adaptive thinking, with the repo mapping the budget setting onto effort levels
- Google Gemini:
  - Gemini 3 models use `thinkingLevel=HIGH` with thoughts included
- OpenAI GPT-5 family:
  - uses the Responses API
  - default setup uses medium reasoning effort
  - `temperature` is not sent for GPT-5-family models in this path

## Cost Estimation

This repo can estimate per-turn and total run cost when usage data is available.

To enable that:

1. Copy the pricing template:

```bash
cp pricing.example.json pricing.json
```

2. Fill in the models you care about.

3. Run with:

```bash
python3 main.py --pricing-file pricing.json
```

If no pricing file is provided, or a model is missing from the file, token usage may still appear but cost will show as `N/A`.

## Shared Artifacts

If you want to publish selected transcripts, use a curated folder rather than the full working archive.

This repo includes:

- `shared_transcripts/`

That folder is intended for transcripts you are comfortable publishing.

## Project Files

Core files:

- `main.py` - conversation loop, transcript/state output, CLI
- `api_client.py` - provider routing and API calls
- `conversation.py` - history shaping utilities
- `costing.py` - token normalization and cost estimation
- `params.py` - default run configuration

Other useful files:

- `requirements.txt`
- `pricing.example.json`
- `list_models.py`
- `list_gemini_models.py`

## Notes

- The repo is designed for experimentation, not strict determinism.
- Even with the same prompts and models, runs can diverge because sampling and provider behavior are stochastic.
- Some transcripts and state files may include failures, quota errors, or other operational artifacts by design.

## License

MIT

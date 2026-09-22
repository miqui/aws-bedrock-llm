# aws-bedrock-llm

Minimal Python package + CLI to send a prompt to an AWS Bedrock LLM using the
boto3 **Converse API** (`bedrock-runtime.converse()`).

## Install

```bash
uv sync
```

Requires Python >= 3.11 (uses stdlib `tomllib`). Only runtime dependency is `boto3`.
Uses [uv](https://docs.astral.sh/uv/) for dependency management; `uv sync` creates
`.venv` and installs runtime + dev (ruff) dependencies from `uv.lock`.

## Configuration

Settings resolved: `model_id`, `region`, `max_tokens`, `temperature`, `profile`.

**Precedence (highest to lowest), evaluated per-key:**

1. Explicit keyword argument to `load_config()` / a `BedrockConfig(...)` you build yourself.
2. A TOML config file, **if it exists**, for any key it defines under `[bedrock]`.
3. Environment variable — used when the TOML file doesn't exist at all, OR
   the file exists but doesn't define that particular key.
4. Built-in default.

So: "toml present -> toml wins for values it defines; missing file -> env var
fallback; sane defaults last." A toml file that only sets `model_id` will
still let `region`/`max_tokens`/etc. fall through to env vars or defaults.

Config file location: `config_path` arg > `AWS_BEDROCK_CONFIG_PATH` env var >
`config.toml` in the current working directory.

### `config.toml` example (repo root)

```toml
[bedrock]
model_id = "us.anthropic.claude-sonnet-4-6"
region = "us-east-1"
max_tokens = 1024
temperature = 0.7
# profile = "default"
```

### Environment variables (fallback)

| Variable | Meaning | Default |
|---|---|---|
| `AWS_BEDROCK_MODEL_ID` | Bedrock model id | `us.anthropic.claude-sonnet-4-6` |
| `AWS_BEDROCK_REGION` | AWS region | `us-east-1` |
| `AWS_BEDROCK_MAX_TOKENS` | max tokens for inference config | `1024` |
| `AWS_BEDROCK_TEMPERATURE` | sampling temperature | `0.7` |
| `AWS_BEDROCK_PROFILE` | named AWS credentials profile | none (default chain) |
| `AWS_BEDROCK_CONFIG_PATH` | path to the TOML config file | `config.toml` |

AWS credentials are **never** read from config/env by this package — boto3's
standard credential resolution chain (env vars, shared config/credentials
files, SSO, instance/task role, etc.) is used as-is via `boto3.Session`.

## Programmatic use

```python
from aws_bedrock_llm import send_prompt

text = send_prompt("Explain the Converse API in one sentence.")
print(text)
```

Or resolve config yourself:

```python
from aws_bedrock_llm import load_config, send_prompt

cfg = load_config("path/to/config.toml")
text = send_prompt("Hello", cfg=cfg)
```

## Run manually (CLI)

```bash
# Console script (after `uv sync`)
uv run aws-bedrock-llm "What is Amazon Bedrock?"

# Or as a module
uv run python -m aws_bedrock_llm "What is Amazon Bedrock?"

# Point at a specific config file
uv run aws-bedrock-llm --config /path/to/config.toml "Hello there"
```

Expected output shape (stderr shows the resolved config, stdout is the model's reply):

```
[aws-bedrock-llm] model_id=us.anthropic.claude-sonnet-4-6 region=us-east-1 max_tokens=1024 temperature=0.7
<model's text response here>
```

This requires real AWS credentials with Bedrock access and will make a live
network call to AWS — it is not exercised by the test suite.

## Tests

Stdlib `unittest` only, no network calls (the bedrock-runtime client is
faked in tests):

```bash
uv run python -m unittest discover -v
```

## Notes / defaults

- Default model id `us.anthropic.claude-sonnet-4-6` is a placeholder default
  for "a current Claude model on Bedrock" — override via config/env for your
  account's actually-enabled model id.
- `botocore.config.Config(retries={"max_attempts": 5, "mode": "adaptive"})`
  is always applied to the bedrock-runtime client.
- `maxTokens` is always explicitly set in `inferenceConfig` sent to `converse()`.

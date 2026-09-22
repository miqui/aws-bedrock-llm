"""Configuration resolution for aws_bedrock_llm.

Precedence for each setting (highest first):
1. Explicit keyword argument passed to `load_config()` / `BedrockConfig(...)`.
2. Value present in the TOML config file, if the file exists.
3. Environment variable, used only when the TOML file is absent OR the file
   exists but does not define that particular key.
4. Built-in default.

The TOML file's *presence* governs whether the env var is even consulted for
keys the file defines: if the file exists and sets a key, that value wins
outright (even if the env var is also set). If the file does not exist at
all, every key falls back to its environment variable. If the file exists
but omits a key, that individual key falls back to its environment variable.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7

ENV_MODEL_ID = "AWS_BEDROCK_MODEL_ID"
ENV_REGION = "AWS_BEDROCK_REGION"
ENV_MAX_TOKENS = "AWS_BEDROCK_MAX_TOKENS"
ENV_TEMPERATURE = "AWS_BEDROCK_TEMPERATURE"
ENV_PROFILE = "AWS_BEDROCK_PROFILE"
ENV_CONFIG_PATH = "AWS_BEDROCK_CONFIG_PATH"


@dataclass(frozen=True)
class BedrockConfig:
    model_id: str = DEFAULT_MODEL_ID
    region: str = DEFAULT_REGION
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    profile: str | None = None


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return data.get("bedrock", {})


def load_config(config_path: str | os.PathLike[str] | None = None) -> BedrockConfig:
    """Resolve a BedrockConfig using toml-file-first, env-var-fallback rules.

    Args:
        config_path: Optional explicit path to a TOML config file. If not
            given, uses the AWS_BEDROCK_CONFIG_PATH env var, else
            DEFAULT_CONFIG_PATH ("config.toml") relative to the current
            working directory.
    """
    path_str = str(config_path) if config_path is not None else os.environ.get(
        ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH
    )
    path = Path(path_str)

    toml_values: dict[str, Any] = {}
    if path.is_file():
        toml_values = _read_toml(path)

    def resolve(key: str, env_var: str, default: Any, cast=str) -> Any:
        if key in toml_values:
            return toml_values[key]
        env_val = os.environ.get(env_var)
        if env_val is not None:
            return cast(env_val)
        return default

    model_id = resolve("model_id", ENV_MODEL_ID, DEFAULT_MODEL_ID, str)
    region = resolve("region", ENV_REGION, DEFAULT_REGION, str)
    max_tokens = resolve("max_tokens", ENV_MAX_TOKENS, DEFAULT_MAX_TOKENS, int)
    temperature = resolve("temperature", ENV_TEMPERATURE, DEFAULT_TEMPERATURE, float)
    profile = resolve("profile", ENV_PROFILE, None, str)

    return BedrockConfig(
        model_id=model_id,
        region=region,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        profile=profile,
    )

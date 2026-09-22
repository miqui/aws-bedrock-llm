"""Client wrapper around the AWS Bedrock Runtime Converse API."""

from __future__ import annotations

from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from .config import BedrockConfig, load_config


def _build_boto_config() -> BotoConfig:
    return BotoConfig(retries={"max_attempts": 5, "mode": "adaptive"})


def get_client(cfg: BedrockConfig):
    """Build a boto3 bedrock-runtime client from a resolved BedrockConfig.

    Uses the standard boto3 credential resolution chain; no credentials are
    ever hardcoded here.
    """
    session_kwargs: dict[str, Any] = {}
    if cfg.profile:
        session_kwargs["profile_name"] = cfg.profile
    session = boto3.Session(**session_kwargs)
    return session.client(
        "bedrock-runtime",
        region_name=cfg.region,
        config=_build_boto_config(),
    )


def extract_text(response: dict[str, Any]) -> str:
    """Extract the assistant's text from a Converse API response dict."""
    message = response["output"]["message"]
    parts = message.get("content", [])
    texts = [block["text"] for block in parts if "text" in block]
    return "".join(texts)


def send_prompt(
    prompt: str,
    config_path: str | None = None,
    client: Any = None,
    cfg: BedrockConfig | None = None,
) -> str:
    """Send a single user prompt to Bedrock via the Converse API.

    Args:
        prompt: The user prompt text.
        config_path: Optional path to a TOML config file (see config.load_config).
        client: Optional pre-built bedrock-runtime client (mainly for testing).
        cfg: Optional pre-resolved BedrockConfig (skips re-resolving config).

    Returns:
        The extracted assistant response text.
    """
    if cfg is None:
        cfg = load_config(config_path)
    if client is None:
        client = get_client(cfg)

    inference_config = {
        "maxTokens": cfg.max_tokens,
        "temperature": cfg.temperature,
    }

    response = client.converse(
        modelId=cfg.model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=inference_config,
    )
    return extract_text(response)

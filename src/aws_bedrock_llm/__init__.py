"""aws_bedrock_llm: minimal package to send prompts to AWS Bedrock LLMs."""

from .client import extract_text, get_client, send_prompt
from .config import BedrockConfig, load_config

__all__ = [
    "BedrockConfig",
    "extract_text",
    "get_client",
    "load_config",
    "send_prompt",
]

__version__ = "0.1.0"

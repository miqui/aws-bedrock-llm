"""CLI entry point: `python -m aws_bedrock_llm` or `aws-bedrock-llm` console script."""

from __future__ import annotations

import argparse
import sys

from .client import send_prompt
from .config import DEFAULT_CONFIG_PATH, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aws-bedrock-llm",
        description="Send a prompt to an AWS Bedrock LLM via the Converse API.",
    )
    parser.add_argument("prompt", help="The prompt text to send to the model.")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help=(
            "Path to a TOML config file (default: %s, or "
            "$AWS_BEDROCK_CONFIG_PATH)" % DEFAULT_CONFIG_PATH
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config_path)
    print(f"[aws-bedrock-llm] model_id={cfg.model_id} region={cfg.region} "
          f"max_tokens={cfg.max_tokens} temperature={cfg.temperature}",
          file=sys.stderr)

    try:
        text = send_prompt(args.prompt, cfg=cfg)
    except Exception as exc:  # pragma: no cover - network/AWS error path
        print(f"Error calling Bedrock: {exc}", file=sys.stderr)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

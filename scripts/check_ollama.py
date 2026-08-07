"""Check Ollama server connectivity and model availability.

Usage:

    python scripts/check_ollama.py
    python scripts/check_ollama.py --model gemma3:4b
    python scripts/check_ollama.py --pull-command

The script does not automatically download a model. It prints the exact pull
command so model installation remains an explicit user action.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.ollama_llm_client import (
    OllamaLLMClient,
    OllamaLLMConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--model",
        default="gemma3:4b",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--pull-command",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.pull_command:
        print(f"ollama pull {args.model}")
        return 0

    client = OllamaLLMClient(
        OllamaLLMConfig(
            host=args.host,
            model=args.model,
            timeout_seconds=args.timeout,
        )
    )
    status = client.check_status()

    print(f"Host:            {status.host}")
    print(f"Reachable:       {status.reachable}")
    print(f"Requested model: {status.model}")
    print(f"Model available: {status.model_available}")

    if status.installed_models:
        print("Installed models:")
        for model in status.installed_models:
            print(f"  - {model}")

    if status.detail:
        print(f"Detail: {status.detail}")

    if not status.reachable:
        print("\nStart Ollama with: ollama serve")
        return 2
    if not status.model_available:
        print(f"\nInstall the model with: ollama pull {args.model}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

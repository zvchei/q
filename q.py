#!/usr/bin/env python3

from enum import Enum
from context import Context, Role
from gemini import Gemini
import sys


def main():
    command, prompts = parse_command_line()

    context_file = get_config_file("context")
    context_json = None

    if context_file.exists():
        with open(context_file, "r") as f:
            context_json = f.read()

    context = Context(context_json)
    llm = Gemini()
    
    if command.reset:
        print("[ Resetting the context. ]", file=sys.stderr)
        context.reset()

    if not command.log and len(prompts) > 0:
        context.add_text(Role.USER, prompts)
        response = llm.generate_response(context)
        context.add_text(Role.MODEL, [response])
        print(response)

    if command.log:
        print(context.to_json())

    with open(context_file, "w") as f:
        f.write(context.to_json())


def get_config_file(name: str) -> str:
    from pathlib import Path
    appname = "q_oracle"
    config_dir = Path.home() / ".config" / appname
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / f"{name}.json"


def parse_command_line():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Ask the LLM oracle.")
    parser.add_argument(
        '-r', '--reset', action='store_true', help="Reset the context"
    )
    parser.add_argument(
        '-l', '--log', action='store_true', help="Dump the context"
    )
    parser.add_argument(
        'inputs', nargs='*', help='Prompt input for the oracle.'
    )
    args = parser.parse_args()

    prompt = " ".join(args.inputs)

    # Include additional input from stdin if available, to extend the prompt
    extra_prompt = None
    if not sys.stdin.isatty():
        extra_prompt = sys.stdin.read().strip()

    # Remove the empty prompts
    prompts = [p for p in [prompt, extra_prompt] if p]

    return args, prompts


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from enum import Enum
from context import Context, Role
from gemini import Gemini
import sys


class Command(Enum):
    DUMP = "dump"
    RESET = "reset"


def main():
    commands, prompts = parse_command_line()

    context_file = get_config_file("context")
    context_json = None

    if context_file.exists():
        with open(context_file, "r") as f:
            context_json = f.read()

    context = Context(context_json)
    llm = Gemini()
    
    if Command.RESET in commands:
        print("WARNING: Resetting context.", file=sys.stderr)
        context.reset()

    if prompts and not Command.DUMP in commands:
        context.add_text(Role.USER, prompts)
        response = llm.generate_response(context)
        context.add_text("model", [response])
        print(response)

    if Command.DUMP in commands:
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
        '-d', '--debug', action='store_true', help="Dump the context"
    )
    parser.add_argument(
        'inputs', nargs='*', help='Prompt input for the oracle.'
    )
    args = parser.parse_args()

    commands = []
    if args.reset:
        commands.append(Command.RESET)
    if args.debug:
        commands.append(Command.DUMP)

    prompt = " ".join(args.inputs)

    # Include additional input from stdin if available, to extend the prompt
    extra_prompt = None
    if not sys.stdin.isatty():
        extra_prompt = sys.stdin.read().strip()

    # Remove the empty prompts
    prompts = [p for p in [prompt, extra_prompt] if p]
    return commands, prompts


if __name__ == "__main__":
    main()

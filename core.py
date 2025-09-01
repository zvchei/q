#!/usr/bin/env python3

import logging
import os
import tempfile

from enum import Enum
from context import Context, Role
from gemini import Gemini
from pathlib import Path


def execute_command(context_file, command, prompts):
    context_json = None

    if context_file.exists():
        logging.info(f"Loading existing context from {context_file}")
        with open(context_file, "r") as f:
            context_json = f.read()

    context = Context(context_json)
    llm = Gemini()
    
    if not command.log and len(prompts) > 0:
        context.add_text(Role.USER, prompts)
        response = llm.generate_response(context)
        context.add_text(Role.MODEL, [response])
        print(response)
    else:
        logging.info("No prompts provided; skipping inference.")

    if command.log:
        print(context.to_json())

    with open(context_file, "w") as f:
        f.write(context.to_json())


def parse_command_line():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Ask the LLM oracle.")
    parser.add_argument(
        '-l', '--log', action='store_true', help="Dump the context"
    )
    parser.add_argument(
        '--debug', action='store_true', help="Enable debug logging"
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


def get_process_stime(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            stats = f.read().split()
            # The start time (stime) is the 22nd field (index 21).
            # It's measured in clock ticks since system boot.
            return int(stats[21])
    except (FileNotFoundError):
        return None


def collect_garbage():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if not filename.startswith("q_context_") or not filename.endswith(".json"):
            continue

        try:
            parts = filename[len("q_context_") : -len(".json")].split("_")
            if len(parts) != 2:
                raise ValueError("Unexpected filename format")

            pid = int(parts[0])
            stime = int(parts[1])
            current_stime = get_process_stime(pid)

            if current_stime is None or current_stime != stime:
                filepath = os.path.join(temp_dir, filename)
                os.remove(filepath)
                logging.info(f"Removed stale context file: {filename}")
            else:
                logging.debug(f"Found active context file: {filename}; skipping deletion.")
        except ValueError:
            logging.warning(f"Skipping file with unexpected name format: {filename}")
        except OSError as e:
            logging.warning(f"Error removing file {filename}: {e}")

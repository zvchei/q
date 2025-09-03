#!/usr/bin/env python3

import json
import logging
import os
import subprocess
import tempfile
import urllib.error
import urllib.request

from abc import ABC, abstractmethod
from context import Context, Role


class LLMBackend(ABC):
    @abstractmethod
    def generate_response(self, context):
        pass


class FetchError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def fetch(url: str, data_json: str, headers: dict):
    request = urllib.request.Request(url, data=data_json.encode('utf-8'), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode('utf-8')
            return json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode('utf-8', errors='replace')
        except Exception:
            error_body = str(e)
        raise FetchError(error_body, code=e.code)
    except urllib.error.URLError as e:
        raise FetchError(str(e))


def execute_command(context_file, command, prompts, llm: LLMBackend):
    context_json = None

    if context_file.exists():
        logging.info(f"Loading existing context from {context_file}")
        with open(context_file, "r") as f:
            context_json = f.read()

    context = Context(context_json)

    if command.reset:
        logging.info("Resetting the context.")
        context.reset()
    
    if not command.log and len(prompts) > 0:
        context.add_text(Role.USER, prompts)
        response = llm.generate_response(context)
        context.add_text(Role.MODEL, [response])
        print(response)
    else:
        if not command.reset:
            logging.warning("No prompt provided; skipping inference.")

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
        '-r', '--reset', action='store_true', help="Reset the context"
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


def lookup_secret(service_name: str, key_name: str):
    command = [
        "secret-tool",
        "lookup",
        service_name,
        key_name
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()

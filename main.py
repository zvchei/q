#!/usr/bin/env python3

import logging
import os
import tempfile

from core import collect_garbage, get_process_stime, parse_command_line, execute_command
from gemini import Gemini
from nvidia import NvidiaNim
from pathlib import Path


def main():
    command, prompts = parse_command_line()

    log_level = logging.DEBUG if command.debug else logging.WARNING
    logging.basicConfig(level=log_level)

    collect_garbage()
    
    ppid = os.getppid()
    stime = get_process_stime(ppid)

    if stime is None:
        raise RuntimeError("Could not get process start time")

    temp_dir = Path(tempfile.gettempdir())
    context_file = temp_dir / f"q_context_{ppid}_{stime}.json"

    llm = Gemini("gemini-2.0-flash")
    # llm = NvidiaNim("meta/llama-4-maverick-17b-128e-instruct")

    execute_command(context_file, command, prompts, llm)


if __name__ == "__main__":
    main()

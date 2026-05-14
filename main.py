#!/usr/bin/env python3

import logging

from core import parse_command_line, execute_command
from gemini import Gemini
from iteration import Iteration
from nvidia import NvidiaNim
from session import collect_garbage, get_context_file
from tools import tools
from dice import DiceTool
from console import ConsoleCommandTool


model = 'gemini'


def main():
	command, prompts = parse_command_line()

	log_level = logging.DEBUG if command.debug else logging.WARNING
	logging.basicConfig(level=log_level)

	collect_garbage()

	# Register tools
	tools.register('dice', DiceTool)
	tools.register('console', ConsoleCommandTool)

	context_file = get_context_file()

	gemini = Gemini("gemini-2.0-flash")
	nvidia = NvidiaNim("meta/llama-4-maverick-17b-128e-instruct")

	llm = nvidia if model == 'nvidia' else gemini
	it = Iteration(llm, tools)

	execute_command(context_file, command, prompts, it)


if __name__ == "__main__":
	main()

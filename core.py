#!/usr/bin/env python3

import json
import logging
import os
import subprocess
import tempfile
import urllib.error
import urllib.request

from argparse import Namespace
from context import Context, Message, Part, Request, Result, Role
from iteration import Iteration
from pathlib import Path
from typing import Any, Callable, Optional


class FetchError(Exception):
	def __init__(self, message: str, code: Optional[int] = None):
		super().__init__(message)
		if code is not None:
			self.code = code


type Fetch = Callable[[str, str, dict[str, str]], str]


def fetch(url: str, data: Any, headers: dict[str, str]) -> Any:
	# TODO: Run debug logging only if enabled, to avoid wasting cycles
	logging.debug(f"Request URL: {url}")
	logging.debug(f"Request Headers: {json.dumps(headers, indent=2)}")
	logging.debug(f"Request Data: {json.dumps(data, indent=2)}")

	request = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
	try:
		with urllib.request.urlopen(request) as response:
			body = response.read().decode('utf-8')

			# TODO: Run debug logging only if enabled, to avoid wasting cycles
			logging.debug(f"Response Status: {response.status}")
			logging.debug(f"Response Body: {json.dumps(json.loads(body), indent=2)}")

			return json.loads(body)
	except urllib.error.HTTPError as e:
		try:
			error_body = e.read().decode('utf-8', errors='replace')
		except Exception:
			error_body = str(e)
		raise FetchError(error_body, code=e.code)
	except urllib.error.URLError as e:
		raise FetchError(str(e))


# TODO: Use an abstract class to avoid the need to provide type parameters
def execute_command(context_file: Path, command: Namespace, prompts: list[str], it: Iteration[Any, Any]) -> None:
	context_json = ''

	if context_file.exists():
		logging.info(f'Loading existing context from {context_file}')
		with open(context_file, 'r') as f:
			context_json = f.read()

	context = Context(context_json)

	if command.reset:
		logging.info('Resetting the context.')
		context.reset()

	if command.log:
		if command.parse:
			native_context = it.model.prepare_context(context)  # type: ignore[attr-defined]
			print(json.dumps(native_context, indent=2))
		else:
			print(context.to_json())

	elif len(prompts) > 0:
		def debug_print_response(role: Role, part: Part) -> None:
			role_str = {
				Role.SYSTEM: 'SYSTEM',
				Role.USER: 'USER',
				Role.MODEL: 'MODEL',
				Role.TOOL: 'TOOL'
			}.get(role, 'UNKNOWN')

			if isinstance(part, Message):
				print(f'[{role_str}] {part.text}')
			elif isinstance(part, Request):
				print(f"[{role_str}:{part.id}] {part.name}({', '.join(part.arguments)})")
			elif isinstance(part, Result):
				print(f"[{role_str}:{part.id}] {part.name} => {part.result}")
			else:
				logging.warning(f'Unknown part type: {part}')
				print(f'[{role_str}] {part}')

		def print_response(role: Role, part: Part) -> None:
			if role == Role.MODEL and isinstance(part, Message):
				print(part.text)

		context.add_text(Role.USER, prompts)
		output = debug_print_response if logging.getLogger().isEnabledFor(logging.DEBUG) else print_response
		context = it.execute(context, output, command.tools)

	elif not command.reset:
		# no input, return last response
		last_response = context.get_last_response()
		if last_response:
			print(last_response)
		else:
			logging.warning('No previous response found in context.')
	else:
		logging.info('No prompt provided. Skipping inference.')

	# Save the updated context
	with open(context_file, 'w') as f:
		f.write(context.to_json())

def parse_command_line():
	import sys
	import argparse

	parser = argparse.ArgumentParser(description='Ask the LLM oracle.')
	parser.add_argument(
		'-l', '--log', action='store_true', help='Dump the context'
	)
	parser.add_argument(
		'-p', '--parse', action='store_true', help='When used with --log, outputs parsed context via backend prepare_context'
	)
	parser.add_argument(
		'-r', '--reset', action='store_true', help='Reset the context'
	)
	parser.add_argument(
		'-d', '--debug', action='store_true', help='Enable debug logging'
	)
	parser.add_argument(
		'-t', '--tools', action='append', metavar='TOOL', help='Enable tools mode and specify tool(s) to use. Can be used multiple times.'
	)
	parser.add_argument(
		'inputs', nargs='*', help='Prompt input for the oracle.'
	)

	args = parser.parse_args()
	prompt = ' '.join(args.inputs)

	# Include additional input from stdin if available, to extend the prompt
	extra_prompt = ''
	if not sys.stdin.isatty():
		extra_prompt = str(sys.stdin.read()).strip()

	# Remove the empty prompts
	prompts: list[str] = [p for p in [prompt, extra_prompt] if p]

	return args, prompts


def get_process_stime(pid: int) -> Optional[int]:
	try:
		with open(f'/proc/{pid}/stat') as f:
			stats = f.read().split()
			# The start time (stime) is the 22nd field (index 21).
			# It's measured in clock ticks since system boot.
			return int(stats[21])
	except (FileNotFoundError):
		return None


def collect_garbage():
	temp_dir = tempfile.gettempdir()
	for filename in os.listdir(temp_dir):
		if not filename.startswith('q_context_') or not filename.endswith('.json'):
			continue

		try:
			parts = filename[len('q_context_') : -len('.json')].split('_')
			if len(parts) != 2:
				raise ValueError('Unexpected filename format')

			pid = int(parts[0])
			stime = int(parts[1])
			current_stime = get_process_stime(pid)

			if current_stime is None or current_stime != stime:
				filepath = os.path.join(temp_dir, filename)
				os.remove(filepath)
				logging.info(f'Removed stale context file: {filename}')
			else:
				logging.debug(f'Found active context file: {filename}; skipping deletion.')
		except ValueError:
			logging.warning(f'Skipping file with unexpected name format: {filename}')
		except OSError as e:
			logging.warning(f'Error removing file {filename}: {e}')


def lookup_secret(service_name: str, key_name: str):
	command = [
		'secret-tool',
		'lookup',
		service_name,
		key_name
	]
	result = subprocess.run(command, capture_output=True, text=True, check=True)
	return result.stdout.strip()

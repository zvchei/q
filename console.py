import os
import shlex
import subprocess
import sys

from tools import JsonValue, Tool, ToolDefinition


color = sys.stdout.isatty() and os.environ.get('TERM') not in (None, '', 'dumb')

RED = '\033[91m' if color else '[CONSOLE:ERROR]'
BLUE = '\033[94m' if color else '[CONSOLE:INFO]'
YELLOW = '\033[93m' if color else '[CONSOLE:ALERT]'
RESET = '\033[0m' if color else ''


class ConsoleCommandTool(Tool):
	"""Safely execute a shell command after explicit user confirmation.

	Requests the user to approve the command before running. If approved,
	executes the command and returns structured results including stdout,
	stderr and exit status.
	"""

	MAX_TIMEOUT = 60
	DESCRIPTION = 'Execute a shell command after user confirmation (y/n). Returns stdout, stderr, and exit code.'
	INSTRUCTIONS = 'When the solving user\'s request requires executing a shell command, use the "command" tool immediately, without looking for specific instruction to do so.'

	@staticmethod
	def definition():
		return ToolDefinition(
			instructions=ConsoleCommandTool.INSTRUCTIONS,
			description=ConsoleCommandTool.DESCRIPTION,
			parameters={
				'type': 'object',
				'properties': {
					'command': {
						'type': 'string',
						'description': 'Shell command to execute (will require user confirmation).'
					}
				},
				'required': ['command']
			}
		)

	def execute(self, command: str) -> JsonValue:
		command = command.strip()
		if not command:
			return {'approved': False, 'error': 'Empty command provided'}

		print(f'{BLUE}Requested command:{RESET}')
		print(f'\n{command}\n')
		try:
			answer = input(f'{BLUE}Execute this command? (y/n): {RESET}').strip().lower()
			print()
		except EOFError:
			return {'approved': False, 'error': 'No interactive input available for confirmation'}

		if answer not in ('y', 'yes'):
			print(f'{YELLOW}Command execution denied by user.{RESET}')
			return {
				'approved': False,
				'command': command,
				'error': 'User denied execution'
			}

		shell = True
		args = command
		try:
			if not any(op in command for op in ['|', '&&', '||', ';', '>', '<', '*', '$', '`']):
				parts = shlex.split(command)
				if parts:
					shell = False
					args = parts
		except ValueError:
			pass

		try:
			completed = subprocess.run(
				args,
				shell=shell,
				capture_output=True,
				text=True,
				timeout=self.MAX_TIMEOUT
			)
		except subprocess.TimeoutExpired:
			print(f'{RED}Command timed out ({self.MAX_TIMEOUT}s).{RESET}')
			return {
				'approved': True,
				'command': command,
				'timeout': True,
				'error': f'Command timed out after {self.MAX_TIMEOUT} seconds'
			}
		except Exception as e:
			print(f'{RED}Command execution failed: {e}{RESET}')
			return {
				'approved': True,
				'command': command,
				'error': f'Execution failed: {e}'
			}

		result: JsonValue = {
			'approved': True,
			'command': command,
			'returncode': completed.returncode,
			'stdout': completed.stdout,
			'stderr': completed.stderr
		}

		return result

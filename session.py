import logging
import os
import tempfile

from pathlib import Path
from typing import Optional


_CONTEXT_FILE_PREFIX = 'q_context_'
_CONTEXT_FILE_SUFFIX = '.json'


def get_process_stime(pid: int) -> Optional[int]:
	try:
		with open(f'/proc/{pid}/stat') as f:
			stats = f.read().split()
			# The start time is the 22nd field (index 21),
			# measured in clock ticks since system boot.
			return int(stats[21])
	except FileNotFoundError:
		return None


def collect_garbage():
	temp_dir = tempfile.gettempdir()
	for filename in os.listdir(temp_dir):
		if not filename.startswith(_CONTEXT_FILE_PREFIX) or not filename.endswith(_CONTEXT_FILE_SUFFIX):
			continue

		try:
			parts = filename[len(_CONTEXT_FILE_PREFIX) : -len(_CONTEXT_FILE_SUFFIX)].split('_')
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


def get_context_file() -> Path:
	'''Return the context file path for the current shell session.'''
	ppid = os.getppid()
	stime = get_process_stime(ppid)

	if stime is None:
		raise RuntimeError("Could not get process start time")

	temp_dir = Path(tempfile.gettempdir())
	return temp_dir / f"{_CONTEXT_FILE_PREFIX}{ppid}_{stime}{_CONTEXT_FILE_SUFFIX}"

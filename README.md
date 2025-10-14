# Q

Minimal CLI wrapper around a configured[^1] LLM service to ask quick one‑off questions while keeping an incremental conversation context per shell session.

[^1]: Currently supports only Google Gemini via their REST API.

## Features
- Context is preserved between runs within the lifetime of the parent shell process.
- Simple JSON context log (`-l`) for inspection or tooling.
- Automatic garbage collection of stale context files from previous runs.
- No external Python dependencies; only the standard library + the system `secret-tool` (GNOME Keyring / libsecret).
- Supports tool use (e.g. `dice` throwing tool) via LLM's function calling.
- Input prompt can be provided as a command line argument or piped from a file or both.

## Install
1. Ensure Python 3.11+ (tested on 3.12) and `secret-tool` (part of `libsecret` / GNOME Keyring) are available.
2. Clone repo and make script executable:
	```bash
	git clone https://github.com/zvchei/q.git
	cd q
	chmod +x main.py

	# Recommended: symlink into PATH
	ln -s "$PWD/main.py" ~/.local/bin/q
	```
3. Store your Gemini API key in the desktop secret store:
	```bash
	secret-tool store --label "Gemini API Key" gemini api-key
	# When asked, paste the key and press Enter
	```

## Usage
Basic question:
```bash
q What is the capital of France?
```
Show the help message:
```bash
q -h
```
If the prompt contains special characters it should be quoted, or the characters escaped, e.g.:
```bash
q "What does T&C stand for?"
```
Pipe text input:
```bash
echo "Explain TCP slow start" | q
```
Combine piped and argument input:
```bash
lscpu | q "How many cores do I have?"
```
Use it in an interactive way:
```bash
$ q "How to tell what is the CUDA version on my system?"

nvidia-smi

$ nvidia-smi | q

12.8
```
Redirect stderr from a failing command to get an explanation:
```bash
cat non-existent-file 2>&1 | q
```
Show (dump) current context without querying:
```bash
q -l
```
Reset context (clears prior conversation):
```bash
q -r
```
Or reset the context and run a new prompt in a single command:
```bash
q -r "Start fresh: give me 3 bullet ideas for a dev blog post"
```
Enable debug logging (file operations, GC):
```bash
q --debug "Why is the sky blue?"
```
Run a tool:
```bash
q -t dice "Roll one 12-sided die and two 6-sided dice"
```

## Available Tools

### `dice`:

A tool for rolling dice.

#### Examples:
```bash
q -t dice "Roll one 12-sided die and two 6-sided dice"
q -t dice "Roll 5 4-sided dice"
q -t dice "2d6, 2d8"
q -t dice "Roll a die"
```

### `console`
A tool for executing shell commands.

#### Examples:
```bash
q -t console "ls"
q -t console "List all files modified in the last hour"
q -t console "What is my public IP address?"
q -t console "Create a cron job that runs backup.sh every day at midnight."
q -t console "Find the most frequent IP address in access.log"
```

## Output
By default prints only the model answer. With `-l` prints full JSON context to stdout (after any new inference if a prompt was provided).

## Context Storage
- Files are created in the system temp directory (e.g. `/tmp`) named: `q_context_<parent_shell_pid>_<parent_shell_starttime>.json`.
- Start time (from `/proc/<pid>/stat`) ensures uniqueness across reused PIDs after shell restarts.
- Garbage collection removes any context file whose originating shell process no longer exists.
- Resetting (`-r`) clears the context for the current shell both in-memory and on-disk.

## Exit Codes
- 0 on success.
- Non‑zero if the LLM API returns HTTP error (propagated) or local runtime errors occur.

## Security / Privacy
- API key is retrieved at runtime from the local secret storage (never written to disk by the tool).
- Context files contain your prompts & model replies in plain JSON. Avoid placing sensitive information in prompts.

## Changing the Model
Edit `main.py` and change the `model` variable at the `llm` construction line to your desired model name (e.g. `llm = Gemini('gemini-1.5-pro')`).

## License
MIT-0 (see `LICENSE`).

## Roadmap Ideas (Not Implemented)
- Context pruning strategy when approaching context limits.
- Support for image/file parts beyond plain text.
- Configurable system prompt and model.
- Support more LLM backends.
- Support other operating systems by using alternative credentials providers.

## Troubleshooting
- Missing key: ensure `secret-tool lookup gemini api-key` returns your key.
- Permission denied: verify `main.py` is executable and/or symlink path in `$PATH`.
- Run `q --debug` to see debug logs.

Enjoy concise terminal Q&A with persistent lightweight context.

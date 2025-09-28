import logging
from typing import Any, Callable, List, Mapping, Sequence, cast
from context import Context, Entry, Message, Part, Request, Result, Role
from core import Fetch, fetch, lookup_secret
from iteration import LLMBackend
from tools import JsonValue, ToolDefinition


# TODO: This class need major rework to adhere to the project's architecture and coding standards


# TODO: Define strict types for Nvidia NIM's JSON structures
class NvidiaNim(LLMBackend[Any, Any]):
	def __init__(self, model: str, fetch: Fetch = fetch, lookup_secret: Callable[[str, str], str] = lookup_secret):
		self.max_context_length = 1048576
		self.model = model
		self.api_key = lookup_secret('nvidia-nim', 'api-key')
		self.fetch = fetch

	def generate_response(self, context: Any) -> Any:
		url = "https://integrate.api.nvidia.com/v1/chat/completions"
		headers = {
			"Content-Type": "application/json",
			"Authorization": f"Bearer {self.api_key}",
			"Accept": "application/json"
		}
		return self.fetch(url, context, headers)

	def prepare_context(self, context: Context, tools: Mapping[str, ToolDefinition] = {}) -> Any:
		messages = [self._prepare_entry(entry) for entry in context]
		content: Mapping[str, Any] = {
			"model": self.model,
			"messages": messages,
			"max_tokens": 512,
			"temperature": 1.00,
			"top_p": 1.00,
			"frequency_penalty": 0.00,
			"presence_penalty": 0.00,
			"stream": False
		}
		if tools:
			tool_definitions: List[dict[str, Any]] = []
			for name, definition in tools.items():
				tool_definitions.append({
					"type": "function",
					"function": {
						"name": name,
						"description": definition.description,
						"parameters": definition.parameters
					}
				})
			content["tools"] = tool_definitions
		return content

	def _prepare_entry(self, entry: Entry) -> Any:
		role = {
			Role.SYSTEM: "system",
			Role.USER: "user",
			Role.MODEL: "assistant",
			Role.TOOL: "tool"
		}.get(entry.role, "user")
		# OpenAI/Nvidia NIM expects 'content' to be a string or array of objects
		if entry.role == Role.TOOL:
			part = entry.parts[0] if len(entry.parts) > 0 else None
			if isinstance(part, Result):
				content = str(part.result)
			else:
				content = ""
		elif len(entry.parts) == 1 and isinstance(entry.parts[0], Message):
			content = entry.parts[0].text
		else:
			content = [self._prepare_part(p) for p in entry.parts]
		return {"role": role, "content": content}

	def _prepare_part(self, part: Part) -> Any:
		# For Nvidia NIM, only text parts are supported in messages
		if isinstance(part, Message):
			return {"text": part.text}
		elif isinstance(part, Result):
			return {"text": str(part.result)}
		elif isinstance(part, Request):
			return {"text": str(part.arguments)}
		else:
			logging.warning(f'Unknown part type: {part}')
			return {"text": ""}

	def parse_result(self, result: Mapping[str, Any]) -> Sequence[Entry]:
		choices: List[dict[str, Any]] = (
			cast(List[dict[str, Any]], result.get("choices", [{}]))
			if isinstance(result, dict)
			else [{}]
		)

		choice = choices[0]
		message = cast(dict[str, Any], choice.get("message", {}))
		content = str(message.get("content", ""))
		role_str = str(message.get("role", "assistant"))
		role = {
			"system": Role.SYSTEM,
			"user": Role.USER,
			"assistant": Role.MODEL,
			"tool": Role.TOOL
		}.get(role_str, Role.MODEL)
		parts: list[Any] = []
		if content:
			parts.append(Message(text=content))
		# Handle tool calls
		if "tool_calls" in message:
			tool_calls = cast(list[dict[str, Any]], message.get("tool_calls", []))
			if not tool_calls:
				tool_calls = []
			for tool_call in tool_calls:
				if str(tool_call.get("type", "")) == "function":
					func_call = cast(dict[str, Any], tool_call.get("function", {}))
					if not func_call:
						func_call = {}
					args: Mapping[str, JsonValue] = func_call.get("arguments", "{}")
					if isinstance(args, str):
						import json
						try:
							args = json.loads(args)
						except Exception:
							args = {}
					name = str(func_call.get("name", ""))
					parts.append(Request(id="", name=name, arguments=args))
		return [Entry(role=role, parts=parts)]

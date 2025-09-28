import logging

from context import Context, Message, Part, Request, Result, Role, Entry
from core import Fetch, fetch, lookup_secret
from iteration import LLMBackend
from tools import ToolDefinition
from typing import Any, Callable, Mapping, Sequence


# TODO: Define strict types for Gemini's JSON structures
class Gemini(LLMBackend[Any, Any]):
	def __init__(self, model: str, fetch: Fetch = fetch, lookup_secret: Callable[[str, str], str] = lookup_secret):
		self.max_context_length = 1048576
		self.model = model
		self.api_key = lookup_secret('gemini', 'api-key')
		self.fetch = fetch

	def generate_response(self, context: Any) -> Any:
		url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

		headers = {
			'Content-Type': 'application/json',
			'x-goog-api-key': self.api_key
		}

		return self.fetch(url, context, headers)

	def prepare_context(self, context: Context, tools: Mapping[str, ToolDefinition] = {}) -> Any:
		content: Mapping[str, Any] = {
			"system_instruction": {
				"parts": []
			},
			"contents": []
		}

		system_prompt_extensions = []

		if tools:
			content['tools'] = {
				'functionDeclarations': [
					{
						"name": name,
						"description": definition.description,
						"parameters": definition.parameters
					}
					for name, definition in tools.items()
				]
			}

			system_prompt_extensions = [
				{'text': definition.instructions}
				for definition
				in tools.values()
				if definition.instructions
			]

		for entry in context:
			if entry.role == Role.SYSTEM:
				parts = [self._prepare_part(p) for p in entry.parts]
				content['system_instruction']['parts'].extend(parts)
				content['system_instruction']['parts'].extend(system_prompt_extensions)
			else:
				content['contents'].append(self._prepare_entry(entry))

		return content

	def _prepare_entry(self, entry: Entry) -> Any:
		role = {
			Role.USER: "user",
			Role.MODEL: "model",
			Role.TOOL: "user"  # Gemini uses "user" for tool results
		}.get(entry.role, "user")

		parts = [self._prepare_part(p) for p in entry.parts]
		return {
			"role": role,
			"parts": parts
		}

	def _prepare_part(self, part: Part) -> Any:
		if isinstance(part, Message):
			return {
				"text": part.text
			}
		elif isinstance(part, Request):
			return {
				"functionCall": {
					"id": part.id,
					"name": part.name,
					"args": part.arguments
				}
			}
		elif isinstance(part, Result):
			return {
				"functionResponse": {
					"id": part.id,
					"name": part.name,
					"response": part.result
				}
			}
		else:
			logging.warning(f'Unknown part type: {part}')
			return {
				"text": ''
			}

	def parse_result(self, result: Any) -> Sequence[Entry]:
		candidates = result.get("candidates", [{}])
		candidate = candidates[0]
		content = candidate.get("content", {})

		if not content:
			logging.error("No content in Gemini\'s response")
			raise ValueError("Invalid response from Gemini")

		role = {
			"user": Role.USER,
			"model": Role.MODEL,
		}.get(content.get("role", "model"), Role.MODEL)

		parts = [self._parse_part(p) for p in content.get("parts", [])]

		return [
			Entry(
				role=role,
				parts=parts
			)
		]

	def _parse_message(self, text: Any) -> Message:
		if not text:
			logging.error(f'Message from Gemini is missing text')
			raise ValueError('Message part missing text')
		return Message(text=text)

	def _parse_request(self, part_data: Mapping[str, Any]) -> Request:
		name = part_data.get("name")
		arguments = part_data.get("args", {})

		if not name:
			logging.error(f'Request part missing a name: {part_data}')
			raise ValueError('Request part missing a name')

		return Request(id='', name=name, arguments=arguments)

	def _parse_result(self, part_data: Mapping[str, Any]) -> Result:
		name = part_data.get("name")
		response = part_data.get("response")

		if not name:
			logging.error(f'Result part missing a name: {part_data}')
			raise ValueError('Result part missing a name')

		return Result(id='', name=name, result=response)

	def _parse_part(self, part_data: Mapping[str, Any]) -> Part:
		part_type = next(iter(part_data), None)
		cls_map: dict[str, Callable[[Mapping[str, Any]], Part]] = {
			"text": self._parse_message,
			"functionCall": self._parse_request,
			"functionResponse": self._parse_result
		}

		if part_type in cls_map:
			return cls_map[part_type](part_data[part_type])
		else:
			logging.error(f'Unknown part type: {part_data}')
			raise ValueError(f'Unknown part type: {part_data}')

from abc import ABC, abstractmethod
import logging
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Type


JsonValue = str | int | float | bool | None | Mapping[str, 'JsonValue'] | Sequence['JsonValue']


class ToolDefinition(NamedTuple):
	description: str
	parameters: dict[str, JsonValue]
	instructions: Optional[str] = None


class Tool(ABC):
	@staticmethod
	@abstractmethod
	def definition() -> ToolDefinition:
		raise NotImplementedError()

	@abstractmethod
	def execute(self, *args: Any, **kwargs: Any) -> JsonValue:
		raise NotImplementedError()


# TODO: Add instantiation strategies for tools (singleton, per-use, etc.)
# TODO: Protect the tool registry against modification at runtime

class ToolRegistry(dict[str, Type[Tool]]):
	def register(self, name: str, tool_cls: Type[Tool]) -> None:
		self[name] = tool_cls
		logging.info(f'Registered tool: {name} -> {tool_cls}')

tools = ToolRegistry()

from abc import ABC, abstractmethod
import logging
from context import Context, Entry, Part, Request, Result, Role
from tools import ToolDefinition, ToolRegistry
from typing import Callable, Generic, Mapping, Sequence, TypeVar


TResult = TypeVar('TResult')
TContext = TypeVar('TContext')


class LLMBackend(ABC, Generic[TResult, TContext]):
	@abstractmethod
	def generate_response(self, context: TContext) -> TResult:
		raise NotImplementedError()

	@abstractmethod
	def prepare_context(self, context: Context, tools: Mapping[str, ToolDefinition] = {}) -> TContext:
		raise NotImplementedError()

	@abstractmethod
	def parse_result(self, result: TResult) -> Sequence[Entry]:
		raise NotImplementedError()


class Iteration(Generic[TResult, TContext]):
	def __init__(self, model: LLMBackend[TResult, TContext], tool_registry: ToolRegistry):
		self.model = model
		self.tool_registry = tool_registry

	def execute(self, context: Context, output: Callable[[Role, Part], None], tools: Sequence[str] | None) -> Context:
		# Convert the context to the format required by the model:
		def check_tool(tool_name: str) -> bool:
			found = tool_name in self.tool_registry
			if not found:
				logging.warning(f"Tool not found: {tool_name}")
			return found

		tool_definitions = {
			tool: self.tool_registry[tool].definition()
			for tool in tools or []
			if check_tool(tool)
		}

		prompt = self.model.prepare_context(context, tool_definitions)

		# Generate the response from the model:
		result = self.model.generate_response(prompt)

		# Extract the response from the result:
		entries = self.model.parse_result(result)

		# Update the context with the new parts:
		context.extend(entries)

		# Run the tools requested by the model:
		requests = [
			part
			for entry in entries
			for part in entry.parts
			if isinstance(part, Request)
		]

		results = [
			Result(
				id=request.id,
				name=request.name,
				result=self.tool_registry[request.name]().execute(**request.arguments)
			)
			for request in requests
			if check_tool(request.name)
		]

		for entry in entries:
			for part in entry.parts:
				output(entry.role, part)

		if results:
			context.add_results(results)

			for result in results:
				output(Role.TOOL, result)

			# TODO: Prevent infinite recursion here. Only recurse if new requests are present, or limit recursion depth.
			# TODO: Currently, just one tool round is requested and executed. Find a way to combine tool calls in multiple steps.
			self.execute(context, output, tools)

		return context

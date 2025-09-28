import logging
from typing import Sequence, cast
import unittest
from context import Context, Entry, Request, Result, Role, Message
from tools import Tool, ToolDefinition, ToolRegistry
from typing import Mapping
from iteration import Iteration, LLMBackend


class DummyTool(Tool):
	@staticmethod
	def definition():
		return ToolDefinition(
			description="Dummy tool for testing.",
			parameters={"x": "int"},
			instructions="Increment x by 1."
		)

	@property
	def name(self):
		return 'dummy_tool'

	def execute(self, x: int) -> int:
		return x + 1


class DummyBackend(LLMBackend[str, Context]):
	def generate_response(self, context: Context) -> str:
		return "dummy_result"

	def prepare_context(self, context: Context, tools: 'Mapping[str, ToolDefinition]' = {}) -> Context:
		return context

	def __init__(self):
		self._responses = [
			[Entry(
				role=Role.MODEL,
				parts=[Request(id="random_id", name="dummy_tool", arguments={"x": 1})]
			)],
			[Entry(
				role=Role.MODEL,
				parts=[Message(text="Tool execution complete and result received.")]
			)]
		]

	def parse_result(self, result: str) -> Sequence[Entry]:
		if self._responses:
			return self._responses.pop(0)
		raise RuntimeError("DummyBackend.parse_result was not supposed to be called more than twice.")


class TestIteration(unittest.TestCase):
	def setUp(self):
		logging.getLogger().setLevel(logging.ERROR)

	def tearDown(self):
		logging.getLogger().setLevel(logging.WARNING)

	def test_execute(self):
		backend = DummyBackend()
		tool_registry = ToolRegistry()
		tool_registry.register('dummy_tool', DummyTool)
		context = Context("")
		context.add_text(Role.USER, ["Call the dummy tool with x=1, please."])
		iteration = Iteration(backend, tool_registry)
		def output(role: Role, part: object) -> None:
			pass
		updated_context = iteration.execute(context, output, ['dummy_tool'])

		self.assertIsInstance(updated_context, Context)
		self.assertEqual(len(updated_context), 5)

		self.assertEqual(updated_context[2].role, Role.MODEL)
		self.assertIsInstance(updated_context[2].parts[0], Request)
		request_part = cast(Request, updated_context[2].parts[0])
		self.assertEqual(request_part.id, "random_id")
		self.assertEqual(request_part.name, "dummy_tool")
		self.assertEqual(request_part.arguments, {"x": 1})

		self.assertEqual(updated_context[3].role, Role.TOOL)
		self.assertIsInstance(updated_context[3].parts[0], Result)
		result_part = cast(Result, updated_context[3].parts[0])
		self.assertEqual(result_part.id, "random_id")
		self.assertEqual(result_part.name, "dummy_tool")
		self.assertEqual(result_part.result, 2)

	def test_tool_not_found(self):
		backend = DummyBackend()
		tool_registry = ToolRegistry()  # No tools registered
		context = Context("")
		context.add_text(Role.USER, ["Call the dummy tool with x=1, please."])
		iteration = Iteration(backend, tool_registry)
		def output(role: Role, part: object) -> None:
			pass
		updated_context = iteration.execute(context, output, ['dummy_tool'])  # Tool not registered

		self.assertIsInstance(updated_context, Context)
		self.assertEqual(len(updated_context), 3)  # No tool execution happened

		self.assertEqual(updated_context[2].role, Role.MODEL)
		self.assertIsInstance(updated_context[2].parts[0], Request)
		request_part = cast(Request, updated_context[2].parts[0])
		self.assertEqual(request_part.id, "random_id")
		self.assertEqual(request_part.name, "dummy_tool")
		self.assertEqual(request_part.arguments, {"x": 1})
		# No TOOL entry should be present since the tool was not found
		self.assertTrue(all(entry.role != Role.TOOL for entry in updated_context))

	def test_output_function(self):
		backend = DummyBackend()
		tool_registry = ToolRegistry()
		tool_registry.register('dummy_tool', DummyTool)
		context = Context("")
		context.add_text(Role.USER, ["Call the dummy tool with x=1, please."])
		iteration = Iteration(backend, tool_registry)
		outputs: list[tuple[Role, object]] = []
		def output(role: Role, part: object) -> None:
			outputs.append((role, part))

		iteration.execute(context, output, ['dummy_tool'])

		self.assertEqual(len(outputs), 3)  # MODEL Request, TOOL Result, MODEL Message
		self.assertEqual(outputs[0][0], Role.MODEL)
		self.assertIsInstance(outputs[0][1], Request)
		self.assertEqual(cast(Request, outputs[0][1]).id, "random_id")
		self.assertEqual(cast(Request, outputs[0][1]).name, "dummy_tool")
		self.assertEqual(cast(Request, outputs[0][1]).arguments, {"x": 1})

		self.assertEqual(outputs[1][0], Role.TOOL)
		self.assertIsInstance(outputs[1][1], Result)
		self.assertEqual(cast(Result, outputs[1][1]).id, "random_id")
		self.assertEqual(cast(Result, outputs[1][1]).name, "dummy_tool")
		self.assertEqual(cast(Result, outputs[1][1]).result, 2)

		self.assertEqual(outputs[2][0], Role.MODEL)
		self.assertIsInstance(outputs[2][1], Message)
		self.assertEqual(cast(Message, outputs[2][1]).text, "Tool execution complete and result received.")

if __name__ == "__main__":
	unittest.main()

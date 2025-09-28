import unittest
from tools import Tool, tools, JsonValue, ToolDefinition

class UnregisteredTool(Tool):
	@staticmethod
	def definition() -> ToolDefinition:
		return ToolDefinition(
			description="An unregistered tool.",
			parameters={"type": "object", "properties": {}}
		)
	def execute(self, *args: object, **kwargs: object) -> JsonValue:
		return "unregistered_tool executed"

class RegisteredTool(Tool):
	@staticmethod
	def definition() -> ToolDefinition:
		return ToolDefinition(
			description="A registered tool.",
			parameters={"type": "object", "properties": {}}
		)
	def execute(self, *args: object, **kwargs: object) -> JsonValue:
		return "registered_tool executed"

tools.register('registered_tool', RegisteredTool)

class TestToolRegistry(unittest.TestCase):
	def test_manual_tool_not_registered(self):
		for tool in tools.values():
			self.assertNotIsInstance(tool(), UnregisteredTool)

	def test_decorator_tool_registered(self):
		self.assertIn('registered_tool', tools)
		self.assertIsInstance(tools['registered_tool'](), RegisteredTool)

if __name__ == "__main__":
	unittest.main()

import gemini
import unittest

from context import Message, Request, Result, Context, Role, Entry
from tools import ToolDefinition
from unittest.mock import Mock


class TestGeminiGenerateResponse(unittest.TestCase):
	def setUp(self):
		self.mock_fetch = Mock(return_value="mocked-response")
		self.mock_lookup_secret = Mock(return_value="dummy-key")

		self.backend = gemini.Gemini(
			"test-model",
			fetch=self.mock_fetch,
			lookup_secret=self.mock_lookup_secret
		)

	def test_prepare_context(self):
		ctx = Context("")
		ctx.extend([
			Entry(role=Role.USER, parts=[Message(text="hello from user")]),
			Entry(role=Role.MODEL, parts=[Message(text="hello from assistant")]),
			Entry(role=Role.MODEL, parts=[Request(id="1", name="tool_call", arguments={"arg1": "value1"})]),
			Entry(role=Role.TOOL, parts=[Result(id="1", name="tool_result", result={"data": 123})])
		])

		# No tools
		result = self.backend.prepare_context(ctx)
		self.assertIn("system_instruction", result)
		self.assertIn("contents", result)
		self.assertEqual(result["system_instruction"]["parts"][0]["text"], "Answer the following queries directly and provide the single best answer concisely without any quotes.")
		self.assertEqual(result["contents"][0]["role"], "user")
		self.assertEqual(result["contents"][0]["parts"][0]["text"], "hello from user")
		self.assertEqual(result["contents"][1]["role"], "model")
		self.assertEqual(result["contents"][1]["parts"][0]["text"], "hello from assistant")
		self.assertEqual(result["contents"][2]["role"], "model")
		self.assertEqual(result["contents"][2]["parts"][0]["functionCall"]["name"], "tool_call")
		self.assertEqual(result["contents"][2]["parts"][0]["functionCall"]["args"], {"arg1": "value1"})
		self.assertEqual(result["contents"][3]["role"], "user")
		self.assertEqual(result["contents"][3]["parts"][0]["functionResponse"]["name"], "tool_result")
		self.assertEqual(result["contents"][3]["parts"][0]["functionResponse"]["response"], {"data": 123})

		# With tools
		tools = {
			"foo": ToolDefinition(
				description="tool's description",
				parameters={"type": "object"},
				instructions="do foo"
			),
			"bar": ToolDefinition(
				description="another tool",
				parameters={"type": "object", "additionalProperties": {"type": "string"}}
			)
		}
		result_tools = self.backend.prepare_context(ctx, tools)
		self.assertIn("tools", result_tools)
		self.assertIn("functionDeclarations", result_tools["tools"])
		self.assertEqual(result_tools["tools"]["functionDeclarations"][0]["description"], "tool's description")
		self.assertEqual(result_tools["tools"]["functionDeclarations"][0]["parameters"], {"type": "object"})
		self.assertEqual(result_tools["tools"]["functionDeclarations"][0]["name"], "foo")
		self.assertEqual(result_tools["tools"]["functionDeclarations"][1]["description"], "another tool")
		self.assertEqual(result_tools["tools"]["functionDeclarations"][1]["parameters"], {"type": "object", "additionalProperties": {"type": "string"}})
		self.assertEqual(result_tools["tools"]["functionDeclarations"][1]["name"], "bar")
		self.assertEqual(result_tools["system_instruction"]["parts"][1]["text"], "do foo")
		self.assertEqual(len(result_tools["system_instruction"]["parts"]), 2) # Tool 'bar' has no instructions, so only 2 parts total

	def test_generate_response(self):
		context = {"foo": "bar"}
		result = self.backend.generate_response(context)
		self.assertEqual(result, "mocked-response")
		self.mock_fetch.assert_called_once()
		args, _ = self.mock_fetch.call_args
		self.assertTrue(args[0].endswith("test-model:generateContent"))
		self.assertEqual(args[1], context)
		self.assertEqual(args[2]['x-goog-api-key'], "dummy-key")

	def test_lookup_secret(self):
		self.mock_lookup_secret.assert_called_once()
		self.mock_lookup_secret.assert_called_with('gemini', 'api-key')

	def test_parse_result(self):
		sample_response = {	# type: ignore
			"candidates": [
				{
					"content": {
						"parts": [
							{
								"text": "This is a sample response."
							},
							{
								"functionCall": {
									"name": "sample_tool",
									"args": {"param": "value"}
								}
							},
							{
								"functionResponse": {
									"name": "sample_tool",
									"response": {"result": 42}
								}
							}
						]
					}
				}
			]
		}
		result = self.backend.parse_result(sample_response)
		self.assertTrue(len(result) == 1)
		self.assertIsInstance(result[0], Entry)
		self.assertEqual(result[0].role, Role.MODEL)
		self.assertEqual(result[0].parts[0], Message("This is a sample response."))
		self.assertEqual(result[0].parts[1], Request("", "sample_tool", {"param": "value"}))
		self.assertEqual(result[0].parts[2], Result("", "sample_tool", {"result": 42}))

if __name__ == "__main__":
	unittest.main()

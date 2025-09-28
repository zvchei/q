import json
from typing import cast
import unittest
from context import Context, Entry, PartType, Role, Message, Request, Result

class TestContext(unittest.TestCase):
	def test_initialization(self):
		context = Context('')
		self.assertEqual(len(context), 1)
		self.assertEqual(context[0].role, Role.SYSTEM)
		message = context[0].parts[0]
		self.assertIsInstance(message, Message)
		message = cast(Message, message)
		self.assertEqual(message.text, 'Answer the following queries directly and provide the single best answer concisely without any quotes.')

	def test_initialization_with_json(self):
		json_str = '''
		[
			{
				"role": "user",
				"parts": [
					{
						"type": "text/plain",
						"text": "Hello, how are you?"
					}
				]
			},
			{
				"role": "model",
				"parts": [
					{
						"type": "text/plain",
						"text": "I'm fine, thank you!"
					}
				]
			},
			{
				"role": "model",
				"parts": [
					{
						"type": "application/x-request",
						"id": "1",
						"name": "test_tool",
						"arguments": {"param": "value"}
					}
				]
			},
			{
				"role": "tool",
				"parts": [
					{
						"type": "application/x-result",
						"id": "1",
						"name": "test_tool",
						"result": "ok"
					}
				]
			}
		]
		'''
		context = Context(json_str)
		self.assertEqual(len(context), 4)
		self.assertEqual(context[0].role, Role.USER)
		user_message = context[0].parts[0]
		self.assertIsInstance(user_message, Message)
		user_message = cast(Message, user_message)
		self.assertEqual(user_message.text, 'Hello, how are you?')
		self.assertEqual(context[1].role, Role.MODEL)
		model_message = context[1].parts[0]
		self.assertIsInstance(model_message, Message)
		model_message = cast(Message, model_message)
		self.assertEqual(model_message.text, 'I\'m fine, thank you!')
		self.assertEqual(context[2].role, Role.MODEL)
		tool_request = context[2].parts[0]
		self.assertIsInstance(tool_request, Request)
		tool_request = cast(Request, tool_request)
		self.assertEqual(tool_request.id, '1')
		self.assertEqual(tool_request.name, 'test_tool')
		self.assertEqual(tool_request.arguments, {'param': 'value'})
		self.assertEqual(context[3].role, Role.TOOL)
		tool_response = context[3].parts[0]
		self.assertIsInstance(tool_response, Result)
		tool_response = cast(Result, tool_response)
		self.assertEqual(tool_response.id, '1')
		self.assertEqual(tool_response.name, 'test_tool')
		self.assertEqual(tool_response.result, 'ok')

	def test_reset(self):
		json_str = '''
		[
			{
				"role": "user",
				"parts": [
					{
						"type": "text/plain",
						"text": "Hello, how are you?"
					}
				]
			},
			{
				"role": "model",
				"parts": [
					{
						"type": "text/plain",
						"text": "I'm fine, thank you!"
					}
				]
			}
		]
		'''
		context = Context(json_str)
		context.reset()
		self.assertEqual(len(context), 1)
		self.assertEqual(context[0].role, Role.SYSTEM)
		self.assertIsInstance(context[0].parts[0], Message)
		model_message = cast(Message, context[0].parts[0])
		self.assertEqual(model_message.text, 'Answer the following queries directly and provide the single best answer concisely without any quotes.')

	def test_add_text(self):
		context = Context("")
		context.add_text(Role.USER, ['Hello', 'World'])
		self.assertEqual(context[-1].role, Role.USER)
		self.assertEqual(len(context[-1].parts), 2)
		for i, txt in enumerate(['Hello', 'World']):
			part = context[-1].parts[i]
			self.assertIsInstance(part, Message)
			part = cast(Message, part)
			self.assertEqual(part.text, txt)

	def test_add_results(self):
		results: list[Result] = [Result(id='1', name='test', result='ok')]
		context = Context('')
		context.add_results(results)
		self.assertEqual(context[-1].role, Role.TOOL)
		part = context[-1].parts[0]
		self.assertIsInstance(part, Result)
		part = cast(Result, part)
		self.assertEqual(part.result, 'ok')

	def test_get_last_response(self):
		context = Context('')
		context.add_text(Role.USER, ['Hello'])
		context.add_text(Role.MODEL, ['Hi there!'])
		self.assertEqual(context.get_last_response(), 'Hi there!')
		context.add_results([
			Result(id='2', name='test', result='done')
		])
		self.assertEqual(context.get_last_response(), 'done')

	def test_to_json(self):
		context = Context('')
		context.add_text(Role.USER, ['Hello'])
		# Add a tool request
		context.extend([
			Entry(role=Role.MODEL, parts=[
				Request(id='1', name='test_tool', arguments={'param': 'value'})
			])
		])
		# Add a tool response
		context.add_results([
			Result(id='1', name='test_tool', result='ok')
		])
		json_str = context.to_json()
		expected_json = json.dumps([
			{
				'role': Role.SYSTEM.value,
				'parts': [
					{
						'type': PartType.TEXT.value,
						'text': 'Answer the following queries directly and provide the single best answer concisely without any quotes.'
					}
				]
			},
			{
				'role': Role.USER.value,
				'parts': [
					{
						'type': PartType.TEXT.value,
						'text': 'Hello'
					}
				]
			},
			{
				'role': Role.MODEL.value,
				'parts': [
					{
						'type': PartType.REQUEST.value,
						'id': '1',
						'name': 'test_tool',
						'arguments': {'param': 'value'}
					}
				]
			},
			{
				'role': Role.TOOL.value,
				'parts': [
					{
						'type': PartType.RESULT.value,
						'id': '1',
						'name': 'test_tool',
						'result': 'ok'
					}
				]
			}
		], indent=4)
		self.assertEqual(json.loads(json_str), json.loads(expected_json))


if __name__ == '__main__':
	unittest.main()

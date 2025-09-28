import logging
import random
from tools import JsonValue, Tool, ToolDefinition

class DiceTool(Tool):
	MAX_DICE = 64
	MIN_DICE = 1
	MAX_SIDES = 64
	MIN_SIDES = 2

	def execute(self, number: int = 1, sides: int = 6) -> JsonValue:
		"""
		Throw N dice with M sides each.

		Args:
			number: Number of dice to throw (default: 1)
			sides: Number of sides per die (default: 6)

		Returns:
			Dictionary with the results
		"""
		logging.debug(f'Throwing {number}d{sides}')

		if number < self.MIN_DICE or number > self.MAX_DICE:
			return {'error': f'Number of dice must be between {self.MIN_DICE} and {self.MAX_DICE}'}

		if sides < self.MIN_SIDES or sides > self.MAX_SIDES:
			return {'error': f'Number of sides must be between {self.MIN_SIDES} and {self.MAX_SIDES}'}

		results = [random.randint(1, sides) for _ in range(number)]
		total = sum(results)

		logging.debug(f'Dice results: {results} (total: {total})')

		return {
			'number': number,
			'sides': sides,
			'results': results,
			'total': total
		}

	@staticmethod
	def definition() -> ToolDefinition:
		return ToolDefinition(
			description='Throw N dice with M sides each. Returns the individual results and total sum.',
			parameters={
				'type': 'object',
				'properties': {
					'number': {
						'type': 'integer',
						'description': f'Number of dice to throw ({DiceTool.MIN_DICE}-{DiceTool.MAX_DICE})',
						'minimum': DiceTool.MIN_DICE,
						'maximum': DiceTool.MAX_DICE,
						'default': DiceTool.MIN_DICE
					},
					'sides': {
						'type': 'integer',
						'description': f'Number of sides per die ({DiceTool.MIN_SIDES}-{DiceTool.MAX_SIDES})',
						'minimum': DiceTool.MIN_SIDES,
						'maximum': DiceTool.MAX_SIDES,
						'default': min(max(6, DiceTool.MIN_SIDES), DiceTool.MAX_SIDES)
					}
				},
				'required': []
			}
		)

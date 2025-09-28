import dataclasses
import json
from enum import Enum
from typing import Dict, List, Mapping, Protocol, Sequence, Any, Type, TypeVar
from dataclasses import dataclass
from tools import JsonValue


class Role(Enum):
	USER = 'user'
	MODEL = 'model'
	SYSTEM = 'system'
	TOOL = 'tool'


class PartType(Enum):
	TEXT = 'text/plain'
	PNG = 'image/png'
	REQUEST = 'application/x-request'
	RESULT = 'application/x-result'
	NOTE = 'text/x-note'
	# ...


class PartProtocol(Protocol):
	type: PartType


@dataclass(frozen=True)
class Part:
	pass


registry: Dict[str, Type[Part]] = {}


@dataclass(frozen=True)
class Entry:
	role: Role
	parts: Sequence[Part]

	T = TypeVar('T', bound=PartProtocol)
	@classmethod
	def part(cls, part_cls: Type[T]) -> Type[T]:
		if dataclasses.is_dataclass(part_cls) and issubclass(part_cls, Part):
			type_field = next(f for f in dataclasses.fields(part_cls) if f.name == 'type')
			if type_field.default is not dataclasses.MISSING:
				registry[type_field.default.value] = part_cls
		return part_cls


@Entry.part
@dataclass(frozen=True)
class Message(Part):
	text: str
	type: PartType = PartType.TEXT



@Entry.part
@dataclass(frozen=True)
class Request(Part):
	id: str
	name: str
	arguments: Mapping[str, JsonValue]
	type: PartType = PartType.REQUEST



@Entry.part
@dataclass(frozen=True)
class Result(Part):
	id: str
	name: str
	result: JsonValue
	type: PartType = PartType.RESULT


class Context(List[Entry]):
	def __init__(self, context_json: str):
		if context_json:
			self.from_json(context_json)
		else:
			self.reset()

	def add_text(self, role: Role, text: Sequence[str]):
		parts = [Message(text=t) for t in text]
		entry = Entry(role=role, parts=parts)
		self.append(entry)

	def add_results(self, results: Sequence[Result]):
		entry = Entry(role=Role.TOOL, parts=results)
		self.append(entry)

	def reset(self):
		self.clear()
		self.append(
			Entry(
				role=Role.SYSTEM,
				parts=[
					Message(text='Answer the following queries directly and provide the single best answer concisely without any quotes.')
				]
			),
		)

	def get_last_response(self) -> str | JsonValue | None:
		'''Get the last response from the model.'''
		for entry in reversed(self):
			if entry.role == Role.MODEL or entry.role == Role.TOOL:
				for part in entry.parts:
					if isinstance(part, Message):
						return part.text
					elif isinstance(part, Result):
						return part.result
		return None

	def to_json(self):
		def context_to_dict(o: Any) -> Any:
			if isinstance(o, Part) or isinstance(o, Entry):
				return dataclasses.asdict(o)
			if isinstance(o, Enum):
				return o.value
			return o

		return json.dumps(self, default=context_to_dict, indent=4)

	def from_json(self, json_str: str):
		data = json.loads(json_str)
		self.extend([
			Entry(
				role=Role(e['role']),
				parts=[
					registry[p['type']](**{k: v for k, v in p.items() if k != 'type'})
					for p in e['parts']
				]
			)
			for e in data
		])

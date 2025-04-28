from enum import Enum
import dataclasses
from dataclasses import dataclass
import json


class Role(Enum):
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"


class PartType(Enum):
    TEXT = "text/plain"
    PNG = "image/png"
    # ...


@dataclass
class Part:
    type: PartType
    content: str


@dataclass
class Entry:
    role: Role
    parts: list[Part]


class Context:
    def __init__(self, context_json: str = None):
        if context_json:
            self.from_json(context_json)
        else:
            self.reset()

    def add_text(self, role: Role, text: list[str]):
        parts = [Part(PartType.TEXT, t) for t in text]
        self.add_entry(role, parts)

    def add_entry(self, role: Role, parts: list[Part]):
        entry = Entry(role=role, parts=parts)
        self.entries.append(entry)

    def reset(self):
        self.entries = [
            Entry(
                role=Role.SYSTEM,
                parts=[
                    Part(
                        PartType.TEXT, 
                        "Answer the following queries directly and provide the single best answer without any quotes."
                    ),
                ]
            ),
        ]

    def get_entries(self):
        return self.entries    

    def to_json(self):
        return json.dumps(self.entries, default=dataclass_aware_json_parser, indent=4)
    
    def from_json(self, json_str: str):
        data = json.loads(json_str)

        self.entries = [
            Entry(
                role=Role(e['role']),
                parts=[
                    Part(
                        type=PartType(p['type']), 
                        content=p['content']
                    )
                    for p in e['parts']
                ]
            )
            for e in data
        ]


def enum_aware_dict_factory(fields):
    result = {}
    for name, value in fields:
        if isinstance(value, Enum):
            result[name] = value.value
        else:
            result[name] = value
    return result


def dataclass_aware_json_parser(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o, dict_factory=enum_aware_dict_factory)

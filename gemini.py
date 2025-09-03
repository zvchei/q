from context import Role, PartType, Entry
import json
import subprocess
from core import fetch, LLMBackend


class Gemini(LLMBackend):
    def __init__(self, model="gemini-2.0-flash"):
        self.max_context_length = 1048576
        self.model = model
        self.api_key = self.lookup_secret("gemini", "api-key")

    def lookup_secret(self, service_name: str, key_name: str):
        command = [
            "secret-tool",  
            "lookup",
            service_name,
            key_name
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def generate_response(self, context):
        system, contents = self.parse_context(context)
        data = json.dumps({
            "system_instruction": system,
            "contents": contents
        })

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': self.api_key
        }

        result = fetch(url, data, headers)

        result = result.get("candidates") or [{}]
        result = result[0].get("content") or {}
        result = result.get("parts") or [{}]
        result = result[0].get("text")

        return result

    def parse_context(self, context):
        entries = context.get_entries()
        system = {"parts": []}
        contents = []

        for entry in entries:
            if entry.role == Role.SYSTEM:
                parts = [self._format_part(p) for p in entry.parts]
                system["parts"].extend(parts)
            else:
                contents.append(self._format_entry(entry))

        return system, contents

    def _format_entry(self, entry: Entry):
        role = {
            # Role.SYSTEM: "system",
            Role.USER: "user",
            Role.MODEL: "model"
        }.get(entry.role, "user")

        parts = [self._format_part(p) for p in entry.parts]

        return {"role": role, "parts": parts}

    def _format_part(self, part):
        part_type = {
            PartType.TEXT: "text",
            PartType.PNG: "image",
            # ...
        }.get(part.type, "text")

        return {part_type: part.content}

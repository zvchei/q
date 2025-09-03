import json

from context import PartType, Role, Entry
from core import fetch, lookup_secret, LLMBackend

class NvidiaNim(LLMBackend):
    def __init__(self, model):
        self.model = model
        self.api_key = lookup_secret("nvidia-nim", "api-key")

    def generate_response(self, context):
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        messages = self.parse_context(context)
        data = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 1.00,
            "top_p": 1.00,
            "frequency_penalty": 0.00,
            "presence_penalty": 0.00,
            "stream": False
        })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

        print(url)
        print("")
        print(json.dumps(json.loads(data), indent=2))
        print("")
        print(headers)

        result = fetch(url, data, headers)

        result = result.get("choices") or [{}]
        result = result[0].get("message") or {}
        result = result.get("content")
        return result

    def parse_context(self, context):
        return [self._format_entry(entry) for entry in context.get_entries()]

    def _format_entry(self, entry: Entry):
        role = {
            Role.SYSTEM: "system",
            Role.USER: "user",
            Role.MODEL: "assistant"
        }.get(entry.role, "user")

        if len(entry.parts) == 1 and entry.parts[0].type == PartType.TEXT:
            content = entry.parts[0].content
        else:
            content = [self._format_part(p) for p in entry.parts]

        return {"role": role, "content": content}

    def _format_part(self, part):
        part = {
            PartType.TEXT: {"text": part.content, "type": "text"},
            # PartType.PNG: ...
            # ...
        }.get(part.type, "text")

        return part

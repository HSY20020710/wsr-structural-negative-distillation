from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 180
    retries: int = 3
    use_json_schema: bool = True


class OpenAICompatibleTeacher:
    def __init__(self, config: TeacherConfig):
        self.config = config

    def complete(self, messages: list[dict], response_schema: dict) -> dict:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "welding_extraction",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        last_error = None
        for attempt in range(self.config.retries):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
                last_error = error
                if attempt + 1 < self.config.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Teacher request failed after retries: {last_error}")


def response_content(response: dict) -> str:
    choice = response["choices"][0]["message"]
    content = choice.get("content", "")
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content


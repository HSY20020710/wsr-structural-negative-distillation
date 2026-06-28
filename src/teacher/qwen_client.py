from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


def normalize_api_url(api_url: str) -> tuple[str, str | None]:
    value = api_url.strip()
    normalized = re.sub(r":(\d+)(V\d+)/", r":\1/\2/", value, flags=re.IGNORECASE)
    warning = None
    if normalized != value:
        warning = f"Normalized malformed API URL from {value} to {normalized}"
    return normalized, warning


@dataclass(frozen=True)
class QwenClientConfig:
    api_url: str
    model: str
    protocol: str = "auto"
    timeout_seconds: int = 180
    max_retries: int = 3
    sleep_seconds: float = 2.0
    temperature: float = 0.0
    top_p: float = 0.8
    max_tokens: int = 4096


class QwenGenerateClient:
    def __init__(self, config: QwenClientConfig):
        api_url, warning = normalize_api_url(config.api_url)
        self.config = QwenClientConfig(
            api_url=api_url,
            model=config.model,
            protocol=config.protocol,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            sleep_seconds=config.sleep_seconds,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )
        self.url_warning = warning

    def generate(self, prompt: str) -> dict:
        ollama_payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
        }
        openai_payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        candidates = self._request_candidates(ollama_payload, openai_payload)
        last_error = None
        attempts = 0
        for url, payload, protocol in candidates:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json; charset=utf-8"}
            for retry in range(1, self.config.max_retries + 1):
                attempts += 1
                request = urllib.request.Request(
                    url, data=body, headers=headers, method="POST"
                )
                try:
                    with urllib.request.urlopen(
                        request, timeout=self.config.timeout_seconds
                    ) as response:
                        response_text = response.read().decode("utf-8")
                        try:
                            data = json.loads(response_text)
                        except json.JSONDecodeError:
                            data = {"response": response_text}
                        return {
                            "success": True,
                            "data": data,
                            "response_text": extract_generated_text(data),
                            "attempts": attempts,
                            "protocol": protocol,
                            "resolved_url": url,
                            "url_warning": self.url_warning,
                        }
                except urllib.error.HTTPError as error:
                    last_error = f"{protocol} {url}: HTTP {error.code}"
                    # A missing route should immediately try the next protocol.
                    if error.code == 404:
                        break
                    if retry < self.config.max_retries:
                        time.sleep(self.config.sleep_seconds)
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    last_error = f"{protocol} {url}: {error!r}"
                    if retry < self.config.max_retries:
                        time.sleep(self.config.sleep_seconds)
        return {
            "success": False,
            "data": None,
            "response_text": "",
            "attempts": attempts,
            "error": last_error,
            "url_warning": self.url_warning,
        }

    def _request_candidates(
        self, ollama_payload: dict, openai_payload: dict
    ) -> list[tuple[str, dict, str]]:
        protocol = self.config.protocol.lower()
        if protocol == "ollama":
            return [(self.config.api_url, ollama_payload, "ollama")]
        if protocol == "openai":
            return [(self.config.api_url, openai_payload, "openai")]
        parsed = urllib.parse.urlsplit(self.config.api_url)
        origin = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, "", "", "")
        ).rstrip("/")
        openai_url = f"{origin}/v1/chat/completions"
        if self.config.api_url.rstrip("/").endswith("/v1/chat/completions"):
            return [(self.config.api_url, openai_payload, "openai")]
        return [
            (self.config.api_url, ollama_payload, "ollama"),
            (openai_url, openai_payload, "openai"),
        ]


def extract_generated_text(data: object) -> str:
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)
    for key in ["response", "generated_text", "output", "text", "content"]:
        value = data.get(key)
        if isinstance(value, str):
            return value
    nested = data.get("data")
    if isinstance(nested, str):
        return nested
    if isinstance(nested, dict):
        return extract_generated_text(nested)
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    return json.dumps(data, ensure_ascii=False)

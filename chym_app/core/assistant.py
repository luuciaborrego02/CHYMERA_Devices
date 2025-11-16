"""Lightweight wrapper around the OpenAI Assistants API."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:  # Optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - OpenAI is optional at runtime
    OpenAI = None  # type: ignore


class AssistantUnavailable(RuntimeError):
    """Raised when the assistant feature cannot be initialised."""


@dataclass
class AssistantConfig:
    api_key: Optional[str] = None
    assistant_id: Optional[str] = None
    poll_interval: float = 0.5
    timeout: float = 45.0


class AssistantClient:
    """Simple synchronous client for the OpenAI Assistants beta API."""

    def __init__(self, config: AssistantConfig) -> None:
        if OpenAI is None:
            raise AssistantUnavailable("The 'openai' package is not installed.")
        if not config.api_key or not config.assistant_id:
            raise AssistantUnavailable("Missing OPENAI_API_KEY or OPENAI_ASSISTANT_ID.")

        self._config = config
        self._client = OpenAI(api_key=config.api_key)
        self._assistant_id = config.assistant_id
        self._thread_id: Optional[str] = None

    # ------------------------------------------------------------------
    def send(self, prompt: str) -> str:
        """Send a message to the assistant and return the response text."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        thread_id = self._ensure_thread()
        self._client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=prompt,
        )
        run = self._client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self._assistant_id,
        )

        start = time.time()
        while True:
            run = self._client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id,
            )
            if run.status == "completed":
                break
            if run.status in {"failed", "cancelled", "expired"}:
                raise AssistantUnavailable(f"Assistant run failed with status: {run.status}")
            if time.time() - start > self._config.timeout:
                raise AssistantUnavailable("Assistant response timed out.")
            time.sleep(self._config.poll_interval)

        messages = self._client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)
        if not messages.data:
            raise AssistantUnavailable("Assistant did not return any messages.")
        latest = messages.data[0]
        if not latest.content:
            raise AssistantUnavailable("Assistant response is empty.")
        return latest.content[0].text.value  # type: ignore[index]

    # ------------------------------------------------------------------
    def _ensure_thread(self) -> str:
        if self._thread_id:
            return self._thread_id
        thread = self._client.beta.threads.create()
        self._thread_id = thread.id
        return thread.id


__all__ = ["AssistantClient", "AssistantConfig", "AssistantUnavailable"]

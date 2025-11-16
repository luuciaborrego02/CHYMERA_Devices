"""Dockable widget that lets users chat with the OpenAI assistant."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.assistant import AssistantClient, AssistantUnavailable


class _ChatWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, client: AssistantClient, prompt: str) -> None:
        super().__init__()
        self._client = client
        self._prompt = prompt

    def run(self) -> None:  # noqa: D401 - QThread signature
        try:
            reply = self._client.send(self._prompt)
        except AssistantUnavailable as exc:  # pragma: no cover - network dependent
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - external dependency failures
            self.failed.emit(f"Assistant error: {exc}")
        else:
            self.completed.emit(reply)


class AssistantChatPanel(QWidget):
    """Simple chat UI that mirrors the Streamlit agent experience."""

    def __init__(self, client: Optional[AssistantClient]) -> None:
        super().__init__()
        self._client = client
        self._worker: Optional[_ChatWorker] = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("💬 ChymAI Assistant")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        self._history = QTextEdit()
        self._history.setReadOnly(True)
        self._history.setStyleSheet(
            "background-color: #101826; border: 1px solid #1f2a37; border-radius: 8px;"
        )
        layout.addWidget(self._history, 1)

        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask the assistant anything about the study...")
        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self._input, 1)
        input_row.addWidget(self._send_btn)
        layout.addLayout(input_row)

        self._status = QLabel()
        self._status.setStyleSheet("color: #9ca3af;")
        layout.addWidget(self._status)

        if self._client is None:
            self._input.setDisabled(True)
            self._send_btn.setDisabled(True)
            self._status.setText("Configure OPENAI_API_KEY and OPENAI_ASSISTANT_ID to enable the agent.")

    # ------------------------------------------------------------------
    def _append_message(self, author: str, text: str) -> None:
        self._history.append(f"<b>{author}:</b> {text}")

    def _on_send(self) -> None:
        if not self._client:
            return
        prompt = self._input.text().strip()
        if not prompt:
            return
        self._append_message("You", prompt)
        self._input.clear()
        self._send_btn.setDisabled(True)
        self._status.setText("Assistant is typing…")

        self._worker = _ChatWorker(self._client, prompt)
        self._worker.completed.connect(self._on_reply)
        self._worker.failed.connect(self._on_error)
        self._worker.finished.connect(self._on_finish)
        self._worker.start()

    def _on_reply(self, text: str) -> None:
        self._append_message("ChymAI", text)
        self._status.setText("")

    def _on_error(self, text: str) -> None:
        self._status.setText(text)

    def _on_finish(self) -> None:
        self._send_btn.setDisabled(False)
        self._worker = None


__all__ = ["AssistantChatPanel"]

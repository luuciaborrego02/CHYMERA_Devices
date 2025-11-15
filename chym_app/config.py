"""Application-wide configuration values."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """Color palette for the dark theme."""

    window: str = "#0e1117"
    base: str = "#1a1f2b"
    text: str = "#e5e5e5"
    highlight: str = "#1f77b4"
    accent: str = "#3b82f6"
    muted: str = "#4b5563"


palette = Palette()

APP_NAME = "Quality Chest Analyzer"
APP_ICON = "🩺"

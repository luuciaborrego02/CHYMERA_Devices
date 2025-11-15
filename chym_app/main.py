"""Entry point for the Qt based Quality Chest Analyzer application."""
from __future__ import annotations

import sys

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

from .config import palette
from .ui.main_window import MainWindow


def _apply_dark_theme(app: QApplication) -> None:
    """Apply a dark fusion palette to the application."""
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(palette.window))
    dark_palette.setColor(QPalette.WindowText, QColor(palette.text))
    dark_palette.setColor(QPalette.Base, QColor(palette.base))
    dark_palette.setColor(QPalette.AlternateBase, QColor(palette.window))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(palette.base))
    dark_palette.setColor(QPalette.ToolTipText, QColor(palette.text))
    dark_palette.setColor(QPalette.Text, QColor(palette.text))
    dark_palette.setColor(QPalette.Button, QColor(palette.base))
    dark_palette.setColor(QPalette.ButtonText, QColor(palette.text))
    dark_palette.setColor(QPalette.BrightText, QColor("#ff5555"))
    dark_palette.setColor(QPalette.Highlight, QColor(palette.highlight))
    dark_palette.setColor(QPalette.HighlightedText, QColor(palette.window))
    app.setPalette(dark_palette)


def main() -> int:
    """Launch the Qt application."""
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

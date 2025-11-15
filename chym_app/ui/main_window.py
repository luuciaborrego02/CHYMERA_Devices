"""Main window for the Quality Chest Analyzer Qt application."""
from __future__ import annotations

import pathlib
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
)

from ..config import APP_NAME
from ..core.image_processor import AnalysisBundle, ImageProcessor, ProcessingError
from ..widgets.analysis_views import (
    InclusionView,
    RotationView,
    ArtifactsView,
    RotationBalanceView,
    CTRView,
    ScapulaView,
    GlobalStatsView,
)


class ImageListWidget(QListWidget):
    """List widget that exposes a custom signal when selection changes."""

    selection_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.itemSelectionChanged.connect(self._emit_selection)

    @Slot()
    def _emit_selection(self) -> None:
        current_item = self.currentItem()
        if current_item is not None:
            self.selection_changed.emit(current_item.text())


class MainWindow(QMainWindow):
    """Top level window orchestrating the UI layout and interactions."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)

        self._image_processor = ImageProcessor()
        self._analyses: Dict[str, AnalysisBundle] = {}
        self._selected_image: Optional[str] = None

        self._create_toolbar()
        self._build_layout()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        open_action = QAction("Open images", self)
        open_action.triggered.connect(self._select_images)
        toolbar.addAction(open_action)

        analyze_action = QAction("Run analyses", self)
        analyze_action.triggered.connect(self._run_analysis)
        toolbar.addAction(analyze_action)

    def _build_layout(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        self._image_list = ImageListWidget()
        self._image_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._image_list.selection_changed.connect(self._on_image_selected)
        sidebar_layout.addWidget(QLabel("Loaded images"))
        sidebar_layout.addWidget(self._image_list)

        self._analysis_selector = QListWidget()
        for view_name in (
            "Inclusion",
            "Rotation",
            "Artifacts",
            "Rotation Balance",
            "Cardio-thoracic Ratio",
            "Scapula",
            "Global Statistics",
        ):
            QListWidgetItem(view_name, self._analysis_selector)
        self._analysis_selector.setCurrentRow(0)
        self._analysis_selector.currentRowChanged.connect(self._display_analysis)
        sidebar_layout.addWidget(QLabel("Analysis views"))
        sidebar_layout.addWidget(self._analysis_selector)

        splitter.addWidget(sidebar)

        # Main view area
        self._stack = QStackedWidget()
        self._stack.addWidget(InclusionView())
        self._stack.addWidget(RotationView())
        self._stack.addWidget(ArtifactsView())
        self._stack.addWidget(RotationBalanceView())
        self._stack.addWidget(CTRView())
        self._stack.addWidget(ScapulaView())
        self._stack.addWidget(GlobalStatsView())

        splitter.addWidget(self._stack)
        splitter.setStretchFactor(1, 1)

        # Footer area for logs/status
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 8, 16, 8)
        footer_layout.setSpacing(8)

        self._status_text = QTextEdit()
        self._status_text.setReadOnly(True)
        self._status_text.setFixedHeight(120)
        footer_layout.addWidget(self._status_text)

        self._final_score_label = QLabel("Final score: pending")
        self._final_score_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        footer_layout.addWidget(self._final_score_label)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)
        wrapper_layout.addWidget(splitter)
        wrapper_layout.addWidget(footer)
        self.setCentralWidget(wrapper)

    # ------------------------------------------------------------------
    # Slots / event handlers
    # ------------------------------------------------------------------
    @Slot()
    def _select_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select chest X-ray images",
            str(pathlib.Path.home()),
            "Images (*.png *.jpg *.jpeg)",
        )
        if not files:
            return

        self._image_list.clear()
        for file in files:
            QListWidgetItem(file, self._image_list)

        self._analyses.clear()
        self._status_text.clear()
        self._final_score_label.setText("Final score: pending")

    @Slot()
    def _run_analysis(self) -> None:
        if self._image_list.count() == 0:
            QMessageBox.information(self, APP_NAME, "Please load at least one image first.")
            return

        self._status_text.append("Starting analyses...")
        for index in range(self._image_list.count()):
            path = self._image_list.item(index).text()
            try:
                bundle = self._image_processor.process(pathlib.Path(path))
            except ProcessingError as exc:
                self._status_text.append(f"⚠️ {path}: {exc}")
                continue

            self._analyses[path] = bundle
            self._status_text.append(f"✅ {path}: analysis completed")

        if self._selected_image:
            self._display_current_analysis()

        self._update_final_score_summary()

    @Slot(str)
    def _on_image_selected(self, path: str) -> None:
        self._selected_image = path
        self._display_current_analysis()

    @Slot(int)
    def _display_analysis(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        self._display_current_analysis()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _display_current_analysis(self) -> None:
        if not self._selected_image:
            return

        bundle = self._analyses.get(self._selected_image)
        if not bundle:
            self._status_text.append(f"ℹ️ {self._selected_image}: run analyses to view results")
            return

        widget = self._stack.currentWidget()
        if hasattr(widget, "display"):
            widget.display(bundle)

    def _update_final_score_summary(self) -> None:
        if not self._analyses:
            return

        scores = [bundle.final_score for bundle in self._analyses.values()]
        summary = ", ".join(scores)
        self._final_score_label.setText(f"Final score: {summary}")

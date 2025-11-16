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
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
)

from ..config import APP_NAME, OPENAI_API_KEY, OPENAI_ASSISTANT_ID
from ..core.image_processor import AnalysisBundle, ImageProcessor, ProcessingError
from ..core.assistant import AssistantClient, AssistantConfig, AssistantUnavailable
from ..widgets.analysis_views import (
    InclusionView,
    RotationView,
    ArtifactsView,
    RotationBalanceView,
    CTRView,
    ScapulaView,
    GlobalStatsView,
)
from ..widgets.chat_panel import AssistantChatPanel


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
        self._assistant_client: Optional[AssistantClient] = None
        self._view_menu = self.menuBar().addMenu("&View")

        self._create_toolbar()
        self._build_layout()
        self._init_chat_panel()

        self._apply_styles()

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
        images_label = QLabel("Loaded images")
        images_label.setObjectName("sectionTitle")
        sidebar_layout.addWidget(images_label)
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
        analysis_label = QLabel("Analysis views")
        analysis_label.setObjectName("sectionTitle")
        sidebar_layout.addWidget(analysis_label)
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
        self._status_text.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #1f2937; border-radius: 8px;"
        )
        footer_layout.addWidget(self._status_text)

        self._final_score_label = QLabel("Final score: pending")
        self._final_score_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._final_score_label.setStyleSheet("font-size: 20px; font-weight: 600;")
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
        self._selected_image = None
        for file in files:
            QListWidgetItem(file, self._image_list)

        self._ensure_image_selection()

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

        self._ensure_image_selection()
        if self._selected_image:
            self._display_current_analysis()

        self._update_final_score_label()

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
            self._update_final_score_label()
            return

        bundle = self._analyses.get(self._selected_image)
        if not bundle:
            self._status_text.append(f"ℹ️ {self._selected_image}: run analyses to view results")
            self._update_final_score_label()
            return

        widget = self._stack.currentWidget()
        if hasattr(widget, "display"):
            widget.display(bundle)
        self._update_final_score_label()

    def _update_final_score_label(self) -> None:
        if not self._analyses:
            self._final_score_label.setText("Final score: pending")
            return

        if self._selected_image and self._selected_image in self._analyses:
            selected_bundle = self._analyses[self._selected_image]
            display_name = pathlib.Path(self._selected_image).name
            self._final_score_label.setText(
                f"Final score ({display_name}): {selected_bundle.final_score}"
            )
            return

        scores = [bundle.final_score for bundle in self._analyses.values()]
        summary = ", ".join(scores)
        self._final_score_label.setText(f"Final score: {summary}")

    def _ensure_image_selection(self) -> None:
        if self._image_list.count() == 0:
            self._selected_image = None
            return

        if self._image_list.currentRow() == -1:
            self._image_list.blockSignals(True)
            self._image_list.setCurrentRow(0)
            self._image_list.blockSignals(False)

        current_item = self._image_list.currentItem()
        if current_item is not None:
            self._selected_image = current_item.text()

    def _createDockWidgetAction(self, dock: QDockWidget) -> None:
        if self._view_menu is None:
            return
        action = dock.toggleViewAction()
        action.setText("ChymAI Panel")
        self._view_menu.addAction(action)

    def _init_chat_panel(self) -> None:
        client: Optional[AssistantClient] = None
        if not OPENAI_API_KEY or not OPENAI_ASSISTANT_ID:
            self._status_text.append("ℹ️ ChymAI disabled: missing OPENAI credentials.")
        else:
            config = AssistantConfig(
                api_key=OPENAI_API_KEY,
                assistant_id=OPENAI_ASSISTANT_ID,
            )
            try:
                client = AssistantClient(config)
            except AssistantUnavailable as exc:
                self._status_text.append(f"ℹ️ ChymAI disabled: {exc}")

        self._assistant_client = client
        dock = QDockWidget("ChymAI", self)
        dock.setObjectName("ChymAIDock")
        dock.setWidget(AssistantChatPanel(client))
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._createDockWidgetAction(dock)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QListWidget {
                background-color: #0f172a;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 8px;
            }
            QListWidget::item:selected {
                background-color: #1d4ed8;
                color: #f1f5f9;
            }
            QLabel#sectionTitle {
                color: #94a3b8;
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 1px;
            }
            QToolBar {
                background-color: #0f172a;
                border-bottom: 1px solid #1f2937;
            }
            QSplitter::handle {
                background-color: #1f2937;
            }
            """
        )

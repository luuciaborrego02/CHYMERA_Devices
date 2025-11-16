"""Reusable widgets that visualise individual analysis outputs."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget, QTextBrowser, QSizePolicy

try:  # Optional import for interactive charts
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover - optional dependency
    QWebEngineView = None  # type: ignore

from ..core.image_processor import AnalysisBundle


def _numpy_to_pixmap(array: np.ndarray) -> Optional[QPixmap]:
    data = np.require(array, requirements=["C_CONTIGUOUS", "OWNDATA"])
    if data.ndim == 2:
        height, width = data.shape
        qimage = QImage(data.data, width, height, width, QImage.Format_Grayscale8)
    elif data.ndim == 3 and data.shape[2] == 3:
        height, width, _ = data.shape
        qimage = QImage(data.data, width, height, width * 3, QImage.Format_RGB888)
    else:
        return None
    return QPixmap.fromImage(qimage.copy())


class _BaseView(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        self._title = QLabel(f"<h1>{title}</h1>")
        self._title.setTextFormat(Qt.RichText)
        layout.addWidget(self._title)

        self._image_label = QLabel()
        self._image_label.setMinimumSize(260, 260)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet(
            "background-color: #0e1117; border: 1px solid #2b2f3a; border-radius: 8px;"
        )
        layout.addWidget(self._image_label)

        self._text = QTextBrowser()
        self._text.setOpenExternalLinks(True)
        self._text.setMinimumHeight(160)
        layout.addWidget(self._text)

        self._current_pixmap: Optional[QPixmap] = None

    def _set_image(self, array: Optional[np.ndarray]) -> None:
        if array is None:
            self._image_label.clear()
            self._current_pixmap = None
            return
        pixmap = _numpy_to_pixmap(array)
        if pixmap is None:
            self._image_label.clear()
            self._current_pixmap = None
            return
        self._current_pixmap = pixmap
        self._render_scaled_pixmap()

    def _render_scaled_pixmap(self) -> None:
        if not self._current_pixmap:
            return
        target_size = self._image_label.size()
        if target_size.width() == 0 or target_size.height() == 0:
            return
        scaled = self._current_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._render_scaled_pixmap()

    def display(self, bundle: AnalysisBundle) -> None:
        raise NotImplementedError


class InclusionView(_BaseView):
    def __init__(self) -> None:
        super().__init__("Inclusion Quality")

    def display(self, bundle: AnalysisBundle) -> None:
        self._set_image(bundle.overlay_lung)
        self._text.setMarkdown(
            f"**Classification:** {bundle.inclusion_label}\n\n"
            f"**GradCAM available:** {'Yes' if bundle.unet_cam is not None else 'No'}"
        )


class RotationView(_BaseView):
    def __init__(self) -> None:
        super().__init__("Rotation (Clavicle)")

    def display(self, bundle: AnalysisBundle) -> None:
        self._set_image(bundle.overlay_clavicle)
        if bundle.orientation_status:
            text = f"**Status:** {bundle.orientation_status}"
        else:
            text = "**Status:** No clavicles detected"
        if bundle.hough_angles:
            text += ("\n\n" "**Hough angles:** "
                     f"Left {bundle.hough_angles.left:.1f}°, Right {bundle.hough_angles.right:.1f}°")
        self._text.setMarkdown(text)


class ArtifactsView(_BaseView):
    def __init__(self) -> None:
        super().__init__("Artifacts")

    def display(self, bundle: AnalysisBundle) -> None:
        self._set_image(bundle.overlay_artifacts)
        self._text.setMarkdown(
            f"Model 1 detections: {bundle.artifacts_count_1}\n\n"
            f"Model 2 detections: {bundle.artifacts_count_2}"
        )


class RotationBalanceView(_BaseView):
    def __init__(self) -> None:
        super().__init__("Rotation Balance")

    def display(self, bundle: AnalysisBundle) -> None:
        self._set_image(bundle.overlay_rotation_balance)
        self._text.setMarkdown(f"**Result:** {bundle.rotation_balance}")


class CTRView(_BaseView):
    def __init__(self) -> None:
        super().__init__("Cardio-thoracic Ratio")

    def display(self, bundle: AnalysisBundle) -> None:
        self._set_image(bundle.overlay_ctr)
        self._text.setMarkdown(
            f"**CTR:** {bundle.ctr_ratio:.2f}\n\n"
            f"**Interpretation:** {bundle.ctr_result_text}"
        )


class ScapulaView(_BaseView):
    def __init__(self) -> None:
        super().__init__("Scapula Overlap")

    def display(self, bundle: AnalysisBundle) -> None:
        self._set_image(bundle.overlay_scapula)
        self._text.setMarkdown(
            f"Left overlap: {bundle.scapula_overlap.left:.2f}%\n\n"
            f"Right overlap: {bundle.scapula_overlap.right:.2f}%"
        )


class _PlotlyChartContainer(QWidget):
    """Wrapper that hosts a QWebEngineView when available."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if QWebEngineView is not None:
            self._view: QWidget = QWebEngineView()
            self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(self._view)
            self._supports_interactive = True
        else:
            placeholder = QTextBrowser()
            placeholder.setReadOnly(True)
            placeholder.setPlainText(
                "Install PySide6-WebEngine to enable the interactive dashboard."
            )
            placeholder.setStyleSheet(
                "background-color: #101826; border: 1px dashed #374151; border-radius: 8px;"
            )
            placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(placeholder)
            self._view = placeholder
            self._supports_interactive = False

    def set_html(self, html: str) -> None:
        if self._supports_interactive:
            assert isinstance(self._view, QWebEngineView)
            self._view.setHtml(html)

    @property
    def supports_interactive(self) -> bool:
        return self._supports_interactive


class GlobalStatsView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        title = QLabel("<h1>Global Statistics</h1>")
        title.setTextFormat(Qt.RichText)
        layout.addWidget(title)

        self._chart = _PlotlyChartContainer()
        self._chart.setMinimumHeight(420)
        layout.addWidget(self._chart, 1)

        self._fallback_image = QLabel()
        self._fallback_image.setMinimumHeight(320)
        self._fallback_image.setAlignment(Qt.AlignCenter)
        self._fallback_image.setStyleSheet(
            "background-color: #0e1117; border: 1px solid #2b2f3a; border-radius: 8px;"
        )
        layout.addWidget(self._fallback_image)
        self._fallback_image.hide()

        self._text = QTextBrowser()
        self._text.setOpenExternalLinks(True)
        layout.addWidget(self._text)

    def display(self, bundle: AnalysisBundle) -> None:
        if bundle.global_stats_html and self._chart.supports_interactive:
            self._chart.set_html(bundle.global_stats_html)
            self._fallback_image.clear()
            self._fallback_image.hide()
        else:
            if bundle.global_stats_chart is not None:
                pixmap = _numpy_to_pixmap(bundle.global_stats_chart)
                if pixmap is not None:
                    target_size = self._fallback_image.size()
                    if target_size.width() == 0 or target_size.height() == 0:
                        self._fallback_image.setPixmap(pixmap)
                    else:
                        self._fallback_image.setPixmap(
                            pixmap.scaled(
                                target_size,
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation,
                            )
                        )
                    self._fallback_image.show()
            else:
                self._fallback_image.setText("Plotly is unavailable in this environment.")
                self._fallback_image.show()

        summary_lines = [
            f"**Total processed:** {bundle.global_stats.total_images}",
            f"All conditions met: {bundle.global_stats.all_good}",
            f"Revision required: {bundle.global_stats.needs_review}",
            f"Issues detected: {bundle.global_stats.issues_detected}",
            f"Bad: {bundle.global_stats.bad}",
        ]
        if bundle.global_stats.parameter_counts:
            summary_lines.append("\n**Parameters passing:**")
            for name, value in bundle.global_stats.parameter_counts.items():
                summary_lines.append(f"- {name}: {value}")
        if not bundle.global_stats_html:
            summary_lines.append("Interactive dashboard unavailable; showing static backup.")
        self._text.setMarkdown("\n".join(summary_lines))

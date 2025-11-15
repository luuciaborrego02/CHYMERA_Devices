"""Reusable widgets that visualise individual analysis outputs."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget, QTextBrowser

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
        self._image_label.setMinimumHeight(400)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setScaledContents(True)
        layout.addWidget(self._image_label)

        self._text = QTextBrowser()
        self._text.setOpenExternalLinks(True)
        self._text.setMinimumHeight(160)
        layout.addWidget(self._text)

    def _set_image(self, array: Optional[np.ndarray]) -> None:
        if array is None:
            self._image_label.clear()
            return
        pixmap = _numpy_to_pixmap(array)
        if pixmap is None:
            self._image_label.clear()
            return
        self._image_label.setPixmap(pixmap)

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


class GlobalStatsView(_BaseView):
    def __init__(self) -> None:
        super().__init__("Global Statistics")

    def display(self, bundle: AnalysisBundle) -> None:
        self._set_image(bundle.global_stats_chart)
        summary_lines = [
            f"**Total processed:** {bundle.global_stats.total_images}",
            f"All conditions met: {bundle.global_stats.all_good}",
            f"Revision required: {bundle.global_stats.needs_review}",
            f"Issues detected: {bundle.global_stats.issues_detected}",
            f"Bad: {bundle.global_stats.bad}",
        ]
        if bundle.global_stats_chart is None:
            summary_lines.append("Plot rendering is unavailable in this environment.")
        self._text.setMarkdown("\n".join(summary_lines))

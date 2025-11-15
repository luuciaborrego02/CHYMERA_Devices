"""Business logic for running model inference on chest X-ray images."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
import torch
import torchvision
import torchxrayvision as xrv
from tensorflow.keras.preprocessing.image import img_to_array

from ..config import BASE_PATH
from .model_loader import load_models, ModelLoadError

try:  # Optional dependency
    from torchcam.methods import GradCAM
except Exception:  # pragma: no cover - optional import fallback
    GradCAM = None  # type: ignore

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None  # type: ignore


class ProcessingError(RuntimeError):
    """Raised when a processing step fails."""


@dataclass
class HoughAngles:
    left: float
    right: float
    delta: float


@dataclass
class ScapulaOverlap:
    left: float
    right: float


@dataclass
class GlobalStats:
    total_images: int
    all_good: int
    needs_review: int
    issues_detected: int
    bad: int


@dataclass
class AnalysisBundle:
    image_path: Path
    inclusion_label: str
    orientation_status: Optional[str]
    rotation_balance: str
    ctr_ratio: float
    ctr_result_text: str
    artifacts_count_1: int
    artifacts_count_2: int
    final_score: str
    overlay_lung: Optional[np.ndarray] = None
    overlay_clavicle: Optional[np.ndarray] = None
    overlay_artifacts: Optional[np.ndarray] = None
    overlay_rotation_balance: Optional[np.ndarray] = None
    overlay_ctr: Optional[np.ndarray] = None
    overlay_scapula: Optional[np.ndarray] = None
    global_stats_chart: Optional[np.ndarray] = None
    unet_cam: Optional[np.ndarray] = None
    hough_angles: Optional[HoughAngles] = None
    scapula_overlap: ScapulaOverlap = field(default_factory=lambda: ScapulaOverlap(0.0, 0.0))
    global_stats: GlobalStats = field(default_factory=lambda: GlobalStats(0, 0, 0, 0, 0))


class ImageProcessor:
    """High level orchestrator to run all ML analyses for a given image."""

    def __init__(self) -> None:
        self._history: Dict[Path, AnalysisBundle] = {}
        self._base_path = BASE_PATH

    # ------------------------------------------------------------------
    def process(self, image_path: Path) -> AnalysisBundle:
        if not image_path.exists():
            raise ProcessingError(f"Image not found: {image_path}")

        try:
            models = load_models(self._base_path)
        except ModelLoadError as exc:
            raise ProcessingError(str(exc)) from exc

        unet, clf, yolo_model_1, yolo_model_2, clavicle_model, device = models

        image = Image.open(image_path).convert("RGB")
        origin, origin_tensor = self._preprocess_image(image)
        pred_mask = self._predict_mask(unet, origin_tensor, device)
        filtered_mask = self._get_filtered_mask(pred_mask)
        inclusion_label = self._classify_mask(clf, filtered_mask)

        overlay_lung = self._blend_image(image, filtered_mask)
        unet_cam = self._compute_unet_cam(unet, origin_tensor, image, device)

        rotation_bundle = self._analyse_rotation(image, clavicle_model)

        artifacts_result_1 = yolo_model_1(self._to_bgr(image))[0]
        artifacts_result_2 = yolo_model_2(self._to_bgr(image))[0]
        overlay_artifacts = self._compose_yolo_overlay(image, artifacts_result_1, artifacts_result_2)

        ctr_ratio, ctr_result_text, overlay_ctr = self._compute_ctr(rotation_bundle, image)

        rotation_balance, overlay_rotation_balance = self._compute_rotation_balance(rotation_bundle, image)

        scapula_overlap, overlay_scapula = self._analyse_scapula(rotation_bundle, filtered_mask, image)

        final_score = self._derive_final_score(
            inclusion_label,
            rotation_bundle.orientation_status,
            rotation_balance,
            ctr_ratio,
            artifacts_result_1,  # type: ignore[arg-type]
            artifacts_result_2,  # type: ignore[arg-type]
        )

        bundle = AnalysisBundle(
            image_path=image_path,
            inclusion_label=inclusion_label,
            orientation_status=rotation_bundle.orientation_status,
            rotation_balance=rotation_balance,
            ctr_ratio=ctr_ratio,
            ctr_result_text=ctr_result_text,
            artifacts_count_1=len(getattr(artifacts_result_1, "boxes", [])),
            artifacts_count_2=len(getattr(artifacts_result_2, "boxes", [])),
            final_score=final_score,
            overlay_lung=overlay_lung,
            overlay_clavicle=rotation_bundle.overlay_clavicle,
            overlay_artifacts=overlay_artifacts,
            overlay_rotation_balance=overlay_rotation_balance,
            overlay_ctr=overlay_ctr,
            overlay_scapula=overlay_scapula,
            global_stats_chart=None,
            unet_cam=unet_cam,
            hough_angles=rotation_bundle.hough_angles,
            scapula_overlap=scapula_overlap,
        )

        self._history[image_path] = bundle
        bundle.global_stats = self._global_stats()
        bundle.global_stats_chart = self._global_stats_chart(bundle.global_stats)
        return bundle

    # ------------------------------------------------------------------
    def _global_stats(self) -> GlobalStats:
        total = len(self._history)
        all_good = sum(1 for b in self._history.values() if b.final_score.startswith("🟢"))
        needs_review = sum(1 for b in self._history.values() if b.final_score.startswith("🟡"))
        issues = sum(1 for b in self._history.values() if b.final_score.startswith("🟠"))
        bad = sum(1 for b in self._history.values() if b.final_score.startswith("🔴"))
        return GlobalStats(total, all_good, needs_review, issues, bad)

    def _global_stats_chart(self, stats: GlobalStats) -> Optional[np.ndarray]:
        if px is None or stats.total_images == 0:
            return None
        fig = px.pie(
            names=["All good", "Needs review", "Issues", "Bad"],
            values=[stats.all_good, stats.needs_review, stats.issues_detected, stats.bad],
            title="Distribution of final scores",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(paper_bgcolor="#0e1117", font_color="#fafafa")
        try:
            image_bytes = fig.to_image(format="png")
        except Exception:
            return None
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_image(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
        origin = image.convert("L")
        origin = torchvision.transforms.functional.resize(origin, (512, 512))
        tensor = torchvision.transforms.functional.to_tensor(origin) - 0.5
        return origin, tensor

    @staticmethod
    def _predict_mask(unet: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
        input_tensor = torch.stack([tensor]).to(device)
        with torch.inference_mode():
            output = unet(input_tensor)
            probs = torch.nn.functional.log_softmax(output, dim=1)
            pred_mask = torch.argmax(probs, dim=1)[0].cpu().numpy()
        return (pred_mask * 255).astype(np.uint8)

    @staticmethod
    def _get_filtered_mask(mask: np.ndarray) -> np.ndarray:
        labeled_mask = label(mask)
        regions = sorted(regionprops(labeled_mask), key=lambda x: x.area, reverse=True)[:2]
        filtered_mask = np.zeros_like(mask)
        for region in regions:
            filtered_mask[labeled_mask == region.label] = 255
        return filtered_mask

    @staticmethod
    def _classify_mask(clf, mask: np.ndarray) -> str:
        mask_resized = cv2.resize(mask, (128, 128))
        mask_resized = img_to_array(mask_resized)
        mask_resized = np.repeat(mask_resized, 3, axis=-1)
        mask_resized = np.expand_dims(mask_resized, axis=0) / 255.0
        pred = clf.predict(mask_resized)
        return "INCLUSIÓN APTA ✅" if pred > 0.5 else "INCLUSIÓN NO APTA ❌"

    @staticmethod
    def _blend_image(image: Image.Image, mask: np.ndarray) -> np.ndarray:
        image_arr = np.array(image).astype(np.float32) / 255.0
        image_uint8 = (image_arr * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask, (image_uint8.shape[1], image_uint8.shape[0]))
        mask_rgb = np.stack([mask_resized, mask_resized, mask_resized], axis=-1).astype(np.uint8)
        blended = cv2.addWeighted(image_uint8, 0.6, mask_rgb, 0.4, 0)
        return blended

    def _compute_unet_cam(
        self,
        unet: torch.nn.Module,
        tensor: torch.Tensor,
        image: Image.Image,
        device: torch.device,
    ) -> Optional[np.ndarray]:
        if GradCAM is None:
            return None
        extractor = GradCAM(unet, target_layer="dec1.conv2", input_shape=(1, 512, 512))
        input_tensor = torch.stack([tensor]).to(device)
        input_tensor.requires_grad_(True)
        with torch.enable_grad():
            output = unet(input_tensor)
        class_idx = 1
        activation_map = extractor(class_idx, output)[0].cpu().numpy()
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
        activation_map = cv2.resize(activation_map, (image.width, image.height))
        activation_map_uint8 = np.uint8(activation_map * 255)
        heatmap = cv2.applyColorMap(activation_map_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
        return overlay

    # Rotation / clavicle -------------------------------------------------
    @dataclass
    class _RotationData:
        orientation_status: Optional[str]
        overlay_clavicle: Optional[np.ndarray]
        left_mask: Optional[np.ndarray]
        right_mask: Optional[np.ndarray]
        spine_mask: Optional[np.ndarray]
        heart_mask: Optional[np.ndarray]
        diaphragm_mask: Optional[np.ndarray]
        left_scapula: Optional[np.ndarray]
        right_scapula: Optional[np.ndarray]
        hough_angles: Optional[HoughAngles]

    def _analyse_rotation(self, image: Image.Image, clavicle_model) -> "ImageProcessor._RotationData":
        img_array = np.array(image.convert("RGB"))
        normalized = xrv.datasets.normalize(img_array, 255)
        img_gray = normalized.mean(2)[None, ...]
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(512),
        ])
        img_input = transform(img_gray)
        img_tensor = torch.from_numpy(img_input)
        with torch.inference_mode():
            pred_tensor = clavicle_model(img_tensor)
        pred = torch.sigmoid(pred_tensor).cpu().numpy()
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1

        left_mask = pred[0, clavicle_model.targets.index("Left Clavicle")]
        right_mask = pred[0, clavicle_model.targets.index("Right Clavicle")]
        spine_mask = pred[0, clavicle_model.targets.index("Spine")]
        heart_mask = pred[0, clavicle_model.targets.index("Heart")]
        diaphragm_mask = pred[0, clavicle_model.targets.index("Facies Diaphragmatica")]
        left_scapula = pred[0, clavicle_model.targets.index("Left Scapula")]
        right_scapula = pred[0, clavicle_model.targets.index("Right Scapula")]

        overlay_clavicle = self._overlay_clavicle(image, left_mask, right_mask)
        orientation_status, hough_angles = self._check_rotation(left_mask, right_mask)

        return ImageProcessor._RotationData(
            orientation_status=orientation_status,
            overlay_clavicle=overlay_clavicle,
            left_mask=left_mask,
            right_mask=right_mask,
            spine_mask=spine_mask,
            heart_mask=heart_mask,
            diaphragm_mask=diaphragm_mask,
            left_scapula=left_scapula,
            right_scapula=right_scapula,
            hough_angles=hough_angles,
        )

    @staticmethod
    def _overlay_clavicle(image: Image.Image, left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
        combined_mask = ((left_mask + right_mask) > 0).astype(np.float32)
        original = np.array(image.convert("L").resize((combined_mask.shape[1], combined_mask.shape[0])))
        original_rgb = np.stack([original] * 3, axis=-1)
        mask_colored = np.zeros_like(original_rgb)
        mask_colored[..., 1] = combined_mask * 255
        return ((1 - 0.5) * original_rgb + 0.5 * mask_colored).astype(np.uint8)

    def _check_rotation(self, left_mask: np.ndarray, right_mask: np.ndarray) -> tuple[str, Optional[HoughAngles]]:
        angle_left = self._calculate_orientation(left_mask)
        angle_right = self._calculate_orientation(right_mask)
        if angle_left is None or angle_right is None:
            return "⚠️ No clavicles detected", None
        angle_left = abs(angle_left if angle_left <= 90 else angle_left - 180)
        angle_right = abs(angle_right if angle_right <= 90 else angle_right - 180)
        angle_diff = abs(angle_left - angle_right)
        status = "Correct rotation ✅" if angle_diff <= 10 else "Incorrect rotation 🔄"
        return status, HoughAngles(angle_left, angle_right, angle_diff)

    @staticmethod
    def _calculate_orientation(mask: np.ndarray) -> Optional[float]:
        edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=20)
        if lines is None:
            return None
        angles = [line[0][1] for line in lines]
        return float(np.mean(angles) * (180 / np.pi))

    # YOLO overlays ------------------------------------------------------
    def _compose_yolo_overlay(self, image: Image.Image, *results) -> Optional[np.ndarray]:
        if not results:
            return None
        base = np.array(image).copy()
        for result in results:
            if not hasattr(result, "plot"):
                continue
            plotted = result.plot()
            if plotted is not None:
                rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                base = cv2.addWeighted(base, 0.5, rgb, 0.5, 0)
        return base

    # Cardio-thoracic ratio ----------------------------------------------
    @staticmethod
    def _compute_ctr(rotation_data: "ImageProcessor._RotationData", image: Image.Image) -> tuple[float, str, Optional[np.ndarray]]:
        heart_mask = rotation_data.heart_mask
        diaphragm_mask = rotation_data.diaphragm_mask
        if heart_mask is None or diaphragm_mask is None:
            return 0.0, "No structures detected", None
        heart_width = ImageProcessor._max_x_distance(heart_mask)
        chest_width = ImageProcessor._max_x_distance(diaphragm_mask)
        ctr_ratio = heart_width / chest_width if chest_width > 0 else 0
        ctr_result_text = "🟢 Normal CTR (≤ 0.5)" if ctr_ratio <= 0.5 else "🔴 Abnormal CTR (> 0.5)"
        overlay = np.zeros((512, 512, 3), dtype=np.uint8)
        overlay[diaphragm_mask > 0] = [0, 255, 0]
        overlay[heart_mask > 0] = [255, 0, 0]
        original = np.array(image.convert("RGB").resize((512, 512)))
        overlay_ctr = cv2.addWeighted(original, 0.5, overlay, 0.5, 0)
        return ctr_ratio, ctr_result_text, overlay_ctr

    @staticmethod
    def _max_x_distance(mask: np.ndarray) -> int:
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return 0
        min_x = np.min(coords[:, 1])
        max_x = np.max(coords[:, 1])
        return int(max_x - min_x)

    # Rotation balance ----------------------------------------------------
    def _compute_rotation_balance(
        self,
        rotation_data: "ImageProcessor._RotationData",
        image: Image.Image,
    ) -> tuple[str, Optional[np.ndarray]]:
        left_mask = rotation_data.left_mask
        right_mask = rotation_data.right_mask
        spine_mask = rotation_data.spine_mask
        if left_mask is None or right_mask is None or spine_mask is None:
            return "❗️ No structures detected", None

        left_medial = self._get_medial_point(left_mask, side="left")
        right_medial = self._get_medial_point(right_mask, side="right")
        spine_centroid = self._get_centroid(spine_mask, top_percent=10)
        if left_medial is None or right_medial is None or spine_centroid is None:
            return "❗️ No structures detected", None

        spine_x = int(spine_centroid[0])
        img_height, img_width = spine_mask.shape
        combined_overlay = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        combined_overlay[spine_mask > 0] = [0, 255, 0]
        combined_overlay[left_mask > 0] = [255, 0, 0]
        combined_overlay[right_mask > 0] = [0, 0, 255]

        d_left = self._draw_and_measure(combined_overlay, left_medial, spine_x, (255, 0, 0))
        d_right = self._draw_and_measure(combined_overlay, right_medial, spine_x, (0, 0, 255))
        diff = abs(d_left - d_right)
        media = (d_left + d_right) / 2
        tolerancia = 0.1 * media
        rotation_result = "🟢 Correct Rotation" if diff <= tolerancia else "🔴 Incorrect rotation"

        overlay_combined = cv2.addWeighted(
            np.array(image.resize((img_width, img_height))), 0.5, combined_overlay, 0.5, 0
        )
        return rotation_result, overlay_combined

    @staticmethod
    def _get_medial_point(mask: np.ndarray, side: str = "left") -> Optional[np.ndarray]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        if side == "left":
            idx = np.argmin(xs)
        else:
            idx = np.argmax(xs)
        return np.array([xs[idx], ys[idx]])

    @staticmethod
    def _get_centroid(mask: np.ndarray, top_percent: Optional[float] = None) -> Optional[np.ndarray]:
        labeled = label(mask)
        regions = regionprops(labeled)
        if not regions:
            return None
        largest = max(regions, key=lambda r: r.area)
        coords = largest.coords
        if top_percent:
            y_cutoff = np.percentile(coords[:, 0], top_percent)
            top_coords = coords[coords[:, 0] <= y_cutoff]
            if len(top_coords) == 0:
                return None
            centroid_y = np.mean(top_coords[:, 0])
            centroid_x = np.mean(top_coords[:, 1])
        else:
            centroid_y, centroid_x = largest.centroid
        return np.array([centroid_x, centroid_y])

    @staticmethod
    def _draw_and_measure(img: np.ndarray, point: np.ndarray, spine_x: int, color: tuple[int, int, int]) -> float:
        cx, cy = int(point[0]), int(point[1])
        cv2.circle(img, (cx, cy), 6, color, -1)
        cv2.line(img, (cx, cy), (spine_x, cy), color, 2)
        return abs(cx - spine_x)

    # Scapula -------------------------------------------------------------
    def _analyse_scapula(
        self,
        rotation_data: "ImageProcessor._RotationData",
        lung_mask: np.ndarray,
        image: Image.Image,
    ) -> tuple[ScapulaOverlap, Optional[np.ndarray]]:
        left_mask = rotation_data.left_scapula
        right_mask = rotation_data.right_scapula
        if left_mask is None or right_mask is None:
            return ScapulaOverlap(0.0, 0.0), None

        left_bin = (left_mask > 0).astype(np.uint8)
        right_bin = (right_mask > 0).astype(np.uint8)
        lung_bin = (lung_mask > 0).astype(np.uint8)

        intersection_left = np.sum(left_bin * lung_bin)
        intersection_right = np.sum(right_bin * lung_bin)
        left_area = np.sum(left_bin)
        right_area = np.sum(right_bin)

        percent_left = (intersection_left / left_area * 100) if left_area > 0 else 0
        percent_right = (intersection_right / right_area * 100) if right_area > 0 else 0

        original_np = np.array(image.resize(left_bin.shape[::-1]))
        if original_np.ndim == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2BGR)
        overlay = original_np.copy()
        overlay[left_bin > 0] = [255, 0, 0]
        overlay[right_bin > 0] = [0, 0, 255]
        combined = cv2.addWeighted(original_np, 0.6, overlay, 0.4, 0)

        return ScapulaOverlap(percent_left, percent_right), combined

    # Final score ---------------------------------------------------------
    def _derive_final_score(
        self,
        inclusion_label: str,
        orientation_status: Optional[str],
        rotation_balance: str,
        ctr_ratio: float,
        artifacts_result_1,
        artifacts_result_2,
    ) -> str:
        if inclusion_label == "INCLUSIÓN NO APTA ❌":
            return "🔴  BAD!  🔴"
        if inclusion_label == "INCLUSIÓN APTA ✅":
            orientation_ok = orientation_status == "Correct rotation ✅"
            balance_ok = rotation_balance == "🟢 Correct Rotation"
            ctr_ok = ctr_ratio <= 0.5
            artifacts_ok = len(getattr(artifacts_result_1, "boxes", [])) == 0 and len(getattr(artifacts_result_2, "boxes", [])) == 0
            if orientation_ok and balance_ok and ctr_ok and artifacts_ok:
                return "🟢  All conditions are met  🟢"
            if orientation_ok and balance_ok:
                return "🟡 Revision Required  🟡"
            return "🟠 Issues Detected  🟠"
        return "❓ Pending"

    # Utilities -----------------------------------------------------------
    @staticmethod
    def _to_bgr(image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

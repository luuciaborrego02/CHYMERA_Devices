"""Lazy loading helpers for ML models used by the analyser."""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Tuple

import torch
import tensorflow as tf
import torchxrayvision as xrv
from ultralytics import YOLO


class ModelLoadError(RuntimeError):
    """Raised when the required ML models cannot be loaded."""


@functools.lru_cache(maxsize=1)
def load_models(base_path: Path) -> Tuple[torch.nn.Module, tf.keras.Model, YOLO, YOLO, torch.nn.Module, torch.device]:
    """Load all ML models using paths relative to ``base_path``.

    Parameters
    ----------
    base_path:
        Base directory where the ``models`` and ``runs`` folders live. This is
        typically the repository root.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_path = base_path / "models" / "unet-6v.pt"
    cnn_path = base_path / "models" / "cnn_model.h5"
    yolo_1_path = base_path / "runs" / "detect" / "yolov8-chest-xray14" / "weights" / "best.pt"
    yolo_2_path = base_path / "runs" / "detect" / "yolov8-chest-xray24" / "weights" / "best.pt"

    for required in (unet_path, cnn_path, yolo_1_path, yolo_2_path):
        if not required.exists():
            raise ModelLoadError(f"Missing model weight: {required}")

    try:
        from src.models import PretrainedUNet
    except ModuleNotFoundError:
        sys.path.append(str(base_path))
        try:
            from src.models import PretrainedUNet
        except ModuleNotFoundError as inner_exc:  # pragma: no cover - defensive
            raise ModelLoadError("Unable to import src.models.PretrainedUNet") from inner_exc

    unet = PretrainedUNet(in_channels=1, out_channels=2, batch_norm=True, upscale_mode="bilinear")
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.to(device).eval()

    try:
        clf = tf.keras.models.load_model(cnn_path, compile=False)
    except ValueError as exc:
        if "batch_shape" not in str(exc):
            raise ModelLoadError(str(exc)) from exc
        import h5py

        with h5py.File(cnn_path, "r") as f:
            if "model_config" not in f.attrs:
                raise ModelLoadError("Invalid CNN model format") from exc
            model_config = f.attrs["model_config"]
            if isinstance(model_config, bytes):
                model_config = model_config.decode("utf-8")
            repaired = model_config.replace('"batch_shape"', '"batch_input_shape"')
            custom_objects = {
                "DTypePolicy": tf.keras.mixed_precision.Policy,
                "Policy": tf.keras.mixed_precision.Policy,
            }
            clf = tf.keras.models.model_from_json(repaired, custom_objects=custom_objects)
            clf.load_weights(cnn_path)

    yolo_model_1 = YOLO(str(yolo_1_path))
    yolo_model_2 = YOLO(str(yolo_2_path))
    clavicle_model = xrv.baseline_models.chestx_det.PSPNet()

    return unet, clf, yolo_model_1, yolo_model_2, clavicle_model, device

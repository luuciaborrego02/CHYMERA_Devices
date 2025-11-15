# Quality Chest Analyzer (Qt Edition)

This repository hosts a desktop reimplementation of the original Streamlit-based
Quality Chest Analyzer using **PySide6 (Qt for Python)**. The goal is to provide
an offline-friendly, production-ready interface that can be distributed both as
a standalone executable and as a Docker container.

## Features

- Dark-themed Qt interface with dedicated panels for all analysis modalities
  (inclusion, rotations, artifacts, CTR, scapula overlap and global statistics).
- Modular business logic that mirrors the original Streamlit pipeline, including
  UNet segmentation, YOLO artifact detection and cardio-thoracic ratio
  estimation.
- Grad-CAM visualisation for the UNet model (when `torchcam` is available).
- Aggregated statistics persisted across the current session.

> **Note**: model weights are not bundled with this repository. Place your
> existing `models/` and `runs/` directories at the repository root before
> launching the application.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
chym-analyzer
```

The entry point opens the Qt GUI. Use the toolbar to load one or more X-ray
images and trigger the analyses.

## Building a standalone executable

The project is ready for PyInstaller. After installing the dependencies:

```bash
pyinstaller --name QualityChestAnalyzer --noconfirm --windowed --collect-data torchxrayvision chym_app/main.py
```

The resulting `dist/QualityChestAnalyzer` directory can be zipped and shared.
Consider adding your `models/` and `runs/` folders next to the executable.

## Docker image

A simple Dockerfile is provided to package the Qt application with Xvfb. Build
and run it as follows:

```bash
docker build -t chym-analyzer .
docker run --rm -e DISPLAY=:99 -p 5901:5901 chym-analyzer
```

The container launches a virtual display (VNC on port `5901`) so the Qt
application can be accessed remotely via any VNC client.

## Repository layout

```
chym_app/
├── config.py              # Theme palette and app constants
├── core/
│   ├── image_processor.py # Image analysis orchestration
│   └── model_loader.py    # Lazy loading helpers for ML models
├── ui/
│   └── main_window.py     # Main Qt window and navigation
└── widgets/
    └── analysis_views.py  # Reusable widgets for each analysis panel
```

## Environment variables

The application no longer embeds any API keys. If you wish to connect to the
OpenAI API for conversational assistance, export your key in the standard
`OPENAI_API_KEY` environment variable and extend the UI accordingly.

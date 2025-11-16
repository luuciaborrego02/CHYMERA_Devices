# Quality Chest Analyzer

AI-assisted toolkit for evaluating chest X-ray quality. The desktop Qt app (`chym_app/`) delivers the full workflow for loading studies, running the ML pipeline, visualising overlays, dashboards, and chatting with the ChymAI assistant.

All processing happens locally using the bundled PyTorch, TensorFlow, YOLO and TorchXRayVision models. Optional LLM assistance is powered by OpenAI via environment variables or a secure proxy server (recommended).

---

## Highlights

- Lung inclusion, clavicle rotation, artifacts, CTR, rotation balance, and scapula overlap checks.
- Overlay visualisations, interactive Plotly dashboards.
- Global statistics panel with donut + parameter bar chart (auto-falls back to static figure when WebEngine is unavailable).
- ChymAI assistant dock for contextual guidance inside the desktop UI.
- Ready for packaging into a standalone `.exe` via PyInstaller.

---

## Requirements

- Windows 10/11 64-bit (other OS may work but not tested).
- Python 3.10+ (virtual environment strongly recommended).
- GPU optional; CPU inference works albeit slower.
- Model weights located under `models/` and `runs/` as provided.

Python dependencies are declared in `pyproject.toml` (PySide6, torch, torchvision, tensorflow, torchxrayvision, ultralytics, OpenAI SDK, etc.).

---

## Project Layout

```
CHYMERA_Devices/
├── chym_app/              # Qt desktop application
│   ├── core/              # Image processor, assistant wrapper
│   ├── ui/                # Main window
│   ├── widgets/           # Analysis views, chat panel
│   └── main.py            # Entry point
├── models/                # UNet/CNN weights (not tracked in git)
├── runs/detect/           # YOLO weights
├── Dockerfile             # Optional container build
├── pyproject.toml         # Dependencies & packaging metadata
└── README.md
```

---

## Setup

```powershell
cd C:\Users\lucia\CHYMERA_Devices
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

This installs the desktop app as editable (`chym-analyzer` CLI becomes available) and fetches all required libraries.

Ensure the `models/` and `runs/` directories contain the expected weights (UNet, Keras classifier, YOLO checkpoints, etc.).

---

## Running the Qt Desktop App

```powershell
.\.venv\Scripts\activate
python -m chym_app.main
```

or via the installed console script:

```powershell
chym-analyzer
```

### Features inside the app

- **Toolbar** – load one or many chest X-rays and run the analysis pipeline.
- **Sidebar** – pick an image and toggle between Inclusion, Rotation, Artifacts, Rotation Balance, CTR, Scapula, or Global Statistics views.
- **Main view** – responsive overlay canvas plus textual explanation for the selected analysis.
- **Footer** – status log and final score for the currently selected image.
- **ChymAI dock** – chat assistant (hidden by default if credentials are missing; toggle via `View → ChymAI Panel`).

---

## Configuring the LLM Assistant

### ✅ Option 1 — Use your API key through a server (recommended)

Never embed your `OPENAI_API_KEY` in client code or binaries. Instead:

1. Deploy a lightweight backend (FastAPI/Flask/cloud function) that exposes an endpoint (e.g. `/assistant`).
2. The backend holds the OpenAI key, receives prompts from the client, calls OpenAI, and returns the response.
3. The Qt front-end communicates only with your backend; the key remains safe.

Example FastAPI skeleton:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
app = FastAPI()

class Prompt(BaseModel):
	message: str

@app.post("/assistant")
def ask_agent(payload: Prompt):
	result = client.responses.create(
		model="gpt-4.1",
		input=payload.message,
	)
	return {"reply": result.output_text}
```

Qt client snippet:

```python
import requests

resp = requests.post(
    "https://your-backend.com/assistant",
    json={"message": prompt_text},
    timeout=30,
)
agent_reply = resp.json()["reply"]
```

### Option 2 — Local environment variables (for development only)

If you prefer direct OpenAI access during local testing, place your credentials in environment variables **before launching** the app:

```powershell
setx OPENAI_API_KEY "sk-..."
setx OPENAI_ASSISTANT_ID "asst_..."
# open a new PowerShell window afterwards
python -m chym_app.main
```

The ChymAI dock enables itself only when both variables are available. Never commit `.env` files or hard-coded keys to the repository.

Need access or have questions? Contact the authors at `lborrego@santpau.cat` for guidance on provisioning the assistant.

---

## Packaging to a Windows Executable

Bundling the entire desktop app (including models and Qt) into an `.exe` can be done with PyInstaller:

```powershell
.\.venv\Scripts\activate
pyinstaller chym_app\main.py ^
  --name ChymAnalyzer ^
  --noconsole ^
  --icon assets\app.ico ^
  --add-data "models;models" ^
  --add-data "runs;runs" ^
  --collect-all PySide6 ^
  --collect-all torch ^
  --hidden-import torchxrayvision
```

Customise `--add-data` entries to include every directory/file the app reads at runtime. After the first run, edit the generated `.spec` file for fine-grained control, then re-run `pyinstaller ChymAnalyzer.spec`.

Distribute the resulting folder under `dist/`. Provide a small PowerShell launcher that prompts users for their OpenAI credentials (or configure the backend proxy mentioned above).

---

## Troubleshooting

- **ChymAI dock says “missing OpenAI credentials”** – confirm `OPENAI_API_KEY` and `OPENAI_ASSISTANT_ID` exist in the environment of the shell used to launch the app (`echo $Env:VAR` in PowerShell).
- **Plotly chart missing** – install `PySide6-Addons` for WebEngine; otherwise the UI falls back to the static PNG.
- **PyInstaller exe crashes at start** – ensure all model folders were added via `--add-data` and CUDA DLLs (if any) are accessible. Try a non-`--onefile` build first.
- **Large model files absent on another machine** – include `models/` and `runs/` alongside the exe or host them for download.

---

## Contributing / Notes

- Keep secrets out of git (use `.gitignore` for `.venv/`, `dist/`, etc.).
- The ML models are sensitive to directory structure—maintain the same relative paths when copying or packaging.
- Issues and enhancements welcome through GitHub pull requests.

---

Built with ❤️ by CHYMERA for automated chest X-ray quality assessment.



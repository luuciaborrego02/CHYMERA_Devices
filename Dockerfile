FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    QT_QPA_PLATFORM=offscreen \
    DISPLAY=:99

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb x11vnc fluxbox && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY chym_app ./chym_app

RUN pip install --upgrade pip && pip install . && \
    pip install pyinstaller

EXPOSE 5901

CMD ["/bin/bash", "-c", "xvfb-run -s '-screen 0 1920x1080x24' x11vnc -forever -usepw -shared & fluxbox & chym-analyzer"]

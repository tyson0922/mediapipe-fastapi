# Use slim Debian + Python 3.10 (MediaPipe wheels are built for this combo)
FROM python:3.10-slim-bookworm

# --- Runtime hygiene & speed ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- System libs needed by OpenCV/MediaPipe runtime ---
# Keep libgl1 + libglib2.0-0; remove ffmpeg entirely
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install Python deps first (best cache hit)
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# 2) Download MediaPipe models
RUN mkdir -p /app/models && \
    curl -fsSL -o /app/models/face_landmarker_v2_with_blendshapes.task \
    https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2_with_blendshapes.task

# 3) App code (bind-mounted in dev; copied for completeness)
COPY app/ ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
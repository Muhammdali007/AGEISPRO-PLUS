FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_AUTOINSTALL=false
ENV PIP_DEFAULT_TIMEOUT=600
ENV PIP_RETRIES=10

WORKDIR /app
ARG INSTALL_EXTRAS=model,recognition
ARG INSTALL_SYSTEM_PACKAGES=true

RUN if [ "$INSTALL_SYSTEM_PACKAGES" = "true" ]; then \
    apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*; \
  fi

COPY pyproject.toml ./
COPY app ./app

RUN if [ -n "$INSTALL_EXTRAS" ]; then \
    pip install --no-cache-dir --default-timeout=600 --retries=10 -e ".[$INSTALL_EXTRAS]"; \
  else \
    pip install --no-cache-dir --default-timeout=600 --retries=10 -e .; \
  fi

EXPOSE 8100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]

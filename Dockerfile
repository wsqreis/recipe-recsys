# syntax=docker/dockerfile:1
# Build with TORCH=cpu (default) or TORCH=cuda. The CUDA variant ships the CUDA
# runtime inside the PyTorch wheels, so the base image stays the same; the GPU
# driver comes from the host (NVIDIA Container Toolkit / Docker Desktop + WSL2).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

ARG TORCH=cpu
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # Keep the Kaggle download and the text model weights inside the mounted data volume.
    KAGGLEHUB_CACHE=/app/data/raw \
    HF_HOME=/app/data/hf \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall PyTorch.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --extra ${TORCH} --extra text --extra api

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra ${TORCH} --extra text --extra api

ENTRYPOINT ["recsys"]
CMD ["--help"]

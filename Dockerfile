FROM python:3.12-slim-bookworm

# Install Astral uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="/app/model/best_model.pkl"

# Install project dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source code and trained model artifacts
COPY src/ ./src/
COPY model/ ./model/

EXPOSE 9696

ENTRYPOINT ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "9696"]

FROM python:3.12.1-slim-bookworm

RUN pip install uv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY ".python-version" "pyproject.toml" "uv.lock" "./"
RUN uv sync --locked

COPY src/ ./src/
COPY model/model.bin ./model/model.bin

ENV MODEL_PATH="/app/model/model.bin"

EXPOSE 9696

ENTRYPOINT ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "9696"]


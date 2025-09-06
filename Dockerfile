FROM python:3.9-slim-bookworm

RUN pip install uv 

WORKDIR /app

COPY "pyproject.toml", "uv.lock", "./"

RUN uv sync --locked

COPY "predict.py", "model.bin", "./"

EXPOSE 8080

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]

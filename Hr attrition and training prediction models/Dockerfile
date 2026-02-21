FROM python:3.11-slim

WORKDIR /app

# Install uv for faster dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application files
COPY app.py ./
COPY *.joblib ./
COPY combined.csv ./

# Create non-root user
RUN useradd -r -U app && \
    chown -R app:app /app

EXPOSE 7860

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
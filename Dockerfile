# Multi-Stage Production Dockerfile for ShopRank
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Builder stage
# Builder stage
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --default-timeout=1000 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --prefix=/install -r requirements.txt

# Final runtime image
FROM python:3.11-slim as runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages to /usr/local
COPY --from=builder /install /usr/local
ENV PYTHONPATH=/app

# Create unprivileged application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy application source tree, configurations, models, and data
COPY src/ /app/src/
COPY dags/ /app/dags/
COPY deploy/ /app/deploy/
COPY params.yaml /app/params.yaml
COPY requirements.txt /app/requirements.txt
COPY models/ /app/models/
COPY data/ /app/data/

# Ensure permissions
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

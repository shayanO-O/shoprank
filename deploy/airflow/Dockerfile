FROM apache/airflow:2.8.2-python3.11

USER root
# Install system libraries needed by LightGBM and FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --default-timeout=1000 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "email-validator>=2.0.0" \
        -r /requirements.txt

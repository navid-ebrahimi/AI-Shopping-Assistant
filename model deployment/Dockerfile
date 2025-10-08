FROM hub.hamdocker.ir/python:3.12-slim

ENV HF_HOME=/var/lib/data/huggingface \
    TRANSFORMERS_CACHE=/var/lib/data/transformers \
    HF_DATASETS_CACHE=/var/lib/data/datasets \
    DATA_DIR=/var/lib/data \
    TMPDIR=/var/lib/data/tmp \
    TEMP=/var/lib/data/tmp \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN mkdir -p /var/lib/data/{tmp,huggingface,transformers,datasets} && \
    chmod -R 755 /var/lib/data

WORKDIR /app
VOLUME ["/var/lib/data"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    libevent-dev \
    python3-psycopg2 \
    python3-dev \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

COPY ./requirements.txt ./
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

COPY ./ ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
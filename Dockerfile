FROM python:3.12.3-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY pytest.ini ./
COPY hts_bench/ hts_bench/
COPY scripts/ scripts/
COPY tests/ tests/
COPY dataset/ dataset/

RUN pytest tests/ -q

ENTRYPOINT ["python", "scripts/run_benchmark.py"]
